import os
import warnings
import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.image import PeakSignalNoiseRatio

from phantoms.foam_generator import FoamGenerator
from phantoms.sparse_generator import SparseGenerator
from sinogram.sinogram_generator import Sinogram
from noise2inverse.networks.unet import UNet
from noise2inverse.networks.dncnn import DnCNN

# Suppress all warnings
warnings.filterwarnings('ignore')

ATTENUATION = 200

####################################################################################################
#                                               CLASS                                              #
####################################################################################################

class N2I:
    def __init__(self, phantom, network_name: str = "unet", device="cpu", num_splits: int = 4, strategy="X:1",
                 lr: float = 0.001, bs: int = 16, epochs: int = 30, vmax: float = 1, comment: str = ""):
        """
        Initialize the N2I class with the sinogram.
        This class recives the sinogram and has two methods, for training
        and for evaluation.

        :param sinogram: The sinogram in which the model will be trained
        """
        # Set parameter values
        os.makedirs("runs", exist_ok=True)
        current_runs = len(os.listdir("runs"))
        self.id = f"{datetime.date.today()}_run{current_runs+1}" + f"_{comment}" if comment else ""
        self.dir = f"runs/{self.id}"
        os.makedirs(self.dir, exist_ok=True)
        self.num_splits = num_splits
        self.network = None
        self.optimizer = None
        self.device = device
        self.network_name = network_name
        self.strategy = strategy

        # Initialize the network
        self._initialize_network(self.network_name)

        # Train parameters
        self.epochs = epochs
        self.lr = lr
        self.bs = bs
        self.save_nn_settings()

        # Phantom
        self.phantom = phantom
        np.save(f"{self.dir}/phantom.npy", self.phantom)

        # Plotting
        self.vmax = vmax

    def save_nn_settings(self):
        with open(f"{self.dir}/settings.txt", "w") as f:
            f.write(f"Network: {self.network_name}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Num splits: {self.num_splits}\n")
            f.write(f"Strategy: {self.strategy}\n")
            f.write(f"Learning rate: {self.lr}\n")
            f.write(f"Batch size: {self.bs}\n")
            f.write(f"Epochs: {self.epochs}\n")


    def generate_source_target_for_split(self, split_recs, num_split):
        """
            This functions generates the source and terget split in 
            order to perform a reconstruction.

            :param split_recs: The reconstructions of the splits
            :param num_split: Number of splits per reconstruction

            :return: Source and target reconstructions
        """
        # Initiallize values
        source_rec = []
        target_rec = []
        list_indeces = list(range(len(split_recs)))
        list_indeces.remove(num_split)

        # Depending on the strategy we will perform a different split
        if self.strategy == "X:1":
            source_rec = np.mean(np.array([split_recs[i] for i in list_indeces]), axis=0)
            target_rec = np.array(split_recs[num_split])

        elif self.strategy == "1:X":
            source_rec = np.array(split_recs[num_split])
            target_rec = np.mean(np.array([split_recs[i] for i in list_indeces]), axis=0)

        # If the strategy is invalid, return error
        else:
            raise ValueError("Invalid strategy. Please use 'X:1' or '1:X'.")
        
        # Convert to tensor floats
        source_rec = torch.tensor(source_rec).float()
        target_rec = torch.tensor(target_rec).float()

        return source_rec, target_rec
    
    def plot_evaluation_evolution(self, evaluation_results: dict, source_image, target_image):
        num_evals = len(evaluation_results)
        plt.figure()
        axs = plt.subplots(num_evals+2, figsize=(4.5*(num_evals+2), 5))
        axs = axs.flatten()
        axs[0].imshow(source_image, cmap='gray', vmin=0, vmax=self.vmax)
        axs[0].axis('off')
        axs[0].set_title("Noisy reconstruction")
        i = 1
        for epoch, eval_res in evaluation_results.items():
            axs[i].imshow(eval_res, cmap='gray', vmin=0, vmax=self.vmax)
            axs[i].axis('off')
            axs[i].set_title(f"Epoch {epoch}")
            i += 1
        axs[-1].imshow(target_image, cmap='gray', vmin=0, vmax=self.vmax)
        axs[-1].axis('off')
        axs[-1].set_title("Phantom")

    def plot_status(self, output, target, epoch, k_split=None, eval=False, sl: int = 128):
        plt.figure()
        plt.subplot(1, 2, 1)
        VMAX = 1
        plt.imshow(output[sl].cpu().detach(), cmap='gray', vmin=0, vmax=VMAX)
        plt.title("Network output")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(target[sl].cpu().detach(), cmap='gray', vmin=0, vmax=VMAX)
        plt.title("Network target")
        plt.axis('off')
        if eval:
            os.makedirs(f"{self.dir}/figures_eval", exist_ok=True)
            plt.savefig(f"{self.dir}/figures_eval/ep{epoch}eval.png", dpi=300)
        else:
            os.makedirs(f"{self.dir}/figures_train", exist_ok=True)
            plt.savefig(f"{self.dir}/figures_train/ep{epoch}_{k_split}split.png", dpi=300)

    def plot_training_losses(self, losses_split, eval_losses):
        plt.figure(figsize=(12, 5))

        # Evaluation loss
        plt.subplot(1, 2, 1)
        plt.title("Evaluation loss")
        plt.plot(np.concatenate([np.array([1]), np.arange(0, self.epochs + 1, 5)[1:]]), eval_losses)
        plt.grid()
        plt.yscale("log")
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")

        # Training losses
        plt.subplot(1, 2, 2)
        for k in range(self.num_splits):
            plt.plot(losses_split[k], label=f"Split {k}")
        plt.grid()
        plt.title(f"Training losses")
        plt.yscale("log")
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.tight_layout()
        
        os.makedirs(f"{self.dir}/losses", exist_ok=True)
        plt.savefig(f"{self.dir}/losses/losses.png", dpi=300)

    def compute_psnr(self, input, noisy_reconstruction):
        """
        Compute the Peak Signal to Noise Ratio (PSNR) between the input and target images.
        :param input: The input image
        :param target: The target image
        """
        psnr_method = PeakSignalNoiseRatio()
        psnr_denoised = psnr_method(input.cpu(), torch.tensor(self.phantom))
        psnr_noisy = psnr_method(torch.tensor(noisy_reconstruction), torch.tensor(self.phantom))

        print(f"Denoised PSNR: {psnr_denoised}")
        print(f"Noisy PSNR: {psnr_noisy}")
        with open(f"{self.dir}/psnr.txt", "w") as f:
            f.write(f"PSNR Denoised: {psnr_denoised}")
            f.write(f"PSNR Noisy: {psnr_noisy}")

    def Train(self, rec_input, noisy_reconstruction):
        """
        Train the model with the provided training data.

        :param rec_input: Input data of the reconstruction for trianing the network
        :param epochs: Number of epochs to train the model
        :param batch_size: Size of the batches for each training step
        :param learning_rate: Learning rate for the optimizer
        """

        # Get the optimizer and Scaler
        if self.network_name != "msd":
            optimizer = self._get_optimizer(self.lr)
        # scaler = torch.cuda.amp.GradScaler()  # Use GradScaler for mixed precision training

        # Make sure the path for the weights exists
        eval_losses = []
        losses_split = {}
        evaluation_results = {}

        # Epoch training loop
        for epoch in range(1, self.epochs+1):
            self.network.train()
            epoch_loss = 0
            for k in range(self.num_splits):
                slices_pred = []
                source_recs, target_recs = self.generate_source_target_for_split(rec_input, k)
                for i in range(0, source_recs.shape[0], self.bs):
                    batch_slices = source_recs[i:i+self.bs, None, ...].to(self.device, dtype=torch.float32)
                    
                    batch_predictions = self.network(batch_slices)
                    slices_pred.append(batch_predictions.squeeze().cpu())
                    torch.cuda.empty_cache()

                slices_pred = torch.cat(slices_pred, dim=0).to(self.device, dtype=torch.float32)
                if epoch % 20 == 0:
                    self.plot_status(slices_pred, target_recs, epoch, k, eval=False)
                
                target_recs = target_recs.to(self.device, dtype=torch.float32)
                loss = torch.nn.functional.mse_loss(slices_pred, target_recs)
                epoch_loss += loss.item()

                # Update optimizer and scaler
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if epoch == 1:
                    losses_split[k] = [loss.item()]
                else:
                    losses_split[k].append(loss.item())

            # Print progress and loss and save network weights each epoch
            print(f"Epoch {epoch} / {self.epochs} | Loss: {epoch_loss}")
            torch.save(self.network.state_dict(), f"{self.dir}/weights.pth")
            if epoch % 5 == 0 or epoch == 1:
                denoised_image = self.Evaluate(rec_input, noisy_reconstruction, psnr=False)
                with torch.no_grad():
                    tensor_original = torch.tensor(self.phantom).float().to(self.device)
                    loss_eval = torch.nn.functional.mse_loss(denoised_image, tensor_original)
                    eval_losses.append(loss_eval.item())
                    # evaluation_results[epoch] = denoised_image.cpu().numpy()

                    self.plot_status(denoised_image, tensor_original, epoch, eval=True)

        # Plot losses and prediction evolution
        self.plot_training_losses(losses_split, eval_losses)
        # self.plot_evaluation_evolution(evaluation_results, source_image=noisy_reconstruction, target_image=self.phantom)
        

    def Evaluate(self, rec_input, noisy_reconstruction, psnr=True):
        """
        Extract from the denoised reconstruction from the network.
        
        :param rec_input: Input data of the reconstruction to evaluate
        :param batch_size: Size of the batches for each training step

        :return: Denoised image of the pahntom
        """

        self.network.eval()
        torch.cuda.empty_cache()
        pred_splits = []

        with torch.no_grad():
            for im in rec_input:
                im = torch.tensor(im).float()
                slices_pred = []
                for i in range(0, im.shape[1], self.bs):
                    batch_slices = im[i:i+self.bs, None, ...].to(self.device, dtype=torch.float32)
                    batch_predictions = self.network(batch_slices)
                    slices_pred.append(batch_predictions.squeeze())

                slices_pred = torch.cat(slices_pred, dim=0)
                pred_splits.append(slices_pred)

            pred_splits = torch.stack(pred_splits).to(self.device, dtype=torch.float32)
            denoised_image = torch.mean(pred_splits, dim=0)
            if psnr:
                self.compute_psnr(denoised_image, noisy_reconstruction)

        return denoised_image
        

    def load_weights(self, run_id):
        """
            Loads saved weights from the desired path.
        """
        self.network.load_state_dict(torch.load(f"runs/{run_id}/weights.pth"))


    def _initialize_network(self, network_name):
        """
        Helper function to initialize the neural network.

        :param network_name: The name of the network to use
        """
        if network_name == "unet":
            self.network = UNet(1, 1, n_features=8).to(self.device)
        elif network_name == "dncnn":
            self.network = DnCNN(1, num_of_layers=8).to(self.device)
        else:
            raise ValueError(f"Invalid network name: {network_name}")


    def _get_optimizer(self, learning_rate=0.001):
        """
        Helper function to create an optimizer. (Adam)

        :param learning_rate: Learning rate for the optimizer

        :return: An optimizer instance
        """
        return torch.optim.Adam(self.network.parameters(), lr=learning_rate)
