import os
import warnings
import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt

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
                 lr: float = 0.001, bs: int = 16, epochs: int = 30):
        """
        Initialize the N2I class with the sinogram.
        This class recives the sinogram and has two methods, for training
        and for evaluation.

        :param sinogram: The sinogram in which the model will be trained
        """
        # Set parameter values
        os.makedirs("runs", exist_ok=True)
        current_runs = len(os.listdir("runs"))
        self.id = f"{datetime.date.today()}_run{current_runs+1}"
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
            raise ValueError("Invalid source_imgs value")
        
        # Convert to tensor floats
        source_rec = torch.tensor(source_rec).float()
        target_rec = torch.tensor(target_rec).float()

        return source_rec, target_rec
    

    def plot_status(self, output, target, epoch, k_split=None, eval=False, sl: int = 128):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(output[sl].cpu().detach(), cmap='gray', vmin=0, vmax=1/ATTENUATION)
        plt.title("Network output")
        plt.subplot(1, 2, 2)
        plt.imshow(target[sl].cpu().detach(), cmap='gray', vmin=0, vmax=1/ATTENUATION)
        plt.title("Network target")
        if eval:
            os.makedirs(f"{self.dir}/figures_eval", exist_ok=True)
            plt.savefig(f"{self.dir}/figures_eval/ep{epoch}eval.png", dpi=300)
        else:
            os.makedirs(f"{self.dir}/figures_train", exist_ok=True)
            plt.savefig(f"{self.dir}/figures_train/ep{epoch}_{k_split}split.png", dpi=300)

    def plot_training_losses(self, losses_split, epoch_losses, eval_losses):
        plt.figure()
        for k in range(self.num_splits):
            plt.plot(losses_split[k], label=f"Split {k}")
        plt.plot(np.array(epoch_losses)/self.num_splits, label="Split avg")
        plt.plot(np.arange(0, self.epochs, 10), eval_losses, label="Evaluation")
        plt.title(f"Losses {self.network_name} - lr: {self.lr} epochs: {self.epochs}")
        plt.yscale("log")
        plt.legend()
        os.makedirs(f"{self.dir}/losses", exist_ok=True)
        plt.savefig(f"{self.dir}/losses/losses_splits_{self.network_name}_lr{self.lr}_epochs{self.epochs}.png", dpi=300)


    def Train(self, rec_input, original_image=None):
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
        epoch_losses = []
        eval_losses = []
        losses_split = {}

        # Epoch training loop
        for epoch in range(self.epochs):
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
                    import pdb; pdb.set_trace()
                    self.plot_status(slices_pred, target_recs, epoch, k, eval=False)
                
                target_recs = target_recs.to(self.device, dtype=torch.float32)
                # loss = (slices_pred - target_recs).mean()
                loss = torch.nn.functional.l1_loss(slices_pred, target_recs)
                epoch_loss += loss.item()

                # Update optimizer and scaler
                optimizer.zero_grad()
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()
                loss.backward()
                optimizer.step()
                if epoch == 0:
                    losses_split[k] = [loss.item()]
                else:
                    losses_split[k].append(loss.item())

            # Print progress and loss and save network weights each epoch
            print(f"Epoch {epoch+1} / {self.epochs} | Loss: {epoch_loss}")
            torch.save(self.network.state_dict(), f"{self.dir}/weights.pth")
            epoch_losses.append(epoch_loss)
            if epoch % 10 == 0:
                assert original_image is not None, "Original image is required for evaluation"
                denoised_image = self.Evaluate(rec_input)
                with torch.no_grad():
                    tensor_original = torch.tensor(original_image).float().to(self.device)
                    loss_eval = torch.nn.functional.mse_loss(denoised_image, tensor_original)
                    eval_losses.append(loss_eval.item())
                    
                    #Plot
                    self.plot_status(denoised_image, tensor_original, epoch, eval=True)

        # Plot losses
        self.plot_training_losses(losses_split, epoch_losses, eval_losses)
        

    def Evaluate(self, rec_input, batch_size: int = 8):
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
                for i in range(0, im.shape[1], batch_size):
                    batch_slices = im[i:i+batch_size, None, ...].to(self.device, dtype=torch.float32)
                    batch_predictions = self.network(batch_slices)
                    slices_pred.append(batch_predictions.squeeze())

                slices_pred = torch.cat(slices_pred, dim=0)
                pred_splits.append(slices_pred)

            pred_splits = torch.stack(pred_splits).to(self.device, dtype=torch.float32)
            denoised_image = torch.mean(pred_splits, dim=0)

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
            self.network = UNet(1, 1, n_features=16).to(self.device)
        elif network_name == "dncnn":
            self.network = DnCNN(1, num_of_layers=6).to(self.device)
        else:
            raise ValueError(f"Invalid network name: {network_name}")


    def _get_optimizer(self, learning_rate=0.001):
        """
        Helper function to create an optimizer. (Adam)

        :param learning_rate: Learning rate for the optimizer

        :return: An optimizer instance
        """
        return torch.optim.Adam(self.network.parameters(), lr=learning_rate)


# if __name__ == "__main__":
#     # Generate sinogram
#     load_experiment = False
#     foam_generator = FoamGenerator(img_pixels=256, num_spheres=1000, prob_overlap=0)
#     foam = foam_generator.create_phantom()

#     sinogram = Sinogram(foam, num_proj=1024, num_iter=200)
#     sinogram.generate()
#     sinogram.add_poisson_noise(attenuation=ATTENUATION, photon_count=1000)
#     proj_data = sinogram.sinogram
#     sinogram.split_data(num_splits=4)
#     split_sinograms = sinogram.split_sinograms
#     reconstructions = sinogram.reconstruct_splits(split_sinograms, rec_algorithm='FBP_CUDA')
#     reconstruction_noisy = sinogram.reconstruct(rec_algorithm='FBP_CUDA')

#     # Train model
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     n2i = N2I(foam, network_name="unet", device=device, num_splits=4, strategy="X:1", lr=0.002, bs=16, epochs=30)
#     n2i.Train(reconstructions, original_image=foam)

#     # Evaluate model
#     denoised_phantom = n2i.Evaluate(reconstructions)

#     plt.figure()
#     plt.subplot(1, 4, 1)
#     plt.imshow(foam[128], cmap='gray')
#     plt.axis('off')
#     plt.title("Original")
#     plt.subplot(1, 4, 2)
#     plt.imshow(reconstruction_noisy[128], cmap='gray', vmin=0, vmax=1/ATTENUATION)
#     plt.axis('off')
#     plt.title("Noisy")
#     plt.subplot(1, 4, 3)
#     denoised_phantom = denoised_phantom.cpu().numpy()
#     plt.imshow(denoised_phantom[128], cmap='gray', vmin=0, vmax=1/ATTENUATION)
#     plt.axis('off')
#     plt.title("Denoised")
#     plt.subplot(1, 4, 4)
#     plt.imshow(denoised_phantom[128], cmap='gray')
#     plt.axis('off')
#     plt.title("Denoised raw")
#     plt.savefig(f"{n2i.dir}/results.png", dpi=400)
