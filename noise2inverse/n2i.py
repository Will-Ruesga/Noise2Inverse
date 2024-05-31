import os
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt

from phantoms.foam_generator import FoamGenerator
from phantoms.sparse_generator import SparseGenerator
from sinogram.sinogram_generator import Sinogram
from noise2inverse.networks.unet import UNet

# Suppress all warnings
warnings.filterwarnings('ignore')

####################################################################################################
#                                               CLASS                                              #
####################################################################################################

class N2I:
    def __init__(self, network_name: str = "unet", device="cpu", num_splits: int = 4, strategy="X:1"):
        """
        Initialize the N2I class with the sinogram.
        This class recives the sinogram and has two methods, for training
        and for evaluation.

        :param sinogram: The sinogram in which the model will be trained
        """
        # Set parameter values
        self.num_splits = num_splits
        self.network = None
        self.optimizer = None
        self.device = device
        self.network_name = network_name
        self.strategy = strategy
        self.path_weights = "networks/weigths"

        # Initialize the network
        self._initialize_network(self.network_name)



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



    def Train(self, rec_input, epochs: int, batch_size: int, learning_rate: float):
        """
        Train the model with the provided training data.

        :param rec_input: Input data of the reconstruction for trianing the network
        :param epochs: Number of epochs to train the model
        :param batch_size: Size of the batches for each training step
        :param learning_rate: Learning rate for the optimizer
        """

        # Get the optimizer and Scaler
        if self.network_name != "msd":
            optimizer = self._get_optimizer(learning_rate)
        scaler = torch.cuda.amp.GradScaler()  # Use GradScaler for mixed precision training

        # Make sure the path for the weights exists
        os.makedirs(self.path_weights, exist_ok=True)

        # Epoch training loop
        for epoch in range(epochs):
            self.network.train()
            epoch_loss = 0
            for k in range(self.num_splits):
                slices_pred = []
                source_recs, target_recs = self.generate_source_target_for_split(rec_input, k)
                for i in range(0, source_recs.shape[0], batch_size):
                    batch_slices = source_recs[i:i+batch_size, None, ...].to(self.device, dtype=torch.float16)
                    
                    # Use 16 byte to reduce comptutational load
                    with torch.cuda.amp.autocast():
                        batch_predictions = self.network(batch_slices)
                        slices_pred.append(batch_predictions.squeeze().cpu())

                    torch.cuda.empty_cache()

                slices_pred = torch.cat(slices_pred, dim=0).to(self.device, dtype=torch.float16)
                # if epoch % 10 == 0:
                #     plt.figure()
                #     plt.subplot(1, 2, 1)
                #     plt.imshow(slices_pred[128].cpu().detach(), cmap='gray', vmin=0, vmax=1/200)
                #     plt.subplot(1, 2, 2)
                #     plt.imshow(target_recs[128].cpu().detach(), cmap='gray', vmin=0, vmax=1/200)
                #     plt.savefig(f"figures_train/ep{epoch}_{k}split.png")
                
                # Use 16 byte to reduce comptutational load
                with torch.cuda.amp.autocast():
                    target_recs = target_recs.to(self.device, dtype=torch.float16)
                    loss = torch.nn.functional.mse_loss(slices_pred, target_recs)
                    epoch_loss += loss.item()

                # Update optimazer and scaler
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # Print progress and loss and save network weights each epoch
            print(f"Epoch {epoch+1} / {epochs} | Loss: {round(epoch_loss, 4)}")
            torch.save(self.network.state_dict(), self.path_weights + f"/weights.pth")



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
                    batch_slices = im[i:i+batch_size, None, ...].to(self.device, dtype=torch.float16)
                
                    with torch.cuda.amp.autocast():
                        batch_predictions = self.network(batch_slices)
                        slices_pred.append(batch_predictions.squeeze())

                slices_pred = torch.cat(slices_pred, dim=0)
                pred_splits.append(slices_pred)

            pred_splits = torch.stack(pred_splits).to(self.device, dtype=torch.float32)
            denoised_image = torch.mean(pred_splits, dim=0)

        return denoised_image
        

    def load_weights(self, weights_path):
        """
            Loads saved weights from the desired path.
        """
        self.network.load_state_dict(torch.load(weights_path))


    def _initialize_network(self, network_name):
        """
        Helper function to initialize the neural network.

        :param network_name: The name of the network to use
        """
        # Replace with your network initialization code
        if network_name == "unet":
            self.network = UNet(1, 1, n_features=32).to(self.device)
        elif network_name == "msd":
            raise NotImplementedError("MSD network not implemented yet")
            # from msd_pytorch.msd_pytorch.msd_regression_model import MSDRegressionModel
            # model = MSDRegressionModel(1, 1, 100, 1)
            # self.network = model.net
            # self.optimizer = model.optimizer
        else:
            raise ValueError(f"Invalid network name: {network_name}")

    def _get_optimizer(self, learning_rate=0.001):
        """
        Helper function to create an optimizer. (Adam)

        :param learning_rate: Learning rate for the optimizer

        :return: An optimizer instance
        """
        return torch.optim.Adam(self.network.parameters(), lr=learning_rate)


if __name__ == "__main__":
    # Generate sinogram
    foam_generator = FoamGenerator(img_pixels=256, num_spheres=1000, prob_overlap=0)
    foam = foam_generator.create_phantom()

    sinogram = Sinogram(foam, num_proj=1024, num_iter=200)
    sinogram.generate()
    sinogram.add_poisson_noise(attenuation=200, photon_count=1000)
    proj_data = sinogram.sinogram
    sinogram.split_data(num_splits=4)
    split_sinograms = sinogram.split_sinograms
    reconstructions = sinogram.reconstruct_splits(split_sinograms, rec_algorithm='FBP_CUDA')
    reconstruction_noisy = sinogram.reconstruct(rec_algorithm='FBP_CUDA')

    # Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n2i = N2I(network_name="unet", device=device, num_splits=4)
    n2i.Train(reconstructions, epochs=200, batch_size=16, learning_rate=0.001)

    # Evaluate model
    denoised_phantom = n2i.Evaluate(reconstructions)

    plt.figure()
    plt.subplot(1, 4, 1)
    plt.imshow(foam[128], cmap='gray')
    plt.axis('off')
    plt.title("Original")
    plt.subplot(1, 4, 2)
    plt.imshow(reconstruction_noisy[128], cmap='gray', vmin=0, vmax=1/200)
    plt.axis('off')
    plt.title("Noisy")
    plt.subplot(1, 4, 3)
    plt.imshow(denoised_phantom.cpu().numpy()[128], cmap='gray', vmin=0, vmax=1/200)
    plt.axis('off')
    plt.title("Denoised")
    plt.savefig("results.png")
    plt.subplot(1, 4, 4)
    plt.imshow(denoised_phantom.cpu().numpy()[128], cmap='gray')
    plt.axis('off')
    plt.title("Denoised full")
    plt.savefig("results.png", dpi=400)
