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
        self.num_splits = num_splits
        self.network = None
        self.device = device
        self.network_name = network_name
        self.strategy = strategy
        self._initialize_network(self.network_name)

    def generate_source_target_for_split(self, split_reconstructions, num_split):
        source_reconstruction = []
        target_reconstruction = []
        list_indeces = list(range(len(split_reconstructions)))
        list_indeces.remove(num_split)
        if self.strategy == "X:1":
            source_reconstruction = np.mean(np.array([split_reconstructions[i] for i in list_indeces]), axis=0)
            target_reconstruction = np.array(split_reconstructions[num_split])
        elif self.strategy == "1:X":
            source_reconstruction = np.array(split_reconstructions[num_split])
            target_reconstruction = np.mean(np.array([split_reconstructions[i] for i in list_indeces]), axis=0)
        else:
            raise ValueError("Invalid source_imgs value")
        source_reconstruction = torch.tensor(source_reconstruction).float()
        target_reconstruction = torch.tensor(target_reconstruction).float()
        return source_reconstruction, target_reconstruction

    def Train(self, input, epochs: int, batch_size: int, learning_rate: float):
        """
        Train the model with the provided training data.

        :param epochs: Number of epochs to train the model
        :param batch_size: Size of the batches for each training step
        :param learning_rate: Learning rate for the optimizer
        """
        optimizer = self._get_optimizer(learning_rate)
        scaler = torch.cuda.amp.GradScaler()  # Use GradScaler for mixed precision training
        weights_path = "networks/weigths"
        os.makedirs(weights_path, exist_ok=True)

        for epoch in range(epochs):
            self.network.train()
            epoch_loss = 0
            for k in range(self.num_splits):
                slices_pred = []
                source_reconstructions, target_reconstructions = self.generate_source_target_for_split(input, k)
                for i in range(0, source_reconstructions.shape[1], batch_size):
                    batch_slices = source_reconstructions[i:i+batch_size, None, ...].to(self.device, dtype=torch.float16)
                    
                    with torch.cuda.amp.autocast():
                        batch_predictions = self.network(batch_slices)
                        slices_pred.append(batch_predictions.squeeze().cpu())

                    torch.cuda.empty_cache()

                slices_pred = torch.cat(slices_pred, dim=0).to(self.device, dtype=torch.float16)

                with torch.cuda.amp.autocast():
                    target_reconstructions = target_reconstructions.to(self.device, dtype=torch.float16)
                    loss = torch.nn.functional.mse_loss(slices_pred, target_reconstructions)
                    epoch_loss += loss.item()

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            print(f"Epoch {epoch+1} / {epochs} | Loss: {round(epoch_loss, 4)}")
            torch.save(self.network.state_dict(), weights_path + f"/weights.pth")



    def Evaluate(self, input, batch_size: int = 8):
        """
        Evaluate the model with the provided validation data.

        :return: Evaluation metrics
        """

        self.network.eval()
        torch.cuda.empty_cache()
        pred_splits = []

        with torch.no_grad():
            for k in range(self.num_splits):
                slices_pred = []
                source_reconstructions, _ = self.generate_source_target_for_split(input, k)
                for i in range(0, source_reconstructions.shape[1], batch_size):
                    batch_slices = source_reconstructions[i:i+batch_size, None, ...].to(self.device, dtype=torch.float16)
                    
                    with torch.cuda.amp.autocast():
                        batch_predictions = self.network(batch_slices)
                        slices_pred.append(batch_predictions.squeeze())

                slices_pred = torch.cat(slices_pred, dim=0)
                pred_splits.append(slices_pred)

            pred_splits = torch.stack(pred_splits).to(self.device, dtype=torch.float32)
            denoised_image = pred_splits.mean(dim=0)

        return denoised_image
        

    def load_weights(self, weights_path):
        self.network.load_state_dict(torch.load(weights_path))

    # def _get_batches(self, data, batch_size):
    #     """
    #     Helper function to create batches from data.

    #     :param data: The data to be batched
    #     :param batch_size: The size of each batch
    #     :return: A generator yielding batches of data
    #     """
    #     for i in range(0, len(data), batch_size):
    #         yield data[i:i + batch_size]

    def _initialize_network(self, network_name):
        """
        Helper function to initialize the neural network.

        :param network: The name of the network to use
        """
        # Replace with your network initialization code
        if network_name == "unet":
            self.network = UNet(1, 1, n_features=32).to(self.device)
        else:
            raise ValueError(f"Invalid network name: {network_name}")

    def _get_optimizer(self, learning_rate=0.001):
        """
        Helper function to create an optimizer.

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
    sinogram.add_poisson_noise(attenuation=100, photon_count=1000)
    proj_data = sinogram.sinogram
    sinogram.split_data(num_splits=4)
    split_sinograms = sinogram.split_sinograms
    reconstructions = sinogram.reconstruct_splits(split_sinograms, rec_algorithm='FBP_CUDA')
    reconstruction_noisy = sinogram.reconstruct(proj_data, rec_algorithm='FBP_CUDA')

    # Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n2i = N2I(network_name="unet", device=device, num_splits=4)
    n2i.Train(reconstructions, epochs=100, batch_size=8, learning_rate=0.001)

    # # Evaluate model
    denoised_phantom = n2i.Evaluate(reconstructions)

    plt.figure()
    plt.subplot(1, 4, 1)
    plt.imshow(foam[128], cmap='gray')
    plt.axis('off')
    plt.title("Original")
    plt.subplot(1, 4, 2)
    plt.imshow(reconstruction_noisy[128], cmap='gray', vmin=0, vmax=1/100)
    plt.axis('off')
    plt.title("Noisy")
    plt.subplot(1, 4, 3)
    plt.imshow(denoised_phantom.cpu().numpy()[128], cmap='gray', vmin=0, vmax=1/100)
    plt.axis('off')
    plt.title("Denoised")
    plt.savefig("results.png")
    plt.subplot(1, 4, 3)
    plt.imshow(denoised_phantom.cpu().numpy()[128], cmap='gray')
    plt.axis('off')
    plt.title("Denoised full")
    plt.savefig("results.png")
