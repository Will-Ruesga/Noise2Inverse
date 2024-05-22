import astra
import torch
import numpy as np

from tqdm.notebook import tqdm

from ..phantoms.foam_generator import FoamGenerator
from ..phantoms.sparse_generator import SparseGenerator
from ..sinogram.sinogram_generator import Sinogram

####################################################################################################
#                                               CLASS                                              #
####################################################################################################

class N2I:
    def __init__(self, sinogram):
        """
        Initialize the N2I class with the sinogram.
        This class recives the sinogram and has two methods, for training
        and for evaluation.

        :param sinogram: The sinogram in which the model will be trained
        """
        self.sinogram = sinogram

    def Train(self, epochs: int, batch_size: int, learning_rate: float):
        """
        Train the model with the provided training data.

        :param epochs: Number of epochs to train the model
        :param batch_size: Size of the batches for each training step
        :param learning_rate: Learning rate for the optimizer
        """
        # Example training loop (customize according to your framework)
        optimizer = self._get_optimizer(learning_rate)

        for epoch in range(epochs):
            for batch in self._get_batches(self.training_data, batch_size):
                # Example training step (customize as needed)
                loss = self.model.train_on_batch(batch['inputs'], batch['targets'])
                optimizer.minimize(loss)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss}")

    def Evaluate(self):
        """
        Evaluate the model with the provided validation data.

        :return: Evaluation metrics
        """
        # Example evaluation loop (customize according to your framework)
        total_loss = 0
        num_batches = 0

        for batch in self._get_batches(self.validation_data, batch_size=1):
            loss = self.model.evaluate_on_batch(batch['inputs'], batch['targets'])
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Evaluation Loss: {avg_loss}")
        return avg_loss

    # def _get_batches(self, data, batch_size):
    #     """
    #     Helper function to create batches from data.

    #     :param data: The data to be batched
    #     :param batch_size: The size of each batch
    #     :return: A generator yielding batches of data
    #     """
    #     for i in range(0, len(data), batch_size):
    #         yield data[i:i + batch_size]

    # def _get_optimizer(self, learning_rate):
    #     """
    #     Helper function to create an optimizer.

    #     :param learning_rate: Learning rate for the optimizer
    #     :return: An optimizer instance
    #     """
    #     # Replace with your preferred optimizer (e.g., Adam, SGD)
    #     return Optimizer(learning_rate)



####################################################################################################
#                                               COLAB                                              #
####################################################################################################

# # Generate noise
# noisy_proj_data = generate_noisy_sinogram(proj_data, 0.8, 100)
# # Split sinogram
# split_data = []
# for proj in noisy_proj_data:
#   split_data.append(distribute_array_cyclic(proj, K_SPLITS))
# split_data = np.array(split_data)
# split_data = split_data.transpose(1, 0, 2, 3)
# print("Split data shape:", split_data.shape)



# reconstructions = []
# for k_split, split in enumerate(split_data):
#     reconstructions_split = []
#     for proj in split:
#         proj_sino = astra.create_proj_geom('parallel', 1.0, DETECTOR_SHAPE[1], np.linspace(np.pi/NUM_PROJECTIONS*k_split, np.pi, NUM_PROJECTIONS // K_SPLITS, False))
#         sinogram_id = astra.data2d.create('-sino', proj_sino, proj)
#         recon_id = astra.data2d.create('-vol', vol_geom, 0)
#         cfg = astra.astra_dict('FBP_CUDA')
#         cfg['ReconstructionDataId'] = recon_id
#         cfg['ProjectionDataId'] = sinogram_id
#         alg_id = astra.algorithm.create(cfg)
#         astra.algorithm.run(alg_id, NUM_ITERATIONS)
#         rec = astra.data2d.get(recon_id)
#         reconstructions_split.append(rec)

#         astra.algorithm.delete(alg_id)
#         astra.data2d.delete(recon_id)
#         astra.data2d.delete(sinogram_id)
#     reconstructions.append(reconstructions_split)
# astra.projector.delete(proj_id)



# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# net = UNet(1, 1).to(device)
# optim = torch.optim.Adam(net.parameters())
# source_reconstructions = torch.Tensor(source_reconstruction).to(device)
# target_reconstructions = torch.Tensor(target_reconstruction).to(device)
# num_epochs = 100
# batch_size = 8
# for epoch in range(num_epochs):
#     slices_pred = []
#     net.train()
#     for i in range(0, source_reconstructions.shape[1], batch_size):
#         batch_slices = source_reconstructions[i:i+batch_size, None, ...]
#         with torch.autocast(device_type=device, dtype=torch.float16):
#             batch_slices = batch_slices.to(device)
#             batch_predictions = net(batch_slices)
#             slices_pred.append(batch_predictions.squeeze().cpu())
#             torch.cuda.empty_cache()
#     slices_pred = torch.cat(slices_pred, dim=1).to(device)
#     target_reconstructions = target_reconstructions.to(device, dtype=slices_pred.dtype)
#     print(f"Shape prediction: {slices_pred.shape}")
#     print(f"Shape target: {target_reconstructions.shape}")
#     loss = torch.nn.functional.mse_loss(slices_pred, target_reconstructions)
#     optim.zero_grad()
#     loss.backward()
#     optim.step()
#     print(f"Epoch {epoch}, loss: {loss.item()}")