import astra
import torch
import numpy as np
import matplotlib.pyplot as plt

from sinogram.sinogram_generator import Sinogram
from noise2inverse.n2i import N2I
####################################################################################################
#                                             CONSTANTS                                            #
####################################################################################################

# Phantom
PHANTOM_NAME = 'foam_phantom.npy'
PHANTOM_PATH = 'phantoms/save/'

# Sinogram
N_PROJECTIONS = 1024
N_ITERATIONS = 200
ATTENUATION = 5e-3
PHOTON_COUNT = 10
K = 4

REC_ALGORITHM = 'FBP_CUDA'

# Training hyperparameters
EPS = 50
BS = 8
LR = 0.005

####################################################################################################
#                                              MAIN                                                #
####################################################################################################

# Load phantom
foam = np.load(PHANTOM_PATH + PHANTOM_NAME)

# Generate sinogram
sinogram = Sinogram(foam, N_PROJECTIONS, N_ITERATIONS)
sinogram.generate()
sinogram.add_poisson_noise(attenuation=ATTENUATION, photon_count=PHOTON_COUNT)

# Split data in K parts and reconstruct each split
sinogram.split_data(K)
rec_splits = sinogram.reconstruct_splits(sinogram.split_sinograms, REC_ALGORITHM)

# Reconstruction
rec = sinogram.reconstruct(REC_ALGORITHM)

# Train model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n2i = N2I(foam, "unet", device, K, "X:1", LR, BS, EPS)
n2i.Train(rec_splits, original_image=foam)

# Evaluate model
denoised_phantom = n2i.Evaluate(rec_splits)

# Convert denoised_phantom to numpy array if it's a PyTorch tensor
denoised_phantom = denoised_phantom.cpu().numpy()

# Create a figure with two rows and three columns
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Plot the first row without vmin and vmax
axs[0, 0].imshow(foam[128], cmap='gray')
axs[0, 0].axis('off')
axs[0, 0].set_title("Original")

axs[0, 1].imshow(rec[128], cmap='gray')
axs[0, 1].axis('off')
axs[0, 1].set_title("Noisy")

axs[0, 2].imshow(denoised_phantom[128], cmap='gray')
axs[0, 2].axis('off')
axs[0, 2].set_title("Denoised")

# Plot the second row with vmin=0 and vmax=1/ATTENUATION
# VMAX = 1/ATTENUATION
VMAX = 0.004
axs[1, 0].imshow(foam[128], cmap='gray', vmin=0, vmax=VMAX)
axs[1, 0].axis('off')
axs[1, 0].set_title("Original (Scaled)")

axs[1, 1].imshow(rec[128], cmap='gray', vmin=0, vmax=VMAX)
axs[1, 1].axis('off')
axs[1, 1].set_title("Noisy (Scaled)")

axs[1, 2].imshow(denoised_phantom[128], cmap='gray', vmin=0, vmax=VMAX)
axs[1, 2].axis('off')
axs[1, 2].set_title("Denoised (Scaled)")

# Save the figure
plt.tight_layout()
plt.savefig(f"{n2i.dir}/results.png", dpi=400)
plt.show()