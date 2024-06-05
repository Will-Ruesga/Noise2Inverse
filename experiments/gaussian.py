import torch
import numpy as np
import matplotlib.pyplot as plt

from sinogram.sinogram_generator import Sinogram
from noise2inverse.n2i import N2I
####################################################################################################
#                                             CONSTANTS                                            #
####################################################################################################

# Phantom
PHANTOM_NAME = 'foam_phantom_overlap.npy'
PHANTOM_PATH = 'phantoms/save/'

# Sinogram
N_PROJECTIONS = 1024
N_ITERATIONS = 200
STD_GAUSSIAN = 8
K = 4

REC_ALGORITHM = 'FBP_CUDA'

# Training hyperparameters
EPS = 50
BS = 8
LR = 0.005

# Plotting
VMAX = 1

####################################################################################################
#                                              MAIN                                                #
####################################################################################################

# Load phantom
foam = np.load(PHANTOM_PATH + PHANTOM_NAME)

# Generate sinogram
sinogram = Sinogram(foam, N_PROJECTIONS, N_ITERATIONS)
sinogram.generate()

min = np.max(sinogram.sinogram) * 0.1
max = np.max(sinogram.sinogram) * 0.15
bounds = (min, max)

sinogram.add_gaussian_noise(None, STD_GAUSSIAN)

# Split data in K parts and reconstruct each split
sinogram.split_data(K)
rec_splits = sinogram.reconstruct_splits(sinogram.split_sinograms, REC_ALGORITHM)

# Reconstruction
rec = sinogram.reconstruct(REC_ALGORITHM)

# Train model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n2i = N2I(foam, "unet", device, K, "X:1", LR, BS, EPS, comment="gaussian")
n2i.Train(rec_splits, rec)

# Evaluate model
denoised_phantom = n2i.Evaluate(rec_splits, rec, psnr=True)

# Convert denoised_phantom to numpy array if it's a PyTorch tensor
denoised_phantom = denoised_phantom.cpu().numpy()

# Create a figure with two rows and three columns
fig, axs = plt.subplots(2, 3, figsize=(15, 7.5), gridspec_kw={'height_ratios': [2, 1]})

# Plot the first row without vmin and vmax
axs[0, 0].imshow(foam[128], cmap='gray', vmin=0, vmax=VMAX)
axs[0, 0].axis('off')
axs[0, 0].set_title("Original")

axs[0, 1].imshow(rec[128], cmap='gray', vmin=0, vmax=VMAX)
axs[0, 1].axis('off')
axs[0, 1].set_title("Noisy")

axs[0, 2].imshow(denoised_phantom[128], cmap='gray', vmin=0, vmax=VMAX)
axs[0, 2].axis('off')
axs[0, 2].set_title("Denoised")

# Plot the second row with vmin=0 and vmax=1
axs[1, 0].hist(foam[128].flatten(), bins=50, range=(0, 1), color='black')
mean_foam = np.mean(foam[128])
y_pos = np.histogram(foam[128], bins=50, range=(0, 1))[0].max()
axs[1, 0].vlines(mean_foam, 0, y_pos, color='red', linestyle='dashed')
axs[1, 0].text(mean_foam + 0.1, y_pos - 0.1*y_pos, f"Mean: {mean_foam:.2f}", color='red', fontsize=10)
axs[1, 0].set_xlabel("Pixel intensity")
axs[1, 0].set_ylabel("Frequency")
axs[1, 0].set_title("Original histogram")

axs[1, 1].hist(rec[128].flatten(), bins=50, range=(0, 1), color='black')
mean_noisy = np.mean(rec[128])
y_pos = np.histogram(rec[128], bins=50, range=(0, 1))[0].max()
axs[1, 1].vlines(mean_noisy, 0, y_pos, color='red', linestyle='dashed')
axs[1, 1].text(mean_noisy + 0.1, y_pos - 0.1*y_pos, f"Mean: {mean_noisy:.2f}", color='red', fontsize=10)
axs[1, 1].set_xlabel("Pixel intensity")
axs[1, 1].set_ylabel("Frequency")
axs[1, 1].set_title("Noisy histogram")

axs[1, 2].hist(denoised_phantom[128].flatten(), bins=50, range=(0, 1), color='black')
mean_denoised = np.mean(denoised_phantom[128])
y_pos = np.histogram(denoised_phantom[128], bins=50, range=(0, 1))[0].max()
axs[1, 2].vlines(mean_denoised, 0, y_pos, color='red', linestyle='dashed')
axs[1, 2].text(mean_denoised + 0.1, y_pos - 0.1*y_pos, f"Mean: {mean_denoised:.2f}", color='red', fontsize=10)
axs[1, 2].set_xlabel("Pixel intensity")
axs[1, 2].set_ylabel("Frequency")
axs[1, 2].set_title("Denoised histogram")

# Save the figure
plt.tight_layout()
plt.savefig(f"{n2i.dir}/results.png", dpi=400)
plt.show()
