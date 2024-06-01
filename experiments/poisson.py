import astra
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

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
ATTENUATION = 200
K = 4

REC_ALGORITHM = 'FBP_CUDA'

# Training hyperparameters
EPS = 30
BS = 8
LR = 0.005



####################################################################################################
#                                              MAIN                                                #
####################################################################################################

# Generate sinogram
load_experiment = False
foam = np.load(PHANTOM_PATH + PHANTOM_NAME)

sinogram = Sinogram(foam, N_PROJECTIONS, N_ITERATIONS)
sinogram.generate()
sinogram.add_poisson_noise(attenuation=ATTENUATION, photon_count=1000)
proj_data = sinogram.sinogram
sinogram.split_data(K)
split_sinograms = sinogram.split_sinograms
rec_splits = sinogram.reconstruct_splits(split_sinograms, REC_ALGORITHM)
rec = sinogram.reconstruct(REC_ALGORITHM)
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(foam[128], cmap='gray')
plt.axis('off')
plt.title("Original")
plt.subplot(1, 3, 2)
plt.imshow(rec[128], cmap='gray', vmin=0, vmax=1/ATTENUATION)
plt.axis('off')
plt.title("Reconstructed")
plt.subplot(1, 3, 3)
plt.imshow(rec_splits[0][128], cmap='gray')
plt.axis('off')
plt.title("Reconstructed split")
plt.savefig(f"debug.png", dpi=400)

# Train model
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# n2i = N2I(foam, "unet", device, K, "X:1", LR, BS, EPS)
# n2i.Train(rec_splits, original_image=foam[:,::-1])

# # Evaluate model
# denoised_phantom = n2i.Evaluate(rec_splits)

# plt.figure()
# plt.subplot(1, 4, 1)
# plt.imshow(foam[128], cmap='gray')
# plt.axis('off')
# plt.title("Original")
# plt.subplot(1, 4, 2)
# plt.imshow(rec[128], cmap='gray', vmin=0, vmax=1/ATTENUATION)
# plt.axis('off')
# plt.title("Noisy")
# plt.subplot(1, 4, 3)
# denoised_phantom = denoised_phantom.cpu().numpy()
# plt.imshow(denoised_phantom[128], cmap='gray', vmin=0, vmax=1/ATTENUATION)
# plt.axis('off')
# plt.title("Denoised")
# plt.subplot(1, 4, 4)
# plt.imshow(denoised_phantom[128], cmap='gray')
# plt.axis('off')
# plt.title("Denoised raw")
# plt.savefig(f"{n2i.dir}/results.png", dpi=400)