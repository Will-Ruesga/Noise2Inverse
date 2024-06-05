import torch
import numpy as np
import matplotlib.pyplot as plt

from phantoms.foam_generator import FoamGenerator as GeometricalGenerator
from phantoms.sparse_generator import SparseGenerator
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
STD_GAUSSIAN = 8
K = 4

REC_ALGORITHM = 'FBP_CUDA'

# Training hyperparameters
EPS = 50
BS = 8
LR = 0.005

# Phantom
PHANTOM = "sparse"
OVERLAP = [0, 0.2]
NUM_SPHERES = 1000

# Plotting
VMAX = 1

####################################################################################################
#                                              MAIN                                                #
####################################################################################################

if PHANTOM == "geometrical":
    foam_generator_cls = GeometricalGenerator
elif PHANTOM == "sparse":
    foam_generator_cls = SparseGenerator

# Create a figure with two rows and three columns
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for i, overlap in enumerate(OVERLAP):
    foam_generator = foam_generator_cls(num_spheres=NUM_SPHERES, prob_overlap=overlap)
    foam = foam_generator.create_phantom()

    # Generate sinogram
    sinogram = Sinogram(foam, N_PROJECTIONS, N_ITERATIONS)
    sinogram.generate()

    sinogram.add_gaussian_noise(None, STD_GAUSSIAN)

    # Split data in K parts and reconstruct each split
    sinogram.split_data(K)
    rec_splits = sinogram.reconstruct_splits(sinogram.split_sinograms, REC_ALGORITHM)

    # Reconstruction
    rec = sinogram.reconstruct(REC_ALGORITHM)

    # Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n2i = N2I(foam, "unet", device, K, "X:1", LR, BS, EPS, comment="sparse")
    n2i.Train(rec_splits, rec)

    # Evaluate model
    denoised_phantom = n2i.Evaluate(rec_splits, rec)

    # Convert denoised_phantom to numpy array if it's a PyTorch tensor
    denoised_phantom = denoised_phantom.cpu().numpy()

    # Plot the first row without vmin and vmax
    axs[i, 0].imshow(foam[128], cmap='gray', vmin=0, vmax=VMAX)
    axs[i, 0].axis('off')
    axs[i, 0].set_title("Original" + (" overlap" if overlap > 0 else ""))

    axs[i, 1].imshow(rec[128], cmap='gray', vmin=0, vmax=VMAX)
    axs[i, 1].axis('off')
    axs[i, 1].set_title("Noisy" + (" overlap" if overlap > 0 else ""))

    axs[i, 2].imshow(denoised_phantom[128], cmap='gray', vmin=0, vmax=VMAX)
    axs[i, 2].axis('off')
    axs[i, 2].set_title("Denoised" + (" overlap" if overlap > 0 else ""))

# Save the figure
fig.tight_layout()
fig.savefig(f"comparison_{PHANTOM}_foam.png", dpi=400)
