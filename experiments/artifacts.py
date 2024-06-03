import astra
import torch
import numpy as np
import matplotlib.pyplot as plt

from ..sinogram.sinogram_generator import Sinogram
from ..noise2inverse.n2i import N2I
####################################################################################################
#                                             CONSTANTS                                            #
####################################################################################################

# Phantom
PHANTOM_NAME = 'Foam.nyp'
PHANTOM_PATH = 'C:/Users/wilru/Documents/LU/S4/CITO/Noise2Inverse/phantoms/save/'

# Sinogram
N_PROJECTIONS = 1024
N_ITERATIONS = 200

# Experiement
WAVE = 'sin' # 'sinc'
K = 4

# Training hyperparameters
EPS = 100
BS = 8
LR = 0.001

REC_ALGORITHM = 'FBP_CUDA'

# Artifacts
ART_TYPE = 'zinger' # 'ring' 'under'

# Ring artifact
OFFSET = 0,5

# Zinger artifact
NUM_ZINGERS = 50
INTENSITY = 20

# Plotting
VMAX = 1

####################################################################################################
#                                              MAIN                                                #
####################################################################################################

### ---------- INITIALIZE ---------- ###
# Get pahntom
foam = np.load(PHANTOM_PATH + PHANTOM_NAME)

# Generate sinogram with desired noise
sinogram = Sinogram(foam, num_proj=N_PROJECTIONS, num_iter=N_ITERATIONS)
sinogram.generate()


### ---------- ARTIFACT ---------- ###
# Create Non Independent noise
rows, cols = sinogram.shape
if ART_TYPE == 'ring':
    sinogram.add_ring_artifact(position=rows/2, offset=OFFSET)
elif ART_TYPE == 'zinger':
    sinogram.add_zinger_artifact(NUM_ZINGERS, INTENSITY)


### ---------- TRAINING ---------- ###
# Split the sinogram and reconstruct to prepare the data for training 
sinogram.split_data(K)
rec_splits = sinogram.reconstruct_splits(sinogram.split_sinograms, REC_ALGORITHM)

# Reconstruction
rec = sinogram.reconstruct(REC_ALGORITHM)

# Train model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n2i = N2I(foam, "unet", device, K, "X:1", LR, BS, EPS, comment=f"{ART_TYPE}_artifact")
n2i.Train(rec_splits, rec)

### ---------- EVALUATION ---------- ###
# Evaluate model
denoised_phantom = n2i.Evaluate(rec_splits, rec)

# Convert denoised_phantom to numpy array if it's a PyTorch tensor
denoised_phantom = denoised_phantom.cpu().numpy()

# Create a figure with two rows and three columns
fig, axs = plt.subplots(1, 3, figsize=(15, 10))
axs = axs.flatten()

axs[0].imshow(foam[128], cmap='gray', vmin=0, vmax=VMAX)
axs[0].axis('off')
axs[0].set_title("Original")

axs[1].imshow(rec[128], cmap='gray', vmin=0, vmax=VMAX)
axs[1].axis('off')
axs[1].set_title("Noisy")

axs[2].imshow(denoised_phantom[128], cmap='gray', vmin=0, vmax=VMAX)
axs[2].axis('off')
axs[2].set_title("Denoised")

# Save the figure
plt.tight_layout()
plt.savefig(f"{n2i.dir}/results.png", dpi=400)
plt.show()