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

# Artifacts
ART_TYPE = 'zinger' # 'ring' 'under'

# Ring artifact
OFFSET = 0,5

# Zinger artifact
NUM_ZINGERS = 50
INTENSITY = 20

####################################################################################################
#                                              MAIN                                                #
####################################################################################################

### ---------- INITIALIZE ---------- ###
# Get pahntom
phantom = np.load(PHANTOM_PATH + PHANTOM_NAME)

# Generate sinogram with desired noise
sinogram = Sinogram(phantom, num_proj=N_PROJECTIONS, num_iter=N_ITERATIONS)
sinogram.generate()


### ---------- ARTIFACT ---------- ###
# Create Non Independent noise
rows, cols = sinogram.shape
if ART_TYPE == 'ring':
    sinogram.add_ring_artifact(position=rows/2, offset=OFFSET)
elif ART_TYPE == 'zinger':
    sinogram.add_zinger_artifact(NUM_ZINGERS, INTENSITY)
elif ART_TYPE == 'under':
    pass



### ---------- TRAINING ---------- ###
# Split the sinogram and reconstruct to prepare the data for training 
sinogram.split_data(num_splits=K)
reconst = sinogram.reconstruct_splits(sinogram.split_sinograms, rec_algorithm='FBP_CUDA')
reconst_noisy = sinogram.reconstruct(sinogram.sinogram, rec_algorithm='FBP_CUDA')

# Train model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n2i = N2I(network_name="unet", device=device, num_splits=K)
n2i.Train(reconst, epochs=EPS, batch_size=BS, learning_rate=LR)

### ---------- EVALUATION ---------- ###
denoised_phantom = n2i.Evaluate(reconst)

# Plot of the results
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(phantom[128], cmap='gray')
plt.axis('off')
plt.title("Original")
plt.subplot(1, 4, 2)
plt.imshow(reconst_noisy[128], cmap='gray', vmin=0, vmax=1/100)
plt.axis('off')
plt.title("Noisy")
plt.subplot(1, 4, 3)
plt.imshow(denoised_phantom.cpu().numpy()[128], cmap='gray', vmin=0, vmax=1/100)
plt.axis('off')
plt.title("Denoised")