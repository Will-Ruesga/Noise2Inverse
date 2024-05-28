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

####################################################################################################
#                                             FUNCTIONS                                            #
####################################################################################################

def sine(rows, cols, frequency, angle=np.pi/2):
    """
    Creates a 2D sine wave image.

    Parameters:
    - rows: Number of rows in the image.
    - cols: Number of columns in the image.
    - frequency: Frequency of the sine wave.
    - angle: Angle of the sine in radiants.

    Returns:
    - 2D sine wave image.
    """
    x = np.arange(cols)
    y = np.arange(rows)
    x, y = np.meshgrid(x, y)

    x_rotated = x * np.cos(angle) - y * np.sin(angle)
    sine_wave = np.sin(2 * np.pi * frequency * (x_rotated / cols))

    return sine_wave


def sinc(rows, cols, ripples=20):
    """
    Creates a 2D sinc wave image.

    Parameters:
    - rows: Number of rows in the image.
    - cols: Number of columns in the image.
    - ripples: Controls how many ripples we see.

    Returns:
    - 2D sinc wave image.
    """
    x = np.linspace(-ripples, ripples, cols)
    y = np.linspace(-ripples, ripples, rows)
    x, y = np.meshgrid(x, y)
    
    r = np.sqrt(x**2 + y**2)
    sinc_wave = np.sinc(r / np.pi)
    
    return sinc_wave


####################################################################################################
#                                              MAIN                                                #
####################################################################################################

### ---------- INITIALIZE ---------- ###
# Get pahntom
phantom = np.load(PHANTOM_PATH + PHANTOM_NAME)

# Generate sinogram with desired noise
sinogram = Sinogram(phantom, num_proj=N_PROJECTIONS, num_iter=N_ITERATIONS)
sinogram.generate()


### ---------- NOISE ---------- ###
# Create Non Independent noise
rows, cols = sinogram.sinogram.shape
if WAVE == 'sin':
    non_ind_noise = sine(rows, cols, frequency=10, angle=np.pi/2)
    att = 0.2
elif WAVE == 'sinc':
    non_ind_noise = sinc(rows, cols, ripples=20, att = 0.5)

# Add noise to sinogram
sinogram.add_non_independent_noise(non_ind_noise, att)


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

