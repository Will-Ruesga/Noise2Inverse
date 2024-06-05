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

# Experiement
WAVE = 'sin' # 'sinc'
REC_ALGORITHM = 'FBP_CUDA'

# Artifacts
ART_TYPE = 'zinger' # zinger or ring

# Ring artifact
OFFSET = 0.5

# Zinger artifact
NUM_ZINGERS = 10
INTENSITY = 10

# Plotting
VMAX = 1

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
# Load phantom
foam = np.load(PHANTOM_PATH + PHANTOM_NAME)

# Generate sinogram
sinogram = Sinogram(foam, N_PROJECTIONS, N_ITERATIONS)
sinogram.generate()


### ---------- NOISE ---------- ###
# Create Non Independent noise
depth, rows, cols = sinogram.sinogram.shape
if WAVE == 'sin':
    non_ind_noise = sine(rows, cols, frequency=10, angle=np.pi/2)
    att = 0.2
elif WAVE == 'sinc':
    non_ind_noise = sinc(rows, cols, ripples=20, att=0.5)

clean_sinogram = sinogram.sinogram.copy()

# Add noise to sinogram
if ART_TYPE == 'ring':
    sinogram.add_ring_artifact(position=int(np.random.uniform(0, cols)), offset=OFFSET)
elif ART_TYPE == 'zinger':
    sinogram.add_zinger_artifact(NUM_ZINGERS, INTENSITY)
print("Finish adding noise")

# Reconstruction
rec = sinogram.reconstruct(REC_ALGORITHM)
print("Finish reconstruction")

# Create a figure with two rows and three columns
fig, axs = plt.subplots(1, 3, figsize=(10, 5))
axs = axs.flatten()

axs[0].imshow(clean_sinogram[128], cmap='gray')
axs[0].axis('off')
axs[0].set_title("Clean sinogram")


axs[1].imshow(sinogram.sinogram[128], cmap='gray')
axs[1].axis('off')
axs[1].set_title("Noisy sinogram")

axs[2].imshow(rec[128], cmap='gray', vmin=0, vmax=VMAX)
axs[2].axis('off')
axs[2].set_title("Noisy reconstruction")

# Save the figure
plt.tight_layout()
plt.savefig(f"sinogram_reconstruction.png", dpi=400)
plt.show()
