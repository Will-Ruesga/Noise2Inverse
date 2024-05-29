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
PHANTOM_PATH = './phantoms/save/'

# Sinogram
N_PROJECTIONS = 1000
N_ITERATIONS = 200

####################################################################################################
#                                              MAIN                                                #
####################################################################################################

# Get pahntom
phantom = np.load(PHANTOM_PATH + PHANTOM_NAME)
phantom = phantom.transpose(2, 0, 1)

# Generate sinogram with desired noise
sinogram = Sinogram(phantom, num_proj=N_PROJECTIONS, num_iter=N_ITERATIONS)
sinogram.generate()
sinogram.add_gaussian_noise(std=5)
rec = sinogram.reconstruct(rec_algorithm='FBP_CUDA')



# Train model
# n2i = N2I(sinogram)
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(phantom[128], cmap='gray')
plt.axis('off')
plt.title("Original")
plt.subplot(1, 2, 2)
plt.imshow(rec[128], cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.title("Noisy")
plt.show()

# TODO