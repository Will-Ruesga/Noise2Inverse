import astra
import torch
import numpy as np

from ..sinogram.sinogram_generator import Sinogram
from ..noise2inverse.n2i import N2I
####################################################################################################
#                                             CONSTANTS                                            #
####################################################################################################

# Phantom
PHANTOM_NAME = 'Foam.nyp'
PHANTOM_PATH = 'C:/Users/wilru/Documents/LU/S4/CITO/Noise2Inverse/phantoms/save/'

# Sinogram
N_PROJECTIONS = 1000
N_ITERATIONS = 200

####################################################################################################
#                                              MAIN                                                #
####################################################################################################

# Get pahntom
phantom = np.load(PHANTOM_PATH + PHANTOM_NAME)

# Generate sinogram with desired noise
sinogram = Sinogram(phantom, num_proj=N_PROJECTIONS, num_iter=N_ITERATIONS)
sinogram.generate()
sinogram.add_non_independent_noise()

# Train model
n2i = N2I(sinogram)

# TODO