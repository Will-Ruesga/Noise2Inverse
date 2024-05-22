import astra
import numpy as np
import matplotlib.pyplot as plt


####################################################################################################
#                                              CLASS                                               #
####################################################################################################
class Sinogram:
    '''
        The Sinogram class provides a suite of methods for generating and processing sinograms.
        Adds the Gaussian or Poisson noise based on the hyperprameters.
    '''

    def __init__(self, phantom, num_proj: int = 1024, num_iter: int = 200):

        # Prjoection variables
        self.phantom = phantom
        self.num_proj = num_proj
        self.num_iter = num_iter
        self.detector_shape = (self.phantom.shape[0], np.ceil(np.sqrt(self.phantom.shape[1]**2 + self.phantom.shape[2]**2)))

        # Sinogram variables
        self.sinogram = None
        self.absorption = None
        self.attenuation = None
        self.photon_count = None


    def generate(self):
        """
        Creates volume and projection geometry in order to generate the sinogram.
        """
        # Creage volumme and projection geometries
        angles = np.linspace(0, np.pi, self.num_proj, False)
        vg = astra.create_vol_geom(self.phantom.shape)
        pg = astra.create_proj_geom('parallel3d', 1.0, 1.0, int(self.detector_shape[0]), int(self.detector_shape[1]), angles)

        # Create sinogram data
        _ , pdata = astra.create_sino3d_gpu(self.phantom, pg, vg)
        self.sinogram = pdata


    def add_poisson_noise(self, attenuation: float, photon_count: int):
        """
        Generate a noisy sinogram from a sinogram with Poisson noise with
        the hyperparameters absortiona dn photon count.

        :param attenuation: The attenuation to apply to the sinogram
        :param photon_count: The initial photon count tha beam has
        """
        # Save values
        self.attenuation = attenuation
        self.photon_count = photon_count

        # Apply attenuation to sinogram
        self.sinogram = self.sinogram / attenuation

        # Compute and save absortion
        self.absorption = 1 - np.mean(np.exp(-self.sinogram[self.sinogram > 0]))
        
        # Add Poisson noise with the photon count desired
        self.sinogram = np.random.poisson(photon_count * np.exp(-self.sinogram))

        # Log transform to undo the exponential, and retain scale to range [0, max]
        self.sinogram[self.sinogram == 0] = 1
        self.sinogram = -np.log(self.sinogram / photon_count)
