import astra
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


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
        self.detector_shape = (self.phantom.shape[0], int(np.ceil(np.sqrt(self.phantom.shape[1]**2 + self.phantom.shape[2]**2))))

        # Sinogram variables
        self.sinogram = None
        self.pg = None
        self.vg = None
        self.absorption = None
        self.attenuation = None
        self.photon_count = None
        
        # Split sinogram
        self.num_splits = None
        self.split_sinograms = []


    ####################################################################################################
    def generate(self):
        """
        Creates volume and projection geometry in order to generate the sinogram.
        """
        # Creage volumme and projection geometries
        angles = np.linspace(0, np.pi, self.num_proj, False)
        self.vg = astra.create_vol_geom(self.phantom.shape)
        self.pg = astra.create_proj_geom('parallel3d', 1.0, 1.0, int(self.detector_shape[0]), int(self.detector_shape[1]), angles)

        # Create sinogram data
        pid , pdata = astra.create_sino3d_gpu(self.phantom, self.pg, self.vg)
        astra.projector.delete(pid)
        self.sinogram = pdata


    ####################################################################################################
    def split_data(self, num_splits: int):
        """
        Create a list of sinograms with the data split in num_splits parts.
        """
        # Initialize
        split_data = []
        self.num_splits = num_splits

        # Generate K splits of the sinogram
        for proj in self.sinogram:
            proj_in_splits = np.array([proj[i::num_splits] for i in range(num_splits)])
            split_data.append(proj_in_splits)

        # Save the splits
        split_data = np.array(split_data)
        self.split_sinograms = split_data.transpose(1, 0, 2, 3)


    ####################################################################################################
    def reconstruct(self, rec_algorithm: str = 'FBP_CUDA'):
        """
        Performs the sinogram reconstruction.

        :param sinogram: The sinograms
        :param rec_algorithm: Type of reconstruction algorithm to use

        :return: The reconstruction of the sinogram
        """
        # Initalize angles and volume geometry
        reconstruction = []
        angles = np.linspace(0, np.pi, self.num_proj, False)
        vg = astra.create_vol_geom(self.phantom.shape[1:])

        # For each slice
        for proj in self.sinogram:
            # Create projection geometry and ids for sinogram and reconstruction
            pg = astra.create_proj_geom('parallel', 1.0, self.detector_shape[1], angles)
            sinogram_id = astra.data2d.create('-sino', pg, proj)
            recon_id = astra.data2d.create('-vol', vg, 0)

            # Create the algorithm
            config = astra.astra_dict(rec_algorithm)
            config['ReconstructionDataId'] = recon_id
            config['ProjectionDataId'] = sinogram_id
            alg_id = astra.algorithm.create(config)

            # Run the reconstruction and store the result
            astra.algorithm.run(alg_id, self.num_iter)
            rec = astra.data2d.get(recon_id)
            reconstruction.append(rec[::-1])

            # Clean up the memory
            astra.algorithm.delete(alg_id)
            astra.data2d.delete(recon_id)
            astra.data2d.delete(sinogram_id)
        return reconstruction
    

    ####################################################################################################
    def reconstruct_splits(self, sinograms: list, rec_algorithm: str = 'FBP_CUDA'):
        """
        Reconstruction of each k split made from the sinogram.

        :param sinogram: List of the sinograms
        :param rec_algorithm: Type of reconstruction algorithm to use

        :return: A list of reconstructions that correspond to each sinogram split
        """
        # Initialize volume geometry 
        reconstruction = []
        vg = astra.create_vol_geom(self.phantom.shape[1:])

        # For each split
        for k_split, sino in enumerate(sinograms):
            # Generate angles
            rec_split = []
            angles = np.linspace(np.pi / self.num_proj * k_split, np.pi, self.num_proj // self.num_splits, False)

            # For each slice
            for proj in sino:
                # Create projection geometry and ids for sinogram and reconstruction
                pg = astra.create_proj_geom('parallel', 1.0, self.detector_shape[1], angles)
                sinogram_id = astra.data2d.create('-sino', pg, proj)
                recon_id = astra.data2d.create('-vol', vg, 0)

                # Create the algorithm
                config = astra.astra_dict(rec_algorithm)
                config['ReconstructionDataId'] = recon_id
                config['ProjectionDataId'] = sinogram_id
                alg_id = astra.algorithm.create(config)

                # Run the reconstruction and store the result
                astra.algorithm.run(alg_id, self.num_iter)
                rec = astra.data2d.get(recon_id)
                rec_split.append(rec[::-1])

                # Clean up the memory
                astra.algorithm.delete(alg_id)
                astra.data2d.delete(recon_id)
                astra.data2d.delete(sinogram_id)
            reconstruction.append(np.array(rec_split))
        return reconstruction



    ####################################################################################################
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
        # self.sinogram = self.sinogram * (1 - attenuation)
        self.sinogram = self.sinogram / attenuation

        # Compute and save absortion
        self.absorption = 1 - np.mean(np.exp(-self.sinogram[self.sinogram > 0]))
        print(f"\n\n Absorption: {self.absorption}")
        
        # Add Poisson noise with the photon count desired
        self.sinogram = np.random.poisson(photon_count * np.exp(-self.sinogram))

        # Log transform to undo the exponential, and retain scale to range [0, max]
        self.sinogram[self.sinogram == 0] = 1
        self.sinogram = -np.log(self.sinogram / photon_count)
        # self.sinogram = self.sinogram / (1 - attenuation)


    ####################################################################################################
    def add_gaussian_noise(self, std: float = 1):
        """
        Generate a noisy sinogram from a sinogram with Gaussian noise
        with mean (sigma) 0.

        :param std: Standard deviation.
        """
        # Sum the Gaussian noise to the sinogram
        self.sinogram += np.random.normal(0, std, self.sinogram.shape)


    ####################################################################################################
    def add_non_independent_noise(self, non_ind_noise, attenuation):
        """
            Adds the non-independent noise to the sinogram with the desired
            attenuation.

            :param non_ind_noise: Noise wave to add.
            :param attenuation: Attenuation factor for the wave.
        """
        # Apply attenuation
        att_noise = non_ind_noise * (attenuation * np.max(self.sinogram))
        att_noise = np.array([att_noise for _ in range(self.sinogram.shape[0])])
        
        # Add the non-independent noise
        self.sinogram = self.sinogram.astype(np.float32) + att_noise.astype(np.float32)

        # Normalize the combined image to fit within 0-1
        # self.sinogram = (noised_sin - np.min(noised_sin)) / (np.max(noised_sin) - np.min(noised_sin))


    ####################################################################################################
    def add_gaussian_noise(self, bound: Optional[Tuple[float, float]] = None, std: float = 1):
        """
            Adds the non-zero mean (Gaussian) noise to the sinogram.

            :param std_mean: Mean of the Standard deviation.
            :param std: Standard deviation.
        """
        # Generate non zero mean gaussian noise
        if bound is None:
            mean_array = 0
        else:
            a, b = bound
            mean_array = np.random.uniform(a, b, self.sinogram.shape)

        # Add it to sinogram
        self.sinogram += np.random.normal(mean_array, std, self.sinogram.shape)


    ####################################################################################################
    def add_ring_artifact(self, position: int, offset: float = 0.5):
        """
            Adds a line artifact to the sinogram.

            :param position: The column index in the sinogram where the artifact will be added.
            :param intensity: The intensity of the artifact.
        """
        # Add the ring artifact
        offset = offset * np.max(self.sinogram)
        self.sinogram[:, :, position] += offset

    ####################################################################################################
    def add_zinger_artifact(self, num_zingers, intensity):
        """
            Adds zinger artifacts to the sinogram.

            Parameters:
            num_zingers (int): The number of zinger artifacts to add.
            intensity (float): The intensity of the zinger artifacts.

            Returns:
            ndarray: The sinogram with the added zinger artifacts.
        """
        # Get the dimensions of the sinogram
        depth, rows, cols = self.sinogram.shape

        intensity = intensity * np.max(self.sinogram)

        # Add zinger artifacts at random positions
        for _ in range(num_zingers):
            # Randomly select a position in the sinogram
            row = np.random.randint(0, rows)
            col = np.random.randint(0, cols)
            
            # Add the intensity to create the zinger artifact
            self.sinogram[:, row, col] += intensity