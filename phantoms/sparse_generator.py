import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


####################################################################################################
#                                             CONSTANTS                                            #
####################################################################################################

RADIUS_CYLINDER = 0.5
CENTER_CYLINDER = (0.5, 0.5)

# Set seed
np.random.seed(42)


####################################################################################################
#                                              CLASS                                               #
####################################################################################################

class SparseGenerator:
    '''
        Generates and saves a foam phantom with the sparse method. Placing each sphere randomly
        if it fits.

        :param img_pixels: number of piels of the phantom
        :param num_spheres: number os small shperes (bubbles) inside the phantom
        :param prob_overlap: prbablility (0-1) of the spheres overlapping with eachother
    '''
    
    def __init__(self, img_pixels: int = 512, num_spheres: int = 1000, prob_overlap: float = 0):
        self.img_pixels = img_pixels
        self.num_spheres = num_spheres
        self.prob_overlap = prob_overlap

    @staticmethod
    def cylinder(im: npt.NDArray, center, radius: int, intensity: float = 1):
        assert len(im.shape) == 3, "Image must be 3D"
        xx, yy, zz = im.shape
        x_coords, y_coords = np.meshgrid(np.arange(xx), np.arange(yy))
        dist_squared = (x_coords - center[0]) ** 2 + (y_coords - center[1]) ** 2

        mask = dist_squared <= radius ** 2
        for z in range(zz):
            im[:, :, z][mask] = intensity
        return im

    @staticmethod
    def add_sphere(im: npt.NDArray, center, radius: int, overlap):
        im = im.copy()
        for z in range(center[2] - radius, center[2] + radius + 1):
            for x in range(center[0] - radius, center[0] + radius + 1):
                for y in range(center[1] - radius, center[1] + radius + 1):
                    if (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2:
                        if im[x][y][z] == 0 and not overlap:
                            return None
                        if (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2 <= (radius - 2) ** 2:
                            im[x][y][z] = 0
        return im

    @staticmethod
    def check_boundaries(im, center, radius, overlap):
        if overlap:
            return True
        cond1 = (im[center[0] - radius, center[1], center[2]] == 0)
        cond2 = (im[center[0] + radius, center[1], center[2]] == 0)
        cond3 = (im[center[0], center[1] - radius, center[2]] == 0)
        cond4 = (im[center[0], center[1] + radius, center[2]] == 0)
        cond5 = (im[center[0], center[1], center[2]] == 0)
        cond6 = (im[center[0], center[1], center[2] - radius] == 0)
        cond7 = (im[center[0], center[1], center[2] + radius] == 0)
        try:
            if cond1 or cond2 or cond3 or cond4 or cond5 or cond6 or cond7:
                return False
            return True
        except:
            return False

    def create_phantom(self, overlap_flag=False):
        phantom = np.zeros((self.img_pixels, self.img_pixels, self.img_pixels))
        # Generate the big circle
        phantom = self.cylinder(phantom, (self.img_pixels // 2, self.img_pixels // 2), self.img_pixels // 2, intensity=1)

        printed_circles = 0
        failed_circle = 0
        high_radius = self.img_pixels // 10
        while printed_circles < self.num_spheres:
            # Generate random circle parameters
            rndm_radius = np.random.randint(2, high_radius)
            rndm_center = (
                np.random.randint(rndm_radius, self.img_pixels - rndm_radius), np.random.randint(rndm_radius, self.img_pixels - rndm_radius),
                np.random.randint(rndm_radius, self.img_pixels - rndm_radius))

            overlap = np.random.rand() < self.prob_overlap

            if not self.check_boundaries(phantom, rndm_center, rndm_radius, overlap):
                continue

            out = self.add_sphere(phantom, rndm_center, rndm_radius, overlap)

            if out is not None:
                # Circle was added successfully, there was an empty space to put it
                failed_circle = 0
                printed_circles += 1
                phantom = out
                if printed_circles % 100 == 0:
                    print(f"{printed_circles} / {self.num_spheres} , high_radius = {high_radius}")
            else:
                failed_circle += 1

            # If fails many times, reduce the maximum radius to make it easier
            if failed_circle == 7:
                high_radius = max(1, high_radius - 2)

        # Save phantom
        if overlap_flag:
            np.save('./phantoms/save/sparse_phantom_overlap.npy', phantom)
        else:
            np.save('./phantoms/save/sparse_phantom.npy', phantom)