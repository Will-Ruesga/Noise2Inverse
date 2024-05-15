import matplotlib.pyplot as plt
import time
import numpy as np

from sparse_generator import SparseGenerator
from foam_generator import FoamGenerator

"""
    This script will generate images to comapre the different phantoms we make, which are:

    - Sparse phantom

    - Foam phantom

    Both of them have overlapping flags, which makes the bubbles in them be able to overlap.

"""
####################################################################################################
#                                             CONSTANTS                                            #
####################################################################################################

NUM_SHPERES = 1000
IMG_PIXELS = 256
OVERLAPPING = 0.2

####################################################################################################
#                                             FUNCTIONS                                            #
####################################################################################################

def plot_phantoms(image_sparse, image_foam, image_sparse_overlap, image_foam_overlap):
    '''
        Plots slices of the four different generated pahntoms.
    '''
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.title("Sparse")
    plt.imshow(image_sparse[IMG_PIXELS // 2, :, :], cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.title("Geometrical")
    plt.imshow(image_foam[IMG_PIXELS // 2, :, :], cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.title("Sparse with overlap")
    plt.imshow(image_sparse_overlap[IMG_PIXELS // 2, :, :], cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.title("Geometrical with overlap")
    plt.imshow(image_foam_overlap[IMG_PIXELS // 2, :, :], cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("./figures/img_phantom_comparison.png", dpi=600)
    plt.show()


####################################################################################################
#                                               MAIN                                               #
####################################################################################################

if __name__ == "__main__":
    
    ### SPARSE ###
    start = time.time()
    generator = SparseGenerator(img_pixels=IMG_PIXELS, num_spheres=NUM_SHPERES)
    image_sparse = generator.create_phantom()
    image_sparse = image_sparse.transpose(2, 0, 1)
    print(f"Finished sparse, time: {(time.time() - start)}")

    ### SPARSE WITH OVERLAPPING ###
    start = time.time()
    generator = SparseGenerator(img_pixels=IMG_PIXELS, num_spheres=NUM_SHPERES, prob_overlap=OVERLAPPING)
    image_sparse_overlap = generator.create_phantom()
    image_sparse_overlap = image_sparse_overlap.transpose(2, 0, 1)
    print(f"Finished sparse with overlap, time: {(time.time() - start)}")

    ### FOAM ###
    start = time.time()
    generator = FoamGenerator(img_pixels=IMG_PIXELS, num_spheres=NUM_SHPERES)
    image_foam = generator.create_phantom()
    image_foam = image_foam.transpose(2, 0, 1)
    print(f"Finished foam, time: {(time.time() - start)}")

    ### FOAM WITH OVERLAPPING ###
    start = time.time()
    generator = FoamGenerator(img_pixels=IMG_PIXELS, num_spheres=NUM_SHPERES, prob_overlap=OVERLAPPING)
    image_foam_overlap = generator.create_phantom()
    image_foam_overlap = image_foam_overlap.transpose(2, 0, 1)
    print(f"Finished foam with overlap, time: {(time.time() - start)}")
    
    plot_phantoms(image_sparse, image_foam, image_sparse_overlap, image_foam_overlap)