import matplotlib.pyplot as plt
import time
import numpy as np

from sparse_generator import SparseGenerator
from foam_generator import FoamGenerator

# Set seed
np.random.seed(42)


if __name__ == "__main__":
    NUM_SHPERES = 1000
    IMG_PIXELS = 256

    time_sparse = time.time()
    generator = SparseGenerator(img_pixels=IMG_PIXELS, num_spheres=NUM_SHPERES, prob_overlap=0)
    image_sparse = generator.run()
    image_sparse = image_sparse.transpose(2, 0, 1)
    time_sparse = (time.time() - time_sparse)
    print(f"Finished sparse, time: {time_sparse}")
    time_foam = time.time()
    generator = FoamGenerator(img_pixels=IMG_PIXELS, num_spheres=NUM_SHPERES, prob_overlap=0)
    image_foam = generator.run()
    image_foam = image_foam.transpose(2, 0, 1)
    time_foam = (time.time() - time_foam)
    print(f"Finished foam, time: {time_foam}")

    time_sparse_overlap = time.time()
    generator = SparseGenerator(img_pixels=IMG_PIXELS, num_spheres=NUM_SHPERES, prob_overlap=0.2)
    image_sparse_overlap = generator.run()
    image_sparse_overlap = image_sparse_overlap.transpose(2, 0, 1)
    time_sparse_overlap = (time.time() - time_sparse_overlap)
    print(f"Finished sparse with overlap, time: {time_sparse_overlap}")
    time_foam_overlap = time.time()
    generator = FoamGenerator(img_pixels=IMG_PIXELS, num_spheres=NUM_SHPERES, prob_overlap=0.2)
    image_foam_overlap = generator.run()
    image_foam_overlap = image_foam_overlap.transpose(2, 0, 1)
    time_foam_overlap = (time.time() - time_foam_overlap)
    print(f"Finished foam with overlap, time: {time_foam_overlap}")
    
    plt.figure()
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
    plt.savefig("different_methods.png", dpi=600)
    plt.show()
    
    print(f"Time sparse: {time_sparse}, time foam: {time_foam}, time sparse with overlap: {time_sparse_overlap}, time foam with overlap: {time_foam_overlap}")
    print("Finished")