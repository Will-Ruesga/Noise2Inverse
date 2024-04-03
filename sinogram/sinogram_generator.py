import astra
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

NUM_PROJECTIONS = 1024
PHOTON_COUNT = 10
ATTENUATION = 2.76


def generate_noise(sinogram, photon_count):
    exp_sinogram = np.exp(-sinogram, dtype=np.float32)
    noisy_sinogram = np.random.poisson(exp_sinogram * photon_count)
    noisy_sinogram[noisy_sinogram == 0] = 1
    noisy_sinogram = -np.log(noisy_sinogram / photon_count, dtype=np.float32)
    noise = noisy_sinogram - sinogram
    return noisy_sinogram, noise


# Create a sinogram from a phantom
phantom = np.load("../phantoms/foam_phantom.npy")

angles = np.linspace(0, np.pi, NUM_PROJECTIONS,False)
max_distance = int(np.sqrt(2) * phantom.shape[1]) + 1
vol_geom = astra.create_vol_geom(phantom.shape)
proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, phantom.shape[0], max_distance, angles)
proj_id, proj_data = astra.create_sino3d_gpu(phantom, proj_geom, vol_geom)

sinograms = np.array(proj_data)
sinograms_noisy = sinograms.copy() * ATTENUATION
sinograms_noisy, noise = generate_noise(sinograms_noisy, PHOTON_COUNT)
sinograms_noisy /= ATTENUATION
sinograms_noisy = sinograms_noisy.astype(np.float32)

# Plot 4 sinograms
plt.figure()
for i in range(4):
    plt.subplot(2, 2, i + 1)
    slice_to_plot = (i + 1) * phantom.shape[0] // 5
    plt.imshow(sinograms[slice_to_plot], cmap='gray')
    plt.axis('off')
    plt.title(f"Slice {slice_to_plot}")
plt.suptitle("Sinograms")
plt.show()

# Plot 4 noisy sinograms
plt.figure()
for i in range(4):
    plt.subplot(2, 2, i + 1)
    slice_to_plot = (i + 1) * phantom.shape[0] // 5
    plt.imshow(sinograms_noisy[slice_to_plot], cmap='gray')
    plt.axis('off')
    plt.title(f"Slice {slice_to_plot}")
plt.suptitle("Noisy sinograms")
plt.show()
