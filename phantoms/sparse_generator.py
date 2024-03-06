import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


def cylinder(im: npt.NDArray, center, radius: int, intensity: float = 1):
    assert len(im.shape) == 3, "Image must be 3D"
    xx, yy, zz = im.shape
    x_coords, y_coords = np.meshgrid(np.arange(xx), np.arange(yy))
    dist_squared = (x_coords - center[0]) ** 2 + (y_coords - center[1]) ** 2

    mask = dist_squared <= radius ** 2
    for z in range(zz):
        im[:, :, z][mask] = intensity
    return im


def add_sphere(im: npt.NDArray, center, radius: int, overlap):
    im = im.copy()
    for z in range(center[2] - radius, center[2] + radius + 1):
        for x in range(center[0] - radius, center[0] + radius + 1):
            for y in range(center[1] - radius, center[1] + radius + 1):
                if (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2:
                    if im[x][y][z] == 0 and overlap:
                        return None
                    if (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2 <= (radius - 2) ** 2:
                        im[x][y][z] = 0
    return im


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


NUM_CIRCLES = 1000

SIZES = 512
HEIGHT = 32
image = np.zeros((SIZES, SIZES, HEIGHT))

PROB_OVERLAP = 0.1

# Generate the big circle
image = cylinder(image, (SIZES // 2, SIZES // 2), SIZES // 2, intensity=1)

printed_circles = 0
failed_circle = 0
high_radius = HEIGHT // 2
while printed_circles < NUM_CIRCLES:
    #Generate random circle parameters
    rndm_radius = np.random.randint(2, high_radius)
    rndm_center = (np.random.randint(rndm_radius, SIZES - rndm_radius), np.random.randint(rndm_radius, SIZES - rndm_radius), np.random.randint(rndm_radius, HEIGHT - rndm_radius))

    overlap = np.random.rand() < PROB_OVERLAP

    if not check_boundaries(image, rndm_center, rndm_radius, overlap):
        continue

    out = add_sphere(image, rndm_center, rndm_radius, overlap)

    if out is not None:
        # Circle was added successfully, there was an empty space to put it
        failed_circle = 0
        printed_circles += 1
        image = out
        if printed_circles % 10 == 0:
            print(f"{printed_circles} / {NUM_CIRCLES} , high_radius = {high_radius}")
    else:
        failed_circle += 1

    # If fails many times, reduce the maximum radius to make it easier
    if failed_circle == 7:
        high_radius = max(1, high_radius - 1)

image = image.transpose(2, 0, 1)
fig = plt.figure()
plt.subplot(2, 2, 1)
plt.title("Slice 2")
plt.imshow(image[2, :, :], cmap='gray')
plt.subplot(2, 2, 2)
plt.title(f"Slice {HEIGHT // 4}")
plt.imshow(image[HEIGHT // 4, :, :], cmap='gray')
plt.subplot(2, 2, 3)
plt.title(f"Slice {HEIGHT // 2}")
plt.imshow(image[HEIGHT // 2, :, :], cmap='gray')
plt.subplot(2, 2, 4)
plt.title(f"Slice {HEIGHT * 3 // 4}")
plt.imshow(image[(3*HEIGHT) // 4, :, :], cmap='gray')
plt.savefig("phantom_diff_levels.png")
plt.show()

fig = plt.figure()
cnt = 0
for i in [-1, 0, 1, 2]:
    plt.subplot(2, 2, cnt + 1)
    plt.title(f"Slice {i + (HEIGHT // 2)}")
    plt.imshow(image[i + (HEIGHT // 2), :, :], cmap='gray')
    cnt += 1
plt.savefig("phantom_seq.png")
plt.show()
