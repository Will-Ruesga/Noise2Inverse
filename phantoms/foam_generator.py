import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from collections import namedtuple


def cylinder(im: npt.NDArray, center, radius: float, intensity: float = 1):
    assert len(im.shape) == 3, "Image must be 3D"
    xx, yy, zz = im.shape
    center_pixels = (int(round(center[0] * IMG_PIXELS)), int(round(center[1] * IMG_PIXELS)))
    radius_pixels = int(round(radius * IMG_PIXELS))
    x_coords, y_coords = np.meshgrid(np.arange(xx), np.arange(yy))
    dist_squared = (x_coords - center_pixels[0]) ** 2 + (y_coords - center_pixels[1]) ** 2

    mask = dist_squared <= radius_pixels ** 2
    for z in range(zz):
        im[:, :, z][mask] = intensity
    return im


def inside_foam(foam_center, sphere_center):
    return (foam_center[0] - sphere_center[0])**2 + (foam_center[1] - sphere_center[1])**2 <= 0.5**2


def distance_to_border(foam_center, foam_radius, sphere_center):
    distance_radius_foam = foam_radius - np.sqrt((foam_center[0] - sphere_center[0]) ** 2 + (foam_center[1] - sphere_center[1]) ** 2)
    distance_height = min(sphere_center[2], 1 - sphere_center[2])
    return np.min([distance_radius_foam, distance_height]) - 1 / IMG_PIXELS


def distances_centers(buffer, sphere_center, overlap: bool = False):
    distance_border = distance_to_border((0.5, 0.5), 0.5, sphere_center)
    if distance_border <= 0:
        return None
    distances = [distance_border]

    for s in buffer:
        dist_radius = np.sqrt((s.center_x - sphere_center[0]) ** 2 + (s.center_y - sphere_center[1]) ** 2 + (s.center_z - sphere_center[2]) ** 2)
        dist = max(dist_radius - s.radius, 0)
        if overlap:
            dist = dist * (1 + np.random.rand() * 3)
        if dist <= 0:
            return None
        distances.append(dist)
    return distances


def print_sphere(im, center, radius):
    center_pixels = (int(round(center[0] * IMG_PIXELS)), int(round(center[1] * IMG_PIXELS)), int(round(center[2] * IMG_PIXELS)))
    radius_pixels = int(round(radius * IMG_PIXELS))

    for z in range(center_pixels[2] - radius_pixels, center_pixels[2] + radius_pixels + 1):
        for x in range(center_pixels[0] - radius_pixels, center_pixels[0] + radius_pixels + 1):
            for y in range(center_pixels[1] - radius_pixels, center_pixels[1] + radius_pixels + 1):
                if (x - center_pixels[0])**2 + (y - center_pixels[1])**2 + (z - center_pixels[2])**2 <= radius_pixels**2:
                    im[x][y][z] = 0
    return im


NUM_SHPERES = 5000

IMG_PIXELS = 512
image = np.zeros((IMG_PIXELS, IMG_PIXELS, IMG_PIXELS))

PROB_OVERLAP = 0.1
sphere = namedtuple("sphere", ["center_x", "center_y", "center_z", "radius"])

# Generate the big cylinder
image = cylinder(image, (0.5, 0.5), 0.5, intensity=1)
print("Finished cylinder")

buffer = []
generated_spheres = 0

while generated_spheres < NUM_SHPERES:
    sphere_center = np.random.uniform(2 / IMG_PIXELS, 1 - 2 / IMG_PIXELS, 3)
    overlap = np.random.rand() < PROB_OVERLAP

    d = distances_centers(buffer, sphere_center, overlap)
    if d is None or not inside_foam((0.5, 0.5), (sphere_center[0], sphere_center[1])):
        continue

    min_radius = 1 / IMG_PIXELS
    max_radius = min(min(d), 0.5 * 0.2)

    sphere_radius = sphere_radius = np.random.uniform(low=min_radius, high=max_radius)

    image = print_sphere(image, sphere_center, sphere_radius)

    sphere_ = sphere(sphere_center[0], sphere_center[1], sphere_center[2], sphere_radius)
    buffer.append(sphere_)

    generated_spheres += 1
    if generated_spheres % 1000 == 0:
        print(f"Generated {generated_spheres} / {NUM_SHPERES} spheres")


image = image.transpose(2, 0, 1)
fig = plt.figure()
plt.subplot(2, 2, 1)
plt.title("Slice 2")
plt.imshow(image[2, :, :], cmap='gray')
plt.subplot(2, 2, 2)
plt.title(f"Slice {IMG_PIXELS // 4}")
plt.imshow(image[IMG_PIXELS // 4, :, :], cmap='gray')
plt.subplot(2, 2, 3)
plt.title(f"Slice {IMG_PIXELS // 2}")
plt.imshow(image[IMG_PIXELS // 2, :, :], cmap='gray')
plt.subplot(2, 2, 4)
plt.title(f"Slice {IMG_PIXELS * 3 // 4}")
plt.imshow(image[(3 * IMG_PIXELS) // 4, :, :], cmap='gray')
plt.savefig("phantom_diff_levels.png")
plt.show()

fig = plt.figure()
cnt = 0
for i in [-1, 0, 1, 2]:
    plt.subplot(2, 2, cnt + 1)
    plt.title(f"Slice {i + (IMG_PIXELS // 2)}")
    plt.imshow(image[i + (IMG_PIXELS // 2), :, :], cmap='gray')
    cnt += 1
plt.savefig("phantom_seq.png")
plt.show()
