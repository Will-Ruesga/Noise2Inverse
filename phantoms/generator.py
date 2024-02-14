import numpy as np
import matplotlib.pyplot as plt


def circle(im, center, radius, intensity: float = 1):
    for r in range(len(im)):
        for c in range(len(im[0])):
            if (r - center[0])**2 + (c - center[1])**2 <= radius**2:
                im[r][c] = intensity
    return im

def print_circle(im, center, radius):
    im = im.copy()
    for r in range(len(im)):
        for c in range(len(im[0])):
            if (r - center[0])**2 + (c - center[1])**2 <= radius**2:
                if im[r][c] == 0:
                    return None
                if (r - center[0]) ** 2 + (c - center[1]) ** 2 <= (radius - 2) ** 2:
                    im[r][c] = 0
    return im


def ellipse(im, center, width, height, intensity: float = 1):
    for r in range(len(im)):
        for c in range(len(im[0])):
            if (r - center[0])**2 / (height/2)**2 + (c - center[1])**2 / (width/2)**2 <= 1:
                im[r][c] = intensity
    return im


def ellipse_contour(im, center, width, height, border, intensity: float = 1):
    for r in range(len(im)):
        for c in range(len(im[0])):
            first_cond = (r - center[0])**2 / (height/2)**2 + (c - center[1])**2 / (width/2)**2 <= 1
            second_cond = (r - center[0]) ** 2 / ((height - border) / 2) ** 2 + (c - center[1]) ** 2 / (
                    (width - border) / 2) ** 2 <= 1
            if first_cond and not second_cond:
                im[r][c] = intensity
    return im


def check_boundaries(im, center, radius):
    try:
        if im[center[0] - radius, center[1]] == 0 or\
            im[center[0] + radius, center[1]] == 0 or\
            im[center[0], center[1] - radius] == 0 or\
            im[center[0], center[1] + radius] == 0 or\
            im[center[0], center[1]] == 0:
            return False
        return True
    except:
        return False


def point_in_mask(x, y, mask):
    return mask[x][y] == 1

NUM_CIRCLES = 100

image = np.zeros((512, 512))
image = circle(image, (256, 256), 256, intensity=1)
printed_circles = 0
failed_circle = 0
high_radius = 50
while printed_circles < NUM_CIRCLES:
    radius = np.random.randint(2, high_radius)
    center = (np.random.randint(0, 512), np.random.randint(0, 512))
    if not check_boundaries(image, center, radius):
        continue
    out = print_circle(image, center, radius)
    if out is not None:
        failed_circle = 0
        printed_circles += 1
        image = out
        if printed_circles % 10 == 0:
            print(f"{printed_circles} / {NUM_CIRCLES} , high_radius = {high_radius}")
    else:
        failed_circle += 1
    if failed_circle == 7:
        high_radius = max(1, high_radius - 3)

plt.figure(1)
plt.imshow(image, cmap='gray')
plt.show()
