import numpy as np
import matplotlib.pyplot as plt
from skimage import io, feature
from skimage.transform import hough_line, hough_line_peaks

def simple_hough(edge_image):
    height, width = edge_image.shape

    thetas = np.deg2rad(np.arange(-90, 90))
    diag_len = int(np.sqrt(height**2 + width**2))
    rhos = np.arange(-diag_len, diag_len)

    accumulator = np.zeros((len(rhos), len(thetas)))

    y_idxs, x_idxs = np.nonzero(edge_image)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t in range(len(thetas)):
            theta = thetas[t]
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            accumulator[rho + diag_len, t] += 1

    return accumulator, rhos, thetas

#peak detection in accumulator
def get_top_lines(accumulator, num_lines=2):
    acc = accumulator.copy()
    lines = []

    for _ in range(num_lines):
        idx = np.argmax(acc)
        r, t = np.unravel_index(idx, acc.shape)
        lines.append((r, t))

       
        r_min = max(0, r - 20)
        r_max = min(acc.shape[0], r + 20)
        t_min = max(0, t - 20)
        t_max = min(acc.shape[1], t + 20)

        acc[r_min:r_max, t_min:t_max] = 0

    return lines

image = io.imread("cross.png", as_gray=True)
edges = feature.canny(image)

# Apply simple Hough Transform
accumulator, rhos, thetas = simple_hough(edges)

plt.imshow(accumulator, cmap='gray')
plt.title("Hough Accumulator")
plt.xlabel("Theta")
plt.ylabel("Rho")
plt.show()


lines = get_top_lines(accumulator, num_lines=2)


plt.imshow(image, cmap='gray')

for r, t in lines:
    rho = rhos[r]
    theta = thetas[t]

    x = np.array([0, image.shape[1]])

    if np.sin(theta) != 0:
        y = (rho - x * np.cos(theta)) / np.sin(theta)
        plt.plot(x, y, '-r')

plt.title("Detected Lines Simple Implementation")
plt.xlim((0, image.shape[1]))
plt.ylim((image.shape[0], 0))
plt.show()


#Scikit-image Hough Transform
hspace, angles, distances = hough_line(edges)
accum, angles_peaks, dists_peaks = hough_line_peaks(hspace, angles, distances)

plt.imshow(image, cmap='gray')

for angle, dist in zip(angles_peaks, dists_peaks):
    x = np.array([0, image.shape[1]])
    if np.sin(angle) != 0:
        y = (dist - x * np.cos(angle)) / np.sin(angle)
        plt.plot(x, y, '-b')

plt.title("Detected Lines with scikit-image")
plt.xlim((0, image.shape[1]))
plt.ylim((image.shape[0], 0))
plt.show()