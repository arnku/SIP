import numpy as np
import matplotlib.pyplot as plt
import skimage

def simple_hough(img: np.ndarray):
    height, width = img.shape

    thetas = np.deg2rad(np.arange(0, 180))
    diag_len = int(np.sqrt(height**2 + width**2))
    rhos = np.arange(-diag_len, diag_len)
    
    accumulator = np.zeros((len(rhos), len(thetas)))

    for x in range(width):
        for y in range(height):
            if not img[y, x]:
                continue
            for i, theta in enumerate(thetas):
                rho = int(round(x * np.cos(theta) + y * np.sin(theta)))
                rho_idx = rho + diag_len
                accumulator[rho_idx, i] += 1

    return accumulator, rhos, thetas

def get_top_lines(accumulator, num_lines=2):
    acc = accumulator.copy()
    lines = []
    for _ in range(num_lines):
        top_point = np.where(acc == acc.max())
        rho, theta = top_point[0][0], top_point[1][0]
        lines.append((rho, theta))
       
        r_min = max(0, rho - 20)
        r_max = min(acc.shape[0], rho + 20)
        t_min = max(0, theta - 20)
        t_max = min(acc.shape[1], theta + 20)

        acc[r_min:r_max, t_min:t_max] = 0

    return lines

image = skimage.io.imread("cross.png", as_gray=True)
edges = skimage.feature.canny(image)

accumulator, rhos, thetas = simple_hough(edges)

plt.imshow(accumulator, cmap='gray')
plt.title("Hough Accumulator")
plt.xlabel("Theta")
plt.ylabel("Rho")
plt.show()


lines = get_top_lines(accumulator, num_lines=2)

plt.imshow(image, cmap='gray')
for r, t in lines:
    x = np.array([0, image.shape[1]])
    y = (rhos[r] - x * np.cos(thetas[t])) / np.sin(thetas[t])
    plt.plot(x, y, '-r')
plt.title("Detected Lines (Simple Implementation)")
plt.xlim((0, image.shape[1]))
plt.ylim((image.shape[0], 0))
plt.show()


hspace, angles, distances = skimage.transform.hough_line(edges)
accum, angles_peaks, dists_peaks = skimage.transform.hough_line_peaks(hspace, angles, distances)

plt.imshow(image, cmap='gray')
for angle, dist in zip(angles_peaks, dists_peaks):
    x = np.array([0, image.shape[1]])
    y = (dist - x * np.cos(angle)) / np.sin(angle)
    plt.plot(x, y, '-b')
plt.title("Detected Lines (scikit-image)")
plt.xlim((0, image.shape[1]))
plt.ylim((image.shape[0], 0))
plt.show()