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

if np.sin(theta) != 0:
        y = (rho - x * np.cos(theta)) / np.sin(theta)
        plt.plot(x, y, '-r')

plt.title("Detected Lines Simple Implementation")
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


image = skimage.io.imread("matrikelnumre_trans.png")

blue_mask = (image[:, :, 0] <= 80) & (image[:, :, 1] <= 150) & (image[:, :, 2] >= 100)

diameter = 15
diameters = np.arange(diameter - 7, diameter + 7)

edges = skimage.feature.canny(blue_mask)

hough_res = skimage.transform.hough_circle(edges, diameters)
accum, cx, cy, radii = skimage.transform.hough_circle_peaks(hough_res, diameters, total_num_peaks=1)

plt.subplot(1, 2, 1)
plt.imshow(blue_mask, cmap='gray')
plt.title("Blue Mask")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title(f"cx: {cx[0]}, cy: {cy[0]}, radius: {radii[0]}")
plt.imshow(image)
for center_y, center_x, radius in zip(cy, cx, radii):
    circle = plt.Circle((center_x, center_y), radius, color='r', fill=False)
    plt.gca().add_patch(circle)
circle = plt.Circle((cx[0], cy[0]), 2, color='r', fill=True)
plt.gca().add_patch(circle)
plt.axis('off')
plt.show()
