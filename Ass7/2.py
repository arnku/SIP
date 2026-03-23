import numpy as np
import scipy
import matplotlib.pyplot as plt
import skimage

IMG_COL = skimage.io.imread('matrikelnumre_nat.png')
IMG = skimage.color.rgb2gray(IMG_COL)

def harries_corner_peaks(image: np.ndarray, method: str, sigma: float, k: float = 0.04, eps: float = 0.0001, top_corners: int = 4, min_distance: int = 100) -> np.ndarray:
    harris_response = skimage.feature.corner_harris(image, method=method, sigma=sigma, k=k, eps=eps)
    corners = skimage.feature.corner_peaks(harris_response, num_peaks=top_corners, min_distance=min_distance)
    return corners

img_blur = scipy.ndimage.gaussian_filter(IMG, sigma=1)
threshold = skimage.filters.threshold_otsu(img_blur)
img_mask = np.where(img_blur > threshold, 1, 0)
img_hull = skimage.morphology.convex_hull_image(img_mask)

peaks = harries_corner_peaks(img_hull, method="k", sigma=1, k=0.04, top_corners=4, min_distance=20)

plt.subplot(1, 3, 1)
plt.title('Blurred Image')
plt.imshow(img_blur, cmap='gray')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.title('Mask Image')
plt.imshow(img_hull, cmap='gray')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.title('Harris Corners')
plt.imshow(IMG, cmap='gray')
plt.scatter(peaks[:, 1], peaks[:, 0], color='red', s=20, marker='x')
plt.axis('off')
plt.show()

top_left     = peaks[np.argmin(peaks[:, 0] + peaks[:, 1])]
top_right    = peaks[np.argmin(peaks[:, 0] - peaks[:, 1])]
bottom_left  = peaks[np.argmax(peaks[:, 0] - peaks[:, 1])]
bottom_right = peaks[np.argmax(peaks[:, 0] + peaks[:, 1])]

width = int(np.mean([np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left)]))
height = int(np.mean([np.linalg.norm(bottom_right - top_right), np.linalg.norm(bottom_left - top_left)]))

dst_corners = [
    [0,         0],           # top-left
    [width - 1, 0],           # top-right
    [width - 1, height - 1],  # bottom-right
    [0,         height - 1]   # bottom-left
]

src_corners = np.array([
    [top_left[1],     top_left[0]],
    [top_right[1],    top_right[0]],
    [bottom_right[1], bottom_right[0]],
    [bottom_left[1],  bottom_left[0]]
], dtype=np.float64)


tform = skimage.transform.ProjectiveTransform()
tform.estimate(dst_corners, src_corners)

print(tform.params)

img_warped = skimage.transform.warp(IMG_COL, tform, output_shape=(height, width))

plt.imshow(img_warped, cmap='gray')
plt.title('Warped Image')
plt.axis('off')
plt.show()

plt.imsave('matrikelnumre_trans.png', img_warped)