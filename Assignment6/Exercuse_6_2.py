import numpy as np
import matplotlib.pyplot as plt
from skimage import io, feature, filters, morphology, measure
from skimage.transform import rotate

# Load image
I = io.imread("textlabel_gray_small.png", as_gray=True)

th = filters.threshold_otsu(I)
mask = I > th

mask = morphology.binary_closing(mask, morphology.square(15))
mask = morphology.remove_small_holes(mask, area_threshold=5000)
mask = morphology.remove_small_objects(mask, min_size=5000)

labels = measure.label(mask)
regions = measure.regionprops(labels)
largest_label = max(regions, key=lambda r: r.area).label
mask = labels == largest_label
R = feature.corner_harris(mask.astype(float), sigma=2)
coords = feature.corner_peaks(R, min_distance=20, threshold_rel=0.1)

ys = coords[:, 0]
xs = coords[:, 1]

top_right = coords[np.argmax(xs - ys)]
bottom_right = coords[np.argmax(xs + ys)]
bottom_left = coords[np.argmin(xs - ys)]
top_left_like = coords[np.argmin(xs + ys)]  # this will be near the clipped corner

dy = bottom_right[0] - top_right[0]
dx = bottom_right[1] - top_right[1]

theta_deg = np.degrees(np.arctan2(dy, dx))
print("Estimated edge angle:", theta_deg)

rotated = rotate(I, -theta_deg, resize=True, mode="constant", cval=0)

h, w = rotated.shape
crop = rotated[int(0.2*h):int(0.8*h), int(0.2*w):int(0.2*w + 0.6*w)]

ink = 1.0 - crop  # dark text gives larger values
top_ink = np.sum(ink[:ink.shape[0]//2, :])
bottom_ink = np.sum(ink[ink.shape[0]//2:, :])

if bottom_ink > top_ink:
    rotated = rotate(rotated, 180, resize=True, mode="constant", cval=0)
    print("Applied 180 degree flip")

plt.figure(figsize=(11, 4))

plt.subplot(1, 3, 1)
plt.imshow(I, cmap="gray")
plt.title("Original image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.scatter(coords[:, 1], coords[:, 0], c="r", s=30)
plt.scatter(
    [top_right[1], bottom_right[1], bottom_left[1], top_left_like[1]],
    [top_right[0], bottom_right[0], bottom_left[0], top_left_like[0]],
    c="yellow", s=80
)
plt.title("Harris corners on label mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(rotated, cmap="gray")
plt.title("Final rotated image")
plt.axis("off")

plt.show()