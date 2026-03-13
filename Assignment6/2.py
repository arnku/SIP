import numpy as np
import matplotlib.pyplot as plt
import skimage

# Load image
IMG = skimage.io.imread("textlabel_gray_small.png", as_gray=True)
mask = IMG > skimage.filters.threshold_mean(IMG)

harris = skimage.feature.corner_harris(mask, method='k', k=0.05, eps=1)
coords = skimage.feature.corner_peaks(harris, min_distance=25, threshold_rel=0.1)

plt.imshow(mask, cmap="gray")
plt.scatter(coords[:, 1], coords[:, 0], c="r", s=30)
plt.title("Harris corners on label mask")
plt.axis("off")
plt.show()

ERROOR = 10

left_most = coords[np.argmin(coords[:, 1])]
left_mosts = coords[np.abs(coords[:, 1] - left_most[1]) < ERROOR]
top_most = coords[np.argmin(coords[:, 0])]
top_mosts = coords[np.abs(coords[:, 0] - top_most[0]) < ERROOR]
right_most = coords[np.argmax(coords[:, 1])]
right_mosts = coords[np.abs(coords[:, 1] - right_most[1]) < ERROOR]
bottom_most = coords[np.argmax(coords[:, 0])]
bottom_mosts = coords[np.abs(coords[:, 0] - bottom_most[0]) < ERROOR]

edge_points = np.concatenate([left_mosts, top_mosts, right_mosts, bottom_mosts])

plt.imshow(mask, cmap="gray")
plt.scatter(edge_points[:, 1], edge_points[:, 0], c="r", s=30, label="Edge points")
plt.title("Corner groups")
plt.axis("off")
plt.show()

def edge_angle(points):
    if len(points) < 2:
        return None
    dy = np.diff(points[:, 0])
    dx = np.diff(points[:, 1])
    return np.median(np.degrees(np.arctan2(dy, dx)))

left_angle_err = 90 - edge_angle(left_mosts) % 90
top_angle_err = 90 - edge_angle(top_mosts) % 90
right_angle_err = 90 - edge_angle(right_mosts) % 90
bottom_angle_err = 90 - edge_angle(bottom_mosts) % 90
angle_err = np.median([left_angle_err, top_angle_err, right_angle_err, bottom_angle_err])

print("Estimated edge angles (degrees):")
print(f"Left edge: {left_angle_err:.2f}")
print(f"Top edge: {top_angle_err:.2f}")
print(f"Right edge: {right_angle_err:.2f}")
print(f"Bottom edge: {bottom_angle_err:.2f}")
print(f"Median edge angle error: {angle_err:.2f}")

left_edge_length = np.ptp(left_mosts[:, 0])
top_edge_length = np.ptp(top_mosts[:, 1])
right_edge_length = np.ptp(right_mosts[:, 0])
bottom_edge_length = np.ptp(bottom_mosts[:, 1])

print("Estimated edge lengths (pixels):")
print(f"Left edge length: {left_edge_length:.2f}")
print(f"Top edge length: {top_edge_length:.2f}")
print(f"Right edge length: {right_edge_length:.2f}")
print(f"Bottom edge length: {bottom_edge_length:.2f}")

shortest_edge = min(left_edge_length, top_edge_length, right_edge_length, bottom_edge_length)
if shortest_edge == right_edge_length:
    pass  # already aligned
elif shortest_edge == left_edge_length:
    angle_err = 180 - angle_err
elif shortest_edge == top_edge_length:
    angle_err = -90 - angle_err
elif shortest_edge == bottom_edge_length:
    angle_err = 90 - angle_err

image_rotated = skimage.transform.rotate(IMG, angle_err, resize=True)

plt.imshow(image_rotated, cmap="gray")
plt.title(f"Rotated image (angle correction: {angle_err:.2f} degrees)")
plt.axis("off")
plt.show()