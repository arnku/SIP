import numpy as np
import matplotlib.pyplot as plt
from skimage import io, feature
from skimage.transform import rotate

# Load image
I = io.imread("textlabel_gray_small.png", as_gray=True)

# Harris corner detection
corners = feature.corner_harris(I)
coords = feature.corner_peaks(corners, min_distance=10, threshold_rel=0.02)

# --- Heuristic: pick extreme corners ---
ys = coords[:,0]
xs = coords[:,1]

left  = coords[np.argmin(xs)]
right = coords[np.argmax(xs)]

# Compute rotation angle using the two extreme corners
dy = right[0] - left[0]
dx = right[1] - left[1]

theta = np.arctan2(dy, dx)
theta_deg = np.degrees(theta)

print("Estimated rotation angle:", theta_deg)

# Rotate image to correct orientation
rotated = rotate(I, -theta_deg, resize=True)

# --- Visualization ---

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(I, cmap="gray")
plt.scatter(coords[:,1], coords[:,0], s=20, c='r')
plt.title("Detected Harris corners")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(rotated, cmap="gray")
plt.title("Rotated image")
plt.axis("off")

plt.show()