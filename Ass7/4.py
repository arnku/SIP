import numpy as np
import matplotlib.pyplot as plt

from skimage import io, color, measure, morphology
from skimage.morphology import disk, opening, closing, rectangle
from skimage.filters import threshold_otsu

from scipy import ndimage as ndi

image = io.imread("cells_binary_inv.png", as_gray=True)
binary = image > 0.5 

selem = disk(2) 

opened = opening(binary, selem)
closed = closing(binary, selem)

y1, y2 = 200, 350
x1, x2 = 250, 400


def draw_box(ax):
    ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                               edgecolor='red', fill=False, linewidth=2))

plt.figure(figsize=(8, 8))

# Original
plt.subplot(2, 2, 1)
plt.imshow(binary, cmap='gray')
plt.title("Original")
draw_box(plt.gca())
plt.axis('off')

# Opening
plt.subplot(2, 2, 2)
plt.imshow(opened, cmap='gray')
plt.title("Opening")
draw_box(plt.gca())
plt.axis('off')

# Original zoom
plt.subplot(2, 2, 3)
plt.imshow(binary[y1:y2, x1:x2], cmap='gray')
plt.title("Original (zoom)")
plt.axis('off')

# Opening zoom
plt.subplot(2, 2, 4)
plt.imshow(opened[y1:y2, x1:x2], cmap='gray')
plt.title("Opening (zoom)")
plt.axis('off')

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 8))

# Original
plt.subplot(2, 2, 1)
plt.imshow(binary, cmap='gray')
plt.title("Original")
draw_box(plt.gca())
plt.axis('off')

# Closing
plt.subplot(2, 2, 2)
plt.imshow(closed, cmap='gray')
plt.title("Closing")
draw_box(plt.gca())
plt.axis('off')

# Original zoom
plt.subplot(2, 2, 3)
plt.imshow(binary[y1:y2, x1:x2], cmap='gray')
plt.title("Original (zoom)")
plt.axis('off')

# Closing zoom
plt.subplot(2, 2, 4)
plt.imshow(closed[y1:y2, x1:x2], cmap='gray')
plt.title("Closing (zoom)")
plt.axis('off')

plt.tight_layout()
plt.show()

#4_1_3

# Label components (8-connectivity)
labels_open = measure.label(opened, connectivity=2)
labels_close = measure.label(closed, connectivity=2)

# Count components
num_open = labels_open.max()
num_close = labels_close.max()

print("Opening components:", num_open)
print("Closing components:", num_close)

# Visualize
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(labels_open, cmap='nipy_spectral')
plt.title(f"Opening labels ({num_open})")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(labels_close, cmap='nipy_spectral')
plt.title(f"Closing labels ({num_close})")
plt.axis("off")

plt.tight_layout()
plt.show()


# --------------------------------------------------
# Load image and binarize
# --------------------------------------------------
image = io.imread("money_bin.png", as_gray=True)
binary = image < 0.5   # coins are black

# --------------------------------------------------
# Morphological cleaning
# --------------------------------------------------
selem = morphology.disk(3)
clean = morphology.opening(binary, selem)
clean = morphology.closing(clean, selem)


labels = measure.label(clean, connectivity=2)
regions = measure.regionprops(labels)

print("Detected coins:", len(regions))


coins = []

for r in regions:
    mask = (labels == r.label)

    # Fill holes to detect if coin has hole
    filled = morphology.remove_small_holes(mask, area_threshold=500)
    hole = filled ^ mask
    has_hole = np.sum(hole) > 0

    coins.append({
        "label": r.label,
        "area": r.area,
        "has_hole": has_hole
    })


with_hole = [c for c in coins if c["has_hole"]]
without_hole = [c for c in coins if not c["has_hole"]]

coin_values = {}

without_hole = sorted(without_hole, key=lambda x: x["area"])

areas_wo = np.array([c["area"] for c in without_hole])
diffs_wo = np.diff(areas_wo)

split_wo = np.argmax(diffs_wo)

small_group = without_hole[:split_wo+1]
large_group = without_hole[split_wo+1:]

for c in small_group:
    coin_values[c["label"]] = 0.5

for c in large_group:
    coin_values[c["label"]] = 20

with_hole = sorted(with_hole, key=lambda x: x["area"])
areas = np.array([c["area"] for c in with_hole])

diffs = np.diff(areas)

# Find two largest jumps
split_idx = np.argsort(diffs)[-2:]
split_idx = np.sort(split_idx)

g1 = with_hole[:split_idx[0]+1]             
g2 = with_hole[split_idx[0]+1:split_idx[1]+1]  
g3 = with_hole[split_idx[1]+1:]            

for c in g1:
    coin_values[c["label"]] = 1

for c in g2:
    coin_values[c["label"]] = 2

for c in g3:
    coin_values[c["label"]] = 5


total = sum(coin_values.values())
print("Total money:", total, "kr")

colors = {
    0.5: [1, 0, 0],    # red
    1:   [0, 1, 0],    # green
    2:   [0, 0, 1],    # blue
    5:   [1, 1, 0],    # yellow
    20:  [1, 0, 1],    # magenta
}

colored = np.zeros((labels.shape[0], labels.shape[1], 3))

for c in coins:
    val = coin_values[c["label"]]
    colored[labels == c["label"]] = colors[val]

plt.figure(figsize=(8,6))
plt.imshow(colored)
plt.title("Coin classification result")
plt.axis("off")
plt.show()

for c in coins:
    print(f"Area: {c['area']:.0f}, Hole: {c['has_hole']}, Value: {coin_values[c['label']]}")
#4_3


# Load image
image = io.imread("matrikelnumre_nat.png")
gray = color.rgb2gray(image)

# Threshold so map is white foreground
thresh = threshold_otsu(gray)
binary = gray > thresh

# Keep only largest connected component = map
labels = measure.label(binary, connectivity=2)
regions = measure.regionprops(labels)
largest = max(regions, key=lambda r: r.area)
map_mask = labels == largest.label

# Close small holes / thin gaps in the map
closed = morphology.closing(map_mask, morphology.disk(6))

# Fill everything to get the plain map support
support = ndi.binary_fill_holes(closed)

# Residual black components that remain after closing
residual_black = support & (~closed)

# Remove tiny junk
residual_black = morphology.remove_small_objects(residual_black, min_size=20)

# Label residual components
comp_labels = measure.label(residual_black, connectivity=2)

# Blue dot label chosen manually from labeled plot
dot_label = 6
dot_mask = comp_labels == dot_label

# Final result: plain map with only blue-dot hole preserved
final = support.copy()
final[dot_mask] = False

#
plt.figure(figsize=(6, 5))
plt.imshow(map_mask, cmap="gray")
plt.title("Original binary map mask")
plt.axis("off")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(closed, cmap="gray")
plt.title("After closing with disk(6)")
plt.axis("off")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(final, cmap="gray")
plt.title("Final cleaned mask")
plt.axis("off")
plt.tight_layout()
plt.show()