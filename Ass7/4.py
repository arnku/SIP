import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, measure, morphology
from skimage.morphology import disk, opening, closing
from skimage.morphology import rectangle

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


#4_2
import matplotlib.pyplot as plt
from skimage import io, measure, morphology

# --------------------------------------------------
# 1. Load + binarize
# --------------------------------------------------
image = io.imread("money_bin.png", as_gray=True)
binary = image < 0.5   # coins are black

# --------------------------------------------------
# 2. Morphological cleaning
# --------------------------------------------------
selem = morphology.disk(3)
clean = morphology.opening(binary, selem)
clean = morphology.closing(clean, selem)

# --------------------------------------------------
# 3. Label coins (8-connectivity)
# --------------------------------------------------
labels = measure.label(clean, connectivity=2)
regions = measure.regionprops(labels)

print("Detected coins:", len(regions))

# --------------------------------------------------
# 4. Extract features (area + hole)
# --------------------------------------------------
coins = []

for r in regions:
    mask = (labels == r.label)

    # Fill holes
    filled = morphology.remove_small_holes(mask, area_threshold=500)

    # Hole = difference
    hole = filled ^ mask
    has_hole = np.sum(hole) > 0

    coins.append({
        "label": r.label,
        "area": r.area,
        "has_hole": has_hole
    })

# --------------------------------------------------
# 5. Split into hole / no-hole groups
# --------------------------------------------------
with_hole = [c for c in coins if c["has_hole"]]
without_hole = [c for c in coins if not c["has_hole"]]

# --------------------------------------------------
# 6. NO-HOLE coins → 50 øre + 20 kr
# --------------------------------------------------
without_hole = sorted(without_hole, key=lambda x: x["area"])

coin_values = {}

# smallest = 50 øre
coin_values[without_hole[0]["label"]] = 0.5

# rest = 20 kr
for c in without_hole[1:]:
    coin_values[c["label"]] = 20

# --------------------------------------------------
# 7. HOLE coins → cluster into 1, 2, 5 kr
# --------------------------------------------------
with_hole = sorted(with_hole, key=lambda x: x["area"])
areas = np.array([c["area"] for c in with_hole])

# find biggest gaps → split into 3 groups
diffs = np.diff(areas)
split_idx = np.argsort(diffs)[-2:]
split_idx = np.sort(split_idx)

g1 = with_hole[:split_idx[0]+1]             # smallest → 1 kr
g2 = with_hole[split_idx[0]+1:split_idx[1]+1]  # middle → 2 kr
g3 = with_hole[split_idx[1]+1:]            # largest → 5 kr

for c in g1:
    coin_values[c["label"]] = 1

for c in g2:
    coin_values[c["label"]] = 2

for c in g3:
    coin_values[c["label"]] = 5

# --------------------------------------------------
# 8. Compute total
# --------------------------------------------------
total = sum(coin_values.values())
print("Total money:", total, "kr")

# --------------------------------------------------
# 9. Visualization (same value = same color)
# --------------------------------------------------
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
plt.title(f"Coins classified (area + holes), total = {total} kr")
plt.axis("off")
plt.show()

for c in coins:
    print(f"Area: {c['area']:.0f}, Hole: {c['has_hole']}, Value: {coin_values[c['label']]}")


#4_3
import matplotlib.pyplot as plt
from skimage import io, color, measure, morphology
from skimage.filters import threshold_otsu

# Load image
image = io.imread("matrikelnumre_nat.png")
gray = color.rgb2gray(image)

# Binary
thresh = threshold_otsu(gray)
binary = gray < thresh

# Fill holes → makes map solid
filled = morphology.remove_small_holes(binary, area_threshold=5000)

# Label connected components (8-connectivity)
labels = measure.label(filled, connectivity=2)

# Get regions
regions = measure.regionprops(labels)

# Sort by area (largest first)
regions = sorted(regions, key=lambda r: r.area, reverse=True)

# Keep:
# - largest region → map
# - second round region → blue dot
clean = (labels == regions[0].label)

# Find circular region (dot)
for r in regions[1:]:
    # circularity check: area vs bounding box
    minr, minc, maxr, maxc = r.bbox
    h = maxr - minr
    w = maxc - minc

    if abs(h - w) < 10:  # roughly square → circle
        clean = clean | (labels == r.label)
        break

# Visualization
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original binary")
plt.imshow(binary, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Final cleaned (map + dot only)")
plt.imshow(clean, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()