import numpy as np
import matplotlib.pyplot as plt
import scipy
import skimage

def njet(image, x, y, sigma=7.5):
    L = lambda n, m: scipy.ndimage.gaussian_filter(image, sigma, order=(m, n))

    L_x,   L_y   = L(1,0), L(0,1)
    L_xx,  L_xy,  L_yy  = L(2,0), L(1,1), L(0,2)
    L_xxx, L_xxy, L_xyy, L_yyy = L(3,0), L(2,1), L(1,2), L(0,3)

    return (x*L_x + y*L_y
            + 1/2 *(x**2*L_xx  + 2*x*y*L_xy              + y**2*L_yy)
            + 1/6 *(x**3*L_xxx + 3*x**2*y*L_xxy + 3*x*y**2*L_xyy + y**3*L_yyy)), {
            "L":     L(0,0),
            "L_x":   L_x,   "L_y":   L_y,
            "L_xx":  L_xx,  "L_xy":  L_xy,  "L_yy":  L_yy,
            "L_xxx": L_xxx, "L_xxy": L_xxy, "L_xyy": L_xyy, "L_yyy": L_yyy,
    }

impulse = np.zeros((100, 100))
impulse[50, 50] = 1

_, jets = njet(impulse, x=1, y=1)  # x,y don't matter, we only use the jets

components_by_order = [
    ["L"],
    ["L_x",   "L_y"],
    ["L_xx",  "L_xy",  "L_yy"],
    ["L_xxx", "L_xxy", "L_xyy", "L_yyy"],
]

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, row in enumerate(components_by_order):
    for j in range(4):
        ax = axes[i, j]
        if j < len(row):
            name = row[j]
            ax.imshow(jets[name], cmap="grey")
            ax.set_title(name)
        ax.axis("off")

plt.suptitle("N-Jet Filter Bank Components (orders 0–3)")
plt.show()

IMG = skimage.io.imread("sunandsea.jpg", as_gray=True)

Lhat, jets = njet(IMG, x=1, y=1)  # x,y don't matter, we only use the jets

fig, axes = plt.subplots(4, 4, figsize=(14, 14))
for i, row in enumerate(components_by_order):
    for j in range(4):
        ax = axes[i, j]
        if j < len(row):
            name = row[j]
            ax.imshow(jets[name], cmap="grey")
            ax.set_title(name, fontsize=10)
        ax.axis("off")

plt.suptitle("N-Jet Filter Bank applied to sunandsea.jpg", fontsize=13)
plt.tight_layout()
plt.show()

plt.imshow(Lhat, cmap="gray")
plt.title("Lhat (N-Jet approximation of original image)", fontsize=10)
plt.axis("off")
plt.show()