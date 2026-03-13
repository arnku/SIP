import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn
from sklearn.feature_extraction.image import extract_patches_2d
import skimage

IMG = skimage.io.imread("sunandsea.jpg", as_gray=True)

patch_size = (8, 8)
n_components = 6

patches = extract_patches_2d(IMG, patch_size).reshape(-1, patch_size[0]*patch_size[1])

pca = sklearn.decomposition.PCA(n_components).fit(patches)
filters = pca.components_.reshape(n_components, patch_size[0], patch_size[1])

responses = [scipy.ndimage.convolve(IMG, f, mode="reflect") for f in filters]

fig, axes = plt.subplots(2, n_components)
for i in range(n_components):
    axes[0, i].imshow(filters[i], cmap="gray")
    axes[0, i].set_title(f"PC{i+1}")
    axes[1, i].imshow(responses[i], cmap="gray")
    axes[1, i].axis("off")
    axes[0, i].axis("off")
plt.show()