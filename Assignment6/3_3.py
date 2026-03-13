import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d
import sklearn
import scipy
import skimage

IMG = skimage.io.imread("sunandsea.jpg", as_gray=True)

def njet(image, sigma=7.5):
    L = lambda n, m: scipy.ndimage.gaussian_filter(image, sigma, order=(m, n))

    L_x,   L_y   = L(1,0), L(0,1)
    L_xx,  L_xy,  L_yy  = L(2,0), L(1,1), L(0,2)
    L_xxx, L_xxy, L_xyy, L_yyy = L(3,0), L(2,1), L(1,2), L(0,3)

    return np.array([L(0,0), L_x, L_y, L_xx, L_xy, L_yy, L_xxx, L_xxy, L_xyy, L_yyy])

jets = njet(IMG)
njet_features = jets.reshape(len(jets), -1).T

patch_size = (8, 8)
n_components = 6
patches = extract_patches_2d(IMG, patch_size).reshape(-1, patch_size[0]*patch_size[1])
pca = sklearn.decomposition.PCA(n_components).fit(patches)
filters = pca.components_.reshape(n_components, patch_size[0], patch_size[1])
responses = [scipy.ndimage.convolve(IMG, f, mode="reflect") for f in filters]
pca_features = np.stack(responses, axis=-1).reshape(-1, n_components)

n_clusters = 3
njet_labels = sklearn.cluster.KMeans(n_clusters=n_clusters).fit_predict(njet_features).reshape(IMG.shape)
pca_labels  = sklearn.cluster.KMeans(n_clusters=n_clusters).fit_predict(pca_features).reshape(IMG.shape)

plt.subplot(1, 3, 1)
plt.imshow(IMG, cmap="gray");        
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(njet_labels)
plt.title("N-Jet Segmentation")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(pca_labels)
plt.title("PCA Segmentation")
plt.axis("off")

plt.show()