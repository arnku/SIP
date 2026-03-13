import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.ndimage import gaussian_filter
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import PCA
from scipy.signal import convolve2d
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load image
img = io.imread("Assignment6/sunandsea.jpg", as_gray=True)

# Scales
sigmas = [1,2,4]

# Derivative orders up to third order
orders = [
(0,0),
(1,0),(0,1),
(2,0),(0,2),(1,1),
(3,0),(0,3),(2,1),(1,2)
]

responses = []

for s in sigmas:
    for o in orders:

        r = gaussian_filter(img, sigma=s, order=o)

        # scale normalization
        r = (s ** (o[0]+o[1])) * r

        responses.append((s,o,r))

plt.figure(figsize=(12,8))

for i,(s,o,r) in enumerate(responses):
    r = (r-r.min())/(r.max()-r.min())

    plt.subplot(len(sigmas),len(orders),i+1)
    plt.imshow(r,cmap="gray")
    plt.title(f"s={s},o={o}",fontsize=7)
    plt.axis("off")

plt.tight_layout()
plt.show()

#3_2
# load image
img = io.imread("Assignment6/sunandsea.jpg", as_gray=True)

# extract random patches
patches = extract_patches_2d(img, (8,8), max_patches=5000)

# reshape for PCA
X = patches.reshape(len(patches), -1)

# PCA
pca = PCA(n_components=10)
pca.fit(X)

filters = pca.components_.reshape(-1,8,8)

# visualize learned filters
plt.figure(figsize=(10,3))
for i,f in enumerate(filters):
    plt.subplot(2,5,i+1)
    plt.imshow(f, cmap='gray')
    plt.axis("off")
plt.suptitle("PCA learned filters")
plt.show()

# filter responses
plt.figure(figsize=(10,3))
for i,f in enumerate(filters):
    r = convolve2d(img, f, mode="same")
    r = (r-r.min())/(r.max()-r.min())
    plt.subplot(2,5,i+1)
    plt.imshow(r, cmap='gray')
    plt.axis("off")
plt.suptitle("Filter responses")
plt.show()


#task 3_3
img = io.imread("Assignment6/sunandsea.jpg", as_gray=True)


sigmas = [1,2,4]

orders = [
    (0,0),
    (1,0),(0,1),
    (2,0),(0,2),(1,1),
    (3,0),(0,3),(2,1),(1,2)
]

responses = []

for s in sigmas:
    for o in orders:

        r = gaussian_filter(img, sigma=s, order=o)

        r = (s ** (o[0]+o[1])) * r

        responses.append(r)


H, W = responses[0].shape

features = np.stack(responses, axis=-1) 
features = features.reshape(-1, features.shape[-1])  


features = StandardScaler().fit_transform(features)


kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
labels = kmeans.fit_predict(features)

segmentation = labels.reshape(H, W)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.title("Original image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(segmentation, cmap="viridis")
plt.title("K-Means segmentation (K=3)")
plt.axis("off")

plt.tight_layout()
plt.show()