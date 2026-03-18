import skimage
import numpy as np
import matplotlib.pyplot as plt
import skimage


IMG1_COL = skimage.io.imread('matrikelnumre_art.png')
IMG2_COL = skimage.io.imread('matrikelnumre_nat.png')
IMG1 = skimage.color.rgb2gray(IMG1_COL)
IMG2 = skimage.color.rgb2gray(IMG2_COL)


threshold1 = skimage.filters.threshold_otsu(IMG1)
threshold2 = skimage.filters.threshold_otsu(IMG2)

IMG1_THRES = np.where(IMG1 > threshold1, 1, 0)
IMG2_THRES = np.where(IMG2 > threshold2, 1, 0)

plt.subplot(2, 3, 1)
plt.imshow(IMG1_COL, cmap='gray')
plt.title('Original Image 1')
plt.axis('off')
plt.subplot(2, 3, 2)
plt.imshow(IMG1, cmap='gray')
plt.title('Grayscale Image 1')
plt.axis('off')
plt.subplot(2, 3, 3)
plt.imshow(IMG1_THRES, cmap='gray')
plt.title('Thresholded Image 1')
plt.axis('off')
plt.subplot(2, 3, 4)
plt.imshow(IMG2_COL, cmap='gray')
plt.title('Original Image 2')
plt.axis('off')
plt.subplot(2, 3, 5)
plt.imshow(IMG2, cmap='gray')
plt.title('Grayscale Image 2')
plt.axis('off')
plt.subplot(2, 3, 6)
plt.imshow(IMG2_THRES, cmap='gray')
plt.title('Thresholded Image 2')
plt.axis('off')
plt.show()

plt.subplot(1, 2, 1)
plt.hist(IMG1.ravel(), bins=256, color='blue', alpha=0.5)
plt.axvline(threshold1, color='red', linestyle='--', label=f'Threshold = {threshold1:.2f}')
plt.title('Histogram of Grayscale Image 1')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.subplot(1, 2, 2)
plt.hist(IMG2.ravel(), bins=256, color='green', alpha=0.5)
plt.axvline(threshold2, color='red', linestyle='--', label=f'Threshold = {threshold2:.2f}')
plt.title('Histogram of Grayscale Image 2')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.show()

edges = skimage.feature.canny(IMG1, sigma=1.0)

plt.imshow(edges, cmap='gray')
plt.title('Canny Edges')
plt.axis('off')
plt.show()

