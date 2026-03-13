import numpy as np
import scipy
import matplotlib.pyplot as plt
import skimage

IMG = skimage.io.imread('trui.png')

##### linear shift invariant degradation #####

def lsi_degradation(image: np.ndarray, kernel: np.ndarray, noise: np.ndarray) -> np.ndarray:
    blurred = scipy.signal.convolve2d(image, kernel, mode='same', boundary='symm')
    degraded = blurred + noise
    return degraded

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

def box_kernel(size: int) -> np.ndarray:
    kernel = np.ones((size, size), dtype=float)
    kernel /= kernel.sum()
    return kernel

def motion_kernel(length: int) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=float)
    kernel[length // 2, :] = 1.0
    kernel /= kernel.sum()
    return kernel

noise1 = np.random.normal(0, 2, IMG.shape)
noise2 = np.random.normal(0, 5, IMG.shape)
noise3 = np.random.normal(0, 10, IMG.shape)

degraded_image_gauss1 = lsi_degradation(IMG, gaussian_kernel(5, 1.0), noise1)
degraded_image_gauss2 = lsi_degradation(IMG, gaussian_kernel(5, 1.0), noise2)
degraded_image_gauss3 = lsi_degradation(IMG, gaussian_kernel(5, 1.0), noise3)

plt.figure(figsize=(12, 5))
plt.title('LSI Degradation with Gaussian Kernel')
plt.axis('off')

plt.subplot(1, 3, 1)
plt.title('noise with std=2')
plt.imshow(degraded_image_gauss1.astype(np.uint8), cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('noise with std=5')
plt.imshow(degraded_image_gauss2.astype(np.uint8), cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('noise with std=10')
plt.imshow(degraded_image_gauss3.astype(np.uint8), cmap='gray')
plt.axis('off')

plt.show()


plt.figure(figsize=(12, 5))
plt.title('LSI Degradation with Box Kernel')
plt.axis('off')

degraded_image_box1 = lsi_degradation(IMG, box_kernel(5), noise1)
degraded_image_box2 = lsi_degradation(IMG, box_kernel(5), noise2)
degraded_image_box3 = lsi_degradation(IMG, box_kernel(5), noise3)

plt.subplot(1, 3, 1)
plt.title('noise with std=2')
plt.imshow(degraded_image_box1.astype(np.uint8), cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('noise with std=5')
plt.imshow(degraded_image_box2.astype(np.uint8), cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('noise with std=10')
plt.imshow(degraded_image_box3.astype(np.uint8), cmap='gray')
plt.axis('off')

plt.show()

plt.figure(figsize=(12, 5))
plt.title('LSI Degradation with Motion Kernel')
plt.axis('off')

degraded_image_motion1 = lsi_degradation(IMG, motion_kernel(12), noise1)
degraded_image_motion2 = lsi_degradation(IMG, motion_kernel(12), noise2)
degraded_image_motion3 = lsi_degradation(IMG, motion_kernel(12), noise3)

plt.subplot(1, 3, 1)
plt.title('noise with std=2')
plt.imshow(degraded_image_motion1.astype(np.uint8), cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('noise with std=5')
plt.imshow(degraded_image_motion2.astype(np.uint8), cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('noise with std=10')
plt.imshow(degraded_image_motion3.astype(np.uint8), cmap='gray')
plt.axis('off')

plt.show()



##### direct inverse filter #####

noise1 = np.random.normal(0, 0.1, IMG.shape)
noise2 = np.random.normal(0, 1, IMG.shape)
noise3 = np.random.normal(0, 10, IMG.shape)

degraded_image_gauss0 = lsi_degradation(IMG, gaussian_kernel(5, 1.0), np.zeros_like(IMG))
degraded_image_gauss1 = lsi_degradation(IMG, gaussian_kernel(5, 1.0), noise1)
degraded_image_gauss2 = lsi_degradation(IMG, gaussian_kernel(5, 1.0), noise2)
degraded_image_gauss3 = lsi_degradation(IMG, gaussian_kernel(5, 1.0), noise3)

def direct_inverse_filter(degraded: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kernel_ft = scipy.fft.fft2(kernel, s=degraded.shape)
    degraded_ft = scipy.fft.fft2(degraded)
    
    restored_ft = degraded_ft / kernel_ft
    restored = scipy.fft.ifft2(restored_ft)
    
    return restored.real


restored_gauss0 = direct_inverse_filter(degraded_image_gauss0, gaussian_kernel(5, 1.0))
restored_gauss1 = direct_inverse_filter(degraded_image_gauss1, gaussian_kernel(5, 1.0))
restored_gauss2 = direct_inverse_filter(degraded_image_gauss2, gaussian_kernel(5, 1.0))
restored_gauss3 = direct_inverse_filter(degraded_image_gauss3, gaussian_kernel(5, 1.0))

plt.figure(figsize=(12, 5))
plt.title('Restored Images with Direct Inverse Filter (Gaussian Kernel)')
plt.axis('off')

plt.subplot(1, 4, 1)
plt.title('no noise')
plt.imshow(np.clip(restored_gauss0, 0, 255).astype(np.uint8), cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('noise with std=0.1')
plt.imshow(np.clip(restored_gauss1, 0, 255).astype(np.uint8), cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('noise with std=1')
plt.imshow(np.clip(restored_gauss2, 0, 255).astype(np.uint8), cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('noise with std=10')
plt.imshow(np.clip(restored_gauss3, 0, 255).astype(np.uint8), cmap='gray')
plt.axis('off')

plt.show()



##### Wiener filter #####

def wiener_filter(degraded: np.ndarray, kernel: np.ndarray, K: float = 0.01) -> np.ndarray:
    kernel_ft = scipy.fft.fft2(kernel, s=degraded.shape)
    degraded_ft = scipy.fft.fft2(degraded)

    inverse_ft = (1 / kernel_ft) * (np.abs(kernel_ft) ** 2 / (np.abs(kernel_ft) ** 2 + K))
    restored_ft = degraded_ft * inverse_ft
    restored = scipy.fft.ifft2(restored_ft)

    return restored.real

noise1 = np.random.normal(0, 0.1, IMG.shape)
noise2 = np.random.normal(0, 1, IMG.shape)
noise3 = np.random.normal(0, 50, IMG.shape)

degraded_image_gauss0 = lsi_degradation(IMG, gaussian_kernel(5, 1.0), np.zeros_like(IMG))
degraded_image_gauss1 = lsi_degradation(IMG, gaussian_kernel(5, 1.0), noise1)
degraded_image_gauss2 = lsi_degradation(IMG, gaussian_kernel(5, 1.0), noise2)
degraded_image_gauss3 = lsi_degradation(IMG, gaussian_kernel(5, 1.0), noise3)

restored_gauss_wiener0 = wiener_filter(degraded_image_gauss0, gaussian_kernel(5, 1.0), 0.01)
restored_gauss_wiener1 = wiener_filter(degraded_image_gauss1, gaussian_kernel(5, 1.0), 0.01)
restored_gauss_wiener2 = wiener_filter(degraded_image_gauss2, gaussian_kernel(5, 1.0), 0.01)
restored_gauss_wiener3 = wiener_filter(degraded_image_gauss3, gaussian_kernel(5, 1.0), 0.01)

plt.figure(figsize=(12, 5))
plt.title('Restored Images with Wiener Filter (Gaussian Kernel) with K=0.01')
plt.axis('off')

plt.subplot(1, 4, 1)
plt.title('no noise')
plt.imshow(np.clip(restored_gauss_wiener0, 0, 255).astype(np.uint8), cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('noise with std=0.1')
plt.imshow(np.clip(restored_gauss_wiener1, 0, 255).astype(np.uint8), cmap='gray')
plt.axis('off')     

plt.subplot(1, 4, 3)
plt.title('noise with std=1')
plt.imshow(np.clip(restored_gauss_wiener2, 0, 255).astype(np.uint8), cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('noise with std=50')
plt.imshow(np.clip(restored_gauss_wiener3, 0, 255).astype(np.uint8), cmap='gray')
plt.axis('off')

plt.show()

restored_gauss_wiener0 = wiener_filter(degraded_image_gauss0, gaussian_kernel(5, 1.0), 0.1)
restored_gauss_wiener1 = wiener_filter(degraded_image_gauss1, gaussian_kernel(5, 1.0), 0.1)
restored_gauss_wiener2 = wiener_filter(degraded_image_gauss2, gaussian_kernel(5, 1.0), 0.1)
restored_gauss_wiener3 = wiener_filter(degraded_image_gauss3, gaussian_kernel(5, 1.0), 0.1)

plt.figure(figsize=(12, 5))
plt.title('Restored Images with Wiener Filter (Gaussian Kernel) with K=0.1')
plt.axis('off')

plt.subplot(1, 4, 1)
plt.title('no noise')
plt.imshow(np.clip(restored_gauss_wiener0, 0, 255).astype(np.uint8), cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('noise with std=0.1')
plt.imshow(np.clip(restored_gauss_wiener1, 0, 255).astype(np.uint8), cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('noise with std=1')
plt.imshow(np.clip(restored_gauss_wiener2, 0, 255).astype(np.uint8), cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('noise with std=50')
plt.imshow(np.clip(restored_gauss_wiener3, 0, 255).astype(np.uint8), cmap='gray')
plt.axis('off') 

plt.show()

restored_gauss_wiener0 = wiener_filter(degraded_image_gauss0, gaussian_kernel(5, 1.0), 1)
restored_gauss_wiener1 = wiener_filter(degraded_image_gauss1, gaussian_kernel(5, 1.0), 1)
restored_gauss_wiener2 = wiener_filter(degraded_image_gauss2, gaussian_kernel(5, 1.0), 1)
restored_gauss_wiener3 = wiener_filter(degraded_image_gauss3, gaussian_kernel(5, 1.0), 1)

plt.figure(figsize=(12, 5))
plt.title('Restored Images with Wiener Filter (Gaussian Kernel) with K=1')
plt.axis('off')

plt.subplot(1, 4, 1)
plt.title('no noise')
plt.imshow(np.clip(restored_gauss_wiener0, 0, 255).astype(np.uint8), cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('noise with std=0.1')
plt.imshow(np.clip(restored_gauss_wiener1, 0, 255).astype(np.uint8), cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('noise with std=1')
plt.imshow(np.clip(restored_gauss_wiener2, 0, 255).astype(np.uint8), cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('noise with std=50')
plt.imshow(np.clip(restored_gauss_wiener3, 0, 255).astype(np.uint8), cmap='gray')
plt.axis('off')

plt.show()