import numpy as np
import matplotlib.pyplot as plt
import scipy
import skimage

SIZE = 101 # image size (odd)
IMG = np.zeros((SIZE, SIZE))
square_size = 20

center = SIZE // 2
IMG[center-square_size:center+square_size, center-square_size:center+square_size] = 1

plt.imshow(IMG, cmap='gray')
plt.axis('off')
plt.show()


def translate_image(image: np.ndarray, dx: int, dy: int, boundary: str) -> np.ndarray:

    # Odd sized kernel big enough to contain the shift
    kernel_height = 2 * abs(dy) + 1
    kernel_width = 2 * abs(dx) + 1
    kernel = np.zeros((kernel_height, kernel_width), dtype=np.float32)
    
    kernel[abs(dy) - dy, abs(dx) - dx] = 1.0
    
    translated_image = scipy.signal.convolve2d(image, kernel, mode='same', boundary=boundary)
    return translated_image

translated_img = translate_image(IMG, 10, 5, 'fill')
plt.imshow(translated_img, cmap='gray')
plt.axis('off')
plt.show()

def translate_homogeneous(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    T_inv = np.array([[1, 0, -dx],
                      [0, 1, -dy],
                      [0, 0,   1]])

    h, w = image.shape
    output = np.zeros_like(image)
    for y in range(h):
        for x in range(w):
            src = T_inv @ np.array([x, y, 1])
            src_x, src_y = round(src[0]), round(src[1])

            # Effectively zero padding
            if 0 <= src_x < w and 0 <= src_y < h:
                output[y, x] = image[src_y, src_x]

    return output

translated_homogeneous = translate_homogeneous(IMG, 0.6, 1.2)
plt.imshow(translated_homogeneous, cmap='gray')
plt.axis('off')
plt.show()


def translate_fourier(image, dx, dy):
    h, w = image.shape
    F = np.fft.fft2(image)
    u = np.fft.fftfreq(w)
    v = np.fft.fftfreq(h)
    U, V = np.meshgrid(u, v)

    phase = np.exp(-2j * np.pi * (U * dx + V * dy))
    F_shifted = F * phase

    image_shifted = np.fft.ifft2(F_shifted)
    return np.real(image_shifted)

translated_fourier = translate_fourier(IMG, 0.6, 1.2)
plt.imshow(translated_fourier, cmap='gray')
plt.axis('off')
plt.show()

IMG = skimage.io.imread('image.png', as_gray=True)
translated = translate_fourier(IMG, 50.5, 10.5)
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(IMG, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(translated, cmap='gray')
plt.title("Translated (50.5, 10.5)")
plt.axis("off")

plt.show()