import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from PIL import Image
# image size (odd)
N = 101

# create zero-valued image
I = np.zeros((N, N))

# square size
s = 20

# center index
c = N // 2

# create centered white square
I[c-s:c+s, c-s:c+s] = 1

# show image
plt.imshow(I, cmap='gray')
plt.axis('off')
plt.title("Centered white square")
plt.show()
plt.imsave("white_square.png", I, cmap='gray')

def translate_integer(I, tx, ty, boundary='fill'):
    """
    Translate image I by (tx, ty) pixels using a filter mask.

    tx : shift in x direction (right positive)
    ty : shift in y direction (down positive)
    boundary : boundary condition
    """

    # kernel size must contain the shift
    kernel = np.zeros((2*abs(ty)+1, 2*abs(tx)+1))

    # place impulse
    kernel[abs(ty)-ty, abs(tx)-tx] = 1

    # apply convolution
    shifted = convolve2d(I, kernel, mode='same', boundary=boundary)

    return shifted

translated = translate_integer(I, 10, 5)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(I, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(translated, cmap='gray')
plt.title("Translated (10,5)")
plt.axis("off")

plt.show()

#Task 1_6

def translate_fractional(I, tx, ty):

    h, w = I.shape
    out = np.zeros_like(I)

    for y in range(h):
        for x in range(w):

            xs = x - tx
            ys = y - ty

            xs_nn = int(round(xs))
            ys_nn = int(round(ys))

            if 0 <= xs_nn < w and 0 <= ys_nn < h:
                out[y, x] = I[ys_nn, xs_nn]

    return out

translated = translate_fractional(I, 0.6, 1.2)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(I, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(translated, cmap='gray')
plt.title("Translated (0.6,1.2)")
plt.axis("off")

plt.show()

#1_7
def translate_fourier(I, tx, ty):

    h, w = I.shape

    F = np.fft.fft2(I)

    u = np.fft.fftfreq(w)
    v = np.fft.fftfreq(h)
    U, V = np.meshgrid(u, v)

    phase = np.exp(-2j * np.pi * (U * tx + V * ty))
    F_shifted = F * phase

    out = np.fft.ifft2(F_shifted)

    return np.real(out)

translated_fourier = translate_fourier(I, 0.6, 1.2)

translated_fourier = translate_fourier(I, 0.6, 1.2)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(I, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(translated_fourier, cmap='gray')
plt.title("Fourier translated (0.6,1.2)")
plt.axis("off")

plt.show()

#Task 1_8 
# load image
img = Image.open("Assignment6/Big_Chungus.webp").convert("L")
I2 = np.array(img) / 255.0                    

# apply Fourier translation
translated_chungus = translate_fourier(I2, 50.5, 10.3)
# show results
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(I2, cmap='gray')
plt.title("Original image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(translated_chungus, cmap='gray')
plt.title("Fourier translated (50.5, 10.3)")
plt.axis("off")

plt.show()