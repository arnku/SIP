import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage

def power_spectrum(fft_img: np.ndarray):
    return np.abs(fft_img) ** 2

def plot_power_spectrum(fft_img: np.ndarray):
    power_spec = power_spectrum(fft_img)
    plt.imshow(np.log(1 + power_spec), cmap='gray')
    plt.axis('off')
    plt.show()

def shift_fft(img: np.ndarray):
    img_fft = scipy.fft.fft2(img)
    img_shift = scipy.fft.fftshift(img_fft)
    return img_shift

def inv_fft(fft: np.ndarray):
    inv_shift = scipy.fft.ifftshift(fft)
    inv_img = scipy.fft.ifft2(inv_shift)
    return np.real(inv_img)

def distance_mask(fft):
    M, N = fft.shape
    cy, cx = M // 2, N // 2
    circle_grid = np.meshgrid(np.arange(N), np.arange(M))[0]
    for x in range(N):
        for y in range(M):
            r = int(np.sqrt((x - cx) ** 2 + (y - cy) ** 2))
            circle_grid[y, x] = r
    return circle_grid

def angle_mask(fft, num_bins, freq_range):    
    M, N = fft.shape
    cy, cx = M // 2, N // 2
    angle_grid = np.full((M, N), np.nan) # Initialize with NaNs

    bins = np.linspace(0, 2 * np.pi, num_bins + 1)
    for x in range(N):
        for y in range(M):
            angle = np.arctan2(y - cy, x - cx)
            radius = np.sqrt((x - cx)**2 + (y - cy)**2)
            if radius < freq_range[0] or radius > freq_range[1]:
                continue 
            if angle < 0:
                angle += 2 * np.pi
            angle_bin = np.digitize(angle, bins) - 1
            angle_grid[y, x] = angle_bin
    return angle_grid

def avg_power_spectrum(fft: np.ndarray, grid: np.ndarray, spectrum_size: int):
    M, N = fft.shape
    power_spectrum_fft = power_spectrum(fft)

    spectrum = np.zeros(spectrum_size)
    for r in range(spectrum_size):
        mask = (grid == r)
        masked_fft = power_spectrum_fft * mask
        # Check if mask is empty to avoid division by zero
        if np.any(mask):
            spectrum[r] = np.average(masked_fft[mask])
        else:
             spectrum[r] = 0
    return spectrum


IMG = skimage.io.imread('bigben_cropped_gray.png')
shift_fft_img = shift_fft(IMG)
power_spectrum_img = power_spectrum(shift_fft_img)

plt.subplot(1, 2, 1)
plt.imshow(IMG, cmap='gray')
plt.axis('off')
plt.subplot(1, 2, 2)
plot_power_spectrum(shift_fft_img)

# Apply noise to the image
noise = np.random.normal(0, 25, IMG.shape) 
noisy_img = np.clip(IMG + noise, 0, 255).astype(np.uint8)

a0 = 10
v0 = 1
w0 = 0.5
x = np.arange(IMG.shape[1])
y = np.arange(IMG.shape[0])
X, Y = np.meshgrid(x, y)
noise2 = a0 * np.cos(v0 * X + w0 * Y)
noisy_img2 = np.clip(IMG + noise2, 0, 255).astype(np.uint8)

plt.subplot(1, 2, 1)
plt.imshow(noisy_img, cmap='gray')
plt.axis('off')
plt.title('Noisy Image')
plt.subplot(1, 2, 2)
plt.imshow(noisy_img2, cmap='gray')
plt.axis('off')
plt.title('Noisy with Cosine')
plt.show()

dist_mask = distance_mask(shift_fft_img)
spectrum = avg_power_spectrum(shift_fft_img, dist_mask, max(shift_fft_img.shape) // 2)
noisy_img_fft = shift_fft(noisy_img)
spectrum_noisy = avg_power_spectrum(noisy_img_fft, dist_mask, max(shift_fft_img.shape) // 2)
spectrum_noisy2 = avg_power_spectrum(shift_fft(noisy_img2), dist_mask, max(shift_fft_img.shape) // 2)

# plot the original spectrum for comparison
plt.loglog(spectrum, label='Original Image')
plt.loglog(spectrum_noisy, label='Noisy Image')
plt.loglog(spectrum_noisy2, label='Noisy with Cosine')
plt.xlabel('Radius (log scale)')
plt.ylabel('Average Power (log scale)')
plt.title('Power Spectrum Comparison')
plt.legend()
plt.show()

angle_grid = angle_mask(shift_fft_img, num_bins=10, freq_range=(10, 100))
# Visualize the angle grid
plt.imshow(angle_grid, cmap='hsv') # hsv is good for cyclical data like angles
plt.colorbar(label='Angle (radians)')
plt.title('Angle Map')
plt.axis('off')
plt.show()

spectrum = avg_power_spectrum(shift_fft_img, angle_grid, 10)
spectrum_noisy = avg_power_spectrum(noisy_img_fft, angle_grid, 10)
spectrum_noisy2 = avg_power_spectrum(shift_fft(noisy_img2), angle_grid, 10)
plt.plot(spectrum, label='Original')
plt.plot(spectrum_noisy, label='Noisy')
plt.plot(spectrum_noisy2, label='Noisy with Cosine')
plt.xlabel('Angle Bin')
plt.ylabel('Average Power')
plt.title('Average Power Spectrum by Angle')
plt.legend()
plt.show()