import numpy as np
import matplotlib.pyplot as plt
import scipy
import skimage

def G(x, y, sigma):
    return (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

size = 100
x = np.linspace(-(size//2), size//2, size)
y = np.linspace(-(size//2), size//2, size)
X, Y = np.meshgrid(x, y)

sigma = 1.0
B = G(X, Y, sigma)

plt.figure(figsize=(12, 5))
plt.subplot(1, 4, 1)
plt.imshow(B, cmap='gray')
plt.title(f'B(x,y), σ={sigma}')
plt.axis('off')

tau = 2.0
G_tau = G(X, Y, tau)
I_convolved = scipy.signal.convolve2d(B, G_tau, mode='same')

plt.subplot(1, 4, 2)
plt.imshow(I_convolved, cmap='gray')
plt.title(f'B * G(τ={tau})')
plt.axis('off')

sigma_combined = np.sqrt(sigma**2 + tau**2)
I_direct = G(X, Y, sigma_combined)

plt.subplot(1, 4, 3)
plt.imshow(I_direct, cmap='gray')
plt.title(f'G(x,y, √(σ²+τ²)={sigma_combined:.2f})')
plt.axis('off')

diff = I_convolved - I_direct

plt.subplot(1, 4, 4)
plt.imshow(diff, cmap='gray')
plt.title(f'Difference (max={np.max(np.abs(diff)):.2e})')
plt.axis('off')

plt.show()

sigma = 1.0
tau = np.linspace(0.01, 5, 1000)

H = -(tau**2) / (np.pi * (sigma**2 + tau**2)**2)

plt.plot(tau, H)
plt.axvline(sigma, color='r', linestyle='--', label=f'τ = σ = {sigma}')
plt.xlabel('τ')
plt.ylabel('H(0,0,τ)')
plt.title('Scale-normalized Laplacian at (0,0) as a function of τ')
plt.legend()
plt.show()

