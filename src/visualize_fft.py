import numpy as np
import matplotlib.pyplot as plt
from load_fashion import get_fashion_mnist

FFT_SIZE = 12  # low-frequency crop size

def compute_fft_amp(img_tensor, size=FFT_SIZE):
    img = img_tensor.squeeze().numpy().astype(np.float32) / 255.0
    F = np.fft.fft2(img)
    F_shift = np.fft.fftshift(F)

    # crop low-frequency
    h, w = F_shift.shape
    c_h, c_w = h // 2, w // 2
    lowF = F_shift[c_h-size:c_h+size, c_w-size:c_w+size]

    amp = np.abs(lowF)
    amp = np.log1p(amp)   # log compression

    return amp

# -----------------------------
# Load data
# -----------------------------
train_loader, _ = get_fashion_mnist(batch_size=1)
print("Loaded Fashion-MNIST!")

# 10 classes: 0~9
class_sum = np.zeros((10, FFT_SIZE * 2, FFT_SIZE * 2))
class_count = np.zeros(10)

# -----------------------------
# Accumulate FFT amplitude
# -----------------------------
for img, label in train_loader:
    cls = label.item()
    amp = compute_fft_amp(img)
    class_sum[cls] += amp
    class_count[cls] += 1

# -----------------------------
# Average FFT spectrum per class
# -----------------------------
class_avg = class_sum / class_count[:, None, None]

# -----------------------------
# Plot all 10 classes
# -----------------------------
titles = [
    "0: T-shirt/top",
    "1: Trouser",
    "2: Pullover",
    "3: Dress",
    "4: Coat",
    "5: Sandal",
    "6: Shirt",
    "7: Sneaker",
    "8: Bag",
    "9: Ankle boot"
]

plt.figure(figsize=(12, 6))

for c in range(10):
    plt.subplot(2, 5, c + 1)
    plt.imshow(class_avg[c], cmap="inferno")
    plt.title(titles[c], fontsize=8)
    plt.axis("off")

plt.suptitle("Average FFT Amplitude (Low-Frequency 24×24)", fontsize=14)
plt.tight_layout()
plt.show()

print("FFT Amplitude Visualization complete.")
