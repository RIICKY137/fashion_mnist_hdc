import numpy as np
import matplotlib.pyplot as plt
from load_fashion import get_fashion_mnist
from hdc import bind

# ---------------- Load vectors ----------------
pixel_hvs = np.load("pixel_hvs.npy")
value_hvs = np.load("value_hvs.npy")
class_hv  = np.load("class_hv.npy")

print("pixel_hvs shape:", pixel_hvs.shape)  # (784, DIM)
print("value_hvs shape:", value_hvs.shape)  # (256, DIM)
print("class_hv shape:", class_hv.shape)    # (10, DIM)

# ---------------- Check 1s ratio in class hypervectors ----------------
for c in range(10):
    ratio = np.sum(class_hv[c]) / class_hv.shape[1]
    print(f"Class {c} HV 1 ratio: {ratio:.3f}")

# ---------------- Visualize first 1000 dimensions of each class ----------------
plt.figure(figsize=(15, 3))
for c in range(10):
    plt.subplot(1, 10, c+1)
    plt.imshow(class_hv[c][:1000].reshape(20, 50), cmap='gray')
    plt.axis('off')
    plt.title(f"Class {c}")
plt.suptitle("Class hypervectors (first 1000 dims)")
plt.show()

# ---------------- Visualize a sample image encoding ----------------
train_loader, _ = get_fashion_mnist(batch_size=1)
img, label = next(iter(train_loader))
img_flat = (img.view(-1).numpy() * 255).astype(int)

sample_hv = np.zeros(pixel_hvs.shape[1], dtype=int)
for i in range(28*28):
    sample_hv += bind(pixel_hvs[i], value_hvs[img_flat[i]])
sample_hv = (sample_hv >= (28*28/2)).astype(int)

print("Sample HV shape:", sample_hv.shape)
print("Number of 1s in sample HV:", np.sum(sample_hv))

# ---------------- Compare sample HV with all class HVs ----------------
def similarity(hv1, hv2):
    return np.sum(hv1 == hv2) / len(hv1)

sims = [similarity(sample_hv, class_hv[c]) for c in range(10)]
for c, sim in enumerate(sims):
    print(f"Similarity with class {c}: {sim:.4f}")

plt.figure(figsize=(6, 1))
plt.imshow(sample_hv[:1000].reshape(20, 50), cmap='gray')
plt.title(f"Sample HV (first 1000 dims), Label={label.item()}")
plt.axis('off')
plt.show()