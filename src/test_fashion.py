import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from src.load_fashion import get_fashion_mnist
from src.hdc import bind, similarity

# ---------------- Config ----------------
DIM = 2**14  # 超向量维度
NUM_CLASSES = 10
PIXEL_THRESHOLD = 25
batch_size = 1

# ---------------- Preprocessing ----------------
def preprocess_image(img_tensor):
    img = img_tensor.squeeze().numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.8)
    img[img < PIXEL_THRESHOLD] = 0
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- Load test data ----------------
_, test_loader = get_fashion_mnist(batch_size=batch_size)
print("Test set size:", len(test_loader.dataset))

# ---------------- Load trained vectors ----------------
pixel_hvs = np.load("pixel_hvs.npy")
value_hvs = np.load("value_hvs.npy")
class_hv = np.load("class_hv.npy")

# ---------------- Test loop ----------------
correct = 0
total = 0
examples = []

for img, label in test_loader:
    img = preprocess_image(img)
    img_flat = (img.reshape(-1) * 255).astype(int)

    # encode sample hypervector
    sample_hv = np.zeros(DIM, dtype=int)
    for i in range(28 * 28):
        pos_hv = pixel_hvs[i]
        val_hv = value_hvs[img_flat[i]]
        sample_hv += bind(pos_hv, val_hv)

    # ✅ binarize sample_hv to match class_hv (both 0/1 before Hamming)
    thr = sample_hv.mean()
    sample_hv_bin = (sample_hv >= thr).astype(int)

    # compute similarity with each class (both binary now)
    sims = [similarity(sample_hv_bin, class_hv[c]) for c in range(NUM_CLASSES)]
    pred = np.argmax(sims)

    if len(examples) < 10:
        examples.append((img, label.item(), pred))

    correct += (pred == label.item())
    total += 1

# ---------------- Accuracy ----------------
accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")

# ---------------- Visualization ----------------
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
axes = axes.flatten()
for i, (img, label, pred) in enumerate(examples):
    axes[i].imshow(img.squeeze(), cmap="gray")
    axes[i].set_title(f"T:{label} / P:{pred}")
    axes[i].axis("off")

plt.suptitle("Fashion-MNIST Predictions (T=True, P=Pred)")
plt.tight_layout()
plt.show()
