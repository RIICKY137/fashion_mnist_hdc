import numpy as np
import matplotlib.pyplot as plt

DIM = 2**14     # 16384
NUM_CLASSES = 10

# ====== 加载你训练好的向量 ======
class_hv = np.load("data/class_hv_fft.npy")        # shape (10, 16384)
pos_hvs   = np.load("data/pos_hvs_fft.npy")         # shape (144, 16384) if FFT_SIZE=12
value_hvs = np.load("data/value_hvs_fft.npy")       # shape (256, 16384)


def show_hv_grid(hv, title, size=128):

    hv = hv.astype(float)

    # 自动裁剪或补齐
    needed = size*size
    if len(hv) > needed:
        hv = hv[:needed]
    else:
        hv = np.pad(hv, (0, needed-len(hv)), mode='constant')

    hv = hv.reshape(size, size)

    plt.imshow(hv, cmap="gray")
    plt.title(title)
    plt.axis("off")


# ====== 图 1：显示每个类别的 class HV ======
plt.figure(figsize=(12, 6))
for i in range(NUM_CLASSES):
    plt.subplot(2, 5, i+1)
    show_hv_grid(class_hv[i], f"Class {i}")
plt.suptitle("Class Hypervectors")
plt.tight_layout()
plt.show()

# ====== 图 2：可视化其中一些 FFT-pos hypervectors ======
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    show_hv_grid(pos_hvs[i], f"pos {i}")
plt.suptitle("Position Hypervectors (FFT features)")
plt.tight_layout()
plt.show()

# ====== 图 3：可视化 value hypervectors (像素/FFT强度编码) ======
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    show_hv_grid(value_hvs[i], f"value {i}")
plt.suptitle("Value Hypervectors")
plt.tight_layout()
plt.show()
