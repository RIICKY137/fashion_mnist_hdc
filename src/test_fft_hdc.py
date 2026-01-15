import numpy as np
import matplotlib.pyplot as plt
from load_fashion import get_fashion_mnist
from hdc import bind, similarity

# ---------------- LOAD CONFIG ----------------
cfg = np.load("data/config_unified.npz")
DIM = int(cfg["DIM"])
CROP = int(cfg["CROP"])
AMP_LEVELS = int(cfg["AMP_LEVELS"])
PHASE_LEVELS = int(cfg["PHASE_LEVELS"])
ALPHA_AMP = float(cfg["ALPHA_AMP"])
BETA_PHASE = float(cfg["BETA_PHASE"])

NUM_CLASSES = 10
batch_size = 1
NUM_FEATS = (2 * CROP) * (2 * CROP)

print("Loaded config:",
      f"DIM={DIM}, block={(2*CROP)}x{(2*CROP)}, AMP_LEVELS={AMP_LEVELS}, PHASE_LEVELS={PHASE_LEVELS}")

# ---------------- FFT PIPELINE (SAME AS TRAIN) ----------------
def fft_lowblock(img_tensor, crop=CROP):
    img = img_tensor.squeeze()
    try:
        img = img.cpu()
    except Exception:
        pass
    img = img.numpy().astype(np.float32)

    if img.max() > 1.5:
        img = img / 255.0

    F = np.fft.fft2(img)
    F = np.fft.fftshift(F)
    h, w = F.shape
    ch, cw = h // 2, w // 2
    block = F[ch - crop: ch + crop, cw - crop: cw + crop]
    amp_log = np.log1p(np.abs(block)).astype(np.float32)
    phase = np.angle(block).astype(np.float32)
    return amp_log, phase


def quantize_amp(amp_log, levels=AMP_LEVELS):
    max_amp = float(amp_log.max())
    if max_amp <= 0.0:
        amp_norm = np.zeros_like(amp_log, dtype=np.float32)
    else:
        amp_norm = amp_log / max_amp

    amp_bin = np.floor(amp_norm * levels).astype(np.int32)
    amp_bin = np.clip(amp_bin, 0, levels - 1)
    return amp_bin


def quantize_phase(phase, levels=PHASE_LEVELS):
    phase_norm = (phase + np.pi) / (2 * np.pi)
    phase_bin = np.floor(phase_norm * levels).astype(np.int32)
    phase_bin = np.clip(phase_bin, 0, levels - 1)
    return phase_bin


# ---------------- LOAD DATA ----------------
_, test_loader = get_fashion_mnist(batch_size=batch_size)
print("Test set size:", len(test_loader.dataset))

# ---------------- LOAD HYPERVECTORS ----------------
pos_hvs_amp = np.load("data/pos_hvs_amp_unified.npy")
pos_hvs_phase = np.load("data/pos_hvs_phase_unified.npy")
value_hvs_amp = np.load("data/value_hvs_amp_unified.npy")
value_hvs_phase = np.load("data/value_hvs_phase_unified.npy")
class_hv = np.load("data/class_hv_unified.npy")

print("\n=== Class HV ones ratio (unified) ===")
for c in range(NUM_CLASSES):
    print(f"Class {c}: {class_hv[c].mean():.4f}")
print("====================================\n")

# ---------------- TEST LOOP ----------------
correct = 0
total = 0
examples = []

for idx, (img, label) in enumerate(test_loader):
    amp_log, phase = fft_lowblock(img, crop=CROP)
    amp_bin = quantize_amp(amp_log, levels=AMP_LEVELS)
    phase_bin = quantize_phase(phase, levels=PHASE_LEVELS)

    sample_hv_float = np.zeros(DIM, dtype=np.float32)

    k = 0
    for x in range(2 * CROP):
        for y in range(2 * CROP):
            a_idx = int(amp_bin[x, y])
            p_idx = int(phase_bin[x, y])

            hv_amp = bind(pos_hvs_amp[k], value_hvs_amp[a_idx]).astype(np.float32)
            hv_phase = bind(pos_hvs_phase[k], value_hvs_phase[p_idx]).astype(np.float32)

            sample_hv_float += ALPHA_AMP * hv_amp + BETA_PHASE * hv_phase
            k += 1

    # binarize by sample mean
    thr = float(sample_hv_float.mean())
    sample_hv = (sample_hv_float >= thr).astype(np.uint8)

    sims = [similarity(sample_hv, class_hv[c]) for c in range(NUM_CLASSES)]
    pred = int(np.argmax(sims))

    correct += (pred == int(label.item()))
    total += 1

    if len(examples) < 5:
        # store original image (0..1) for display
        img_np = img.squeeze()
        try:
            img_np = img_np.cpu()
        except Exception:
            pass
        img_np = img_np.numpy().astype(np.float32)
        if img_np.max() > 1.5:
            img_np = img_np / 255.0

        examples.append((img_np, int(label.item()), pred, amp_bin.copy(), phase_bin.copy(), np.array(sims)))

accuracy = correct / total
print(f"🔥 Unified FFT-HDC Test Accuracy: {accuracy:.4f}")

# ---------------- VISUALIZATION ----------------
print("\nShowing 5 samples with amplitude / phase / similarity...\n")

fig, axes = plt.subplots(5, 4, figsize=(12, 14))
axes = axes.reshape(5, 4)

for i, (img_np, true_label, pred_label, amp_bin, phase_bin, sims) in enumerate(examples):
    axes[i, 0].imshow(img_np, cmap="gray")
    axes[i, 0].set_title(f"Image\nT:{true_label} P:{pred_label}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(amp_bin, cmap="inferno")
    axes[i, 1].set_title(f"FFT Amplitude bins (0..{AMP_LEVELS-1})")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(phase_bin, cmap="twilight")
    axes[i, 2].set_title(f"FFT Phase bins (0..{PHASE_LEVELS-1})")
    axes[i, 2].axis("off")

    axes[i, 3].bar(range(NUM_CLASSES), sims)
    axes[i, 3].set_title("Hamming Similarity")
    axes[i, 3].set_xticks(range(NUM_CLASSES))

plt.tight_layout()
plt.show()
