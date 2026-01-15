import numpy as np
from load_fashion import get_fashion_mnist
from hdc import generate_random_hypervector, bind  # bind is XOR for 0/1 vectors

# ---------------- CONFIG ----------------
DIM = 2**14
NUM_CLASSES = 10
batch_size = 1

# FFT low-frequency block is (2*CROP) x (2*CROP)
CROP = 12              # => 24x24 low-frequency block
NUM_FEATS = (2 * CROP) * (2 * CROP)

# Quantization levels
AMP_LEVELS = 32        # <<<<<< changed from 256 (critical)
PHASE_LEVELS = 16

# Relative importance
ALPHA_AMP = 1.0
BETA_PHASE = 0.7

# Optional: make results reproducible (only affects random HV generation)
np.random.seed(0)


# ---------------- FFT PIPELINE (UNIFIED) ----------------
def fft_lowblock(img_tensor, crop=CROP):
    """
    Returns:
      amp_log: (2*crop, 2*crop) float32, log1p(|FFT|) of centered low-frequency block
      phase:   (2*crop, 2*crop) float32, angle in (-pi, pi]
    """
    img = img_tensor.squeeze()
    # handle torch tensor on GPU/CPU safely
    try:
        img = img.cpu()
    except Exception:
        pass
    img = img.numpy().astype(np.float32)

    # If data is in 0..255, normalize; if already 0..1, keep as is
    if img.max() > 1.5:
        img = img / 255.0

    F = np.fft.fft2(img)
    F = np.fft.fftshift(F)  # center low-frequency at image center

    h, w = F.shape
    ch, cw = h // 2, w // 2
    block = F[ch - crop: ch + crop, cw - crop: cw + crop]  # (2*crop, 2*crop)

    amp_log = np.log1p(np.abs(block)).astype(np.float32)
    phase = np.angle(block).astype(np.float32)
    return amp_log, phase


def quantize_amp(amp_log, levels=AMP_LEVELS):
    """
    amp_log >= 0
    normalize per-sample by max then bucket into [0..levels-1]
    """
    max_amp = float(amp_log.max())
    if max_amp <= 0.0:
        amp_norm = np.zeros_like(amp_log, dtype=np.float32)
    else:
        amp_norm = amp_log / max_amp

    amp_bin = np.floor(amp_norm * levels).astype(np.int32)
    amp_bin = np.clip(amp_bin, 0, levels - 1)
    return amp_bin


def quantize_phase(phase, levels=PHASE_LEVELS):
    """
    phase in (-pi, pi] -> [0,1) -> bucket [0..levels-1]
    """
    phase_norm = (phase + np.pi) / (2 * np.pi)  # [0,1]
    phase_bin = np.floor(phase_norm * levels).astype(np.int32)
    phase_bin = np.clip(phase_bin, 0, levels - 1)
    return phase_bin


# ---------------- LOAD DATA ----------------
train_loader, _ = get_fashion_mnist(batch_size=batch_size)
print("Training set size:", len(train_loader.dataset))

# ---------------- HYPERVECTOR INITIALIZATION ----------------
# Position HVs (independent for amp and phase)
pos_hvs_amp = np.array([generate_random_hypervector(DIM) for _ in range(NUM_FEATS)], dtype=np.uint8)
pos_hvs_phase = np.array([generate_random_hypervector(DIM) for _ in range(NUM_FEATS)], dtype=np.uint8)

# Value HVs (AMP_LEVELS and PHASE_LEVELS)
value_hvs_amp = np.array([generate_random_hypervector(DIM) for _ in range(AMP_LEVELS)], dtype=np.uint8)
value_hvs_phase = np.array([generate_random_hypervector(DIM) for _ in range(PHASE_LEVELS)], dtype=np.uint8)

# Class accumulators (float)
class_accum = np.zeros((NUM_CLASSES, DIM), dtype=np.float32)

# ---------------- TRAINING LOOP ----------------
for idx, (img, label) in enumerate(train_loader):
    amp_log, phase = fft_lowblock(img, crop=CROP)
    amp_bin = quantize_amp(amp_log, levels=AMP_LEVELS)
    phase_bin = quantize_phase(phase, levels=PHASE_LEVELS)

    sample_hv_float = np.zeros(DIM, dtype=np.float32)

    k = 0
    for x in range(2 * CROP):
        for y in range(2 * CROP):
            a_idx = int(amp_bin[x, y])
            p_idx = int(phase_bin[x, y])

            hv_amp = bind(pos_hvs_amp[k], value_hvs_amp[a_idx]).astype(np.float32)   # 0/1 -> float
            hv_phase = bind(pos_hvs_phase[k], value_hvs_phase[p_idx]).astype(np.float32)

            sample_hv_float += ALPHA_AMP * hv_amp + BETA_PHASE * hv_phase
            k += 1

    class_accum[int(label.item())] += sample_hv_float

    if (idx + 1) % 10000 == 0:
        print(f"Processed {idx + 1} samples...")

# ---------------- FINAL CLASS HYPERVECTORS ----------------
# Per-class mean threshold
thr = np.mean(class_accum, axis=1, keepdims=True)
class_hv = (class_accum >= thr).astype(np.uint8)

print("\n=== Class HV ones ratio (unified FFT, phase+amp) ===")
for c in range(NUM_CLASSES):
    print(f"Class {c}: {class_hv[c].mean():.4f}")
print("===============================================\n")

# ---------------- SAVE ----------------
np.save("data/pos_hvs_amp_unified.npy", pos_hvs_amp)
np.save("data/pos_hvs_phase_unified.npy", pos_hvs_phase)
np.save("data/value_hvs_amp_unified.npy", value_hvs_amp)
np.save("data/value_hvs_phase_unified.npy", value_hvs_phase)
np.save("data/class_hv_unified.npy", class_hv)

# Save config so test script can auto-match
np.savez(
    "data/config_unified.npz",
    DIM=DIM,
    CROP=CROP,
    AMP_LEVELS=AMP_LEVELS,
    PHASE_LEVELS=PHASE_LEVELS,
    ALPHA_AMP=ALPHA_AMP,
    BETA_PHASE=BETA_PHASE
)

print("✅ Training complete! Saved unified FFT-HDC hypervectors.")
