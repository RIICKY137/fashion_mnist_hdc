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


# ---------------- v4: Pairwise sentinel (class_hv only, KNIFE dims) ----------------
USE_PAIRWISE_SENTINEL = True

# Only arbitrate on these known confusion corridors
CONFUSION_PAIRS = {(2, 4), (4, 6), (0, 6)}

# Gate toggles (start permissive so flips actually happen)
USE_MARGIN_GATE = True
MARGIN_THR = 0.2
USE_PAIR_CONF_GATE = True
PAIR_CONF_THR = 0.2

# how many "knife" dims per pair (tune: 128, 256, 512, 1024)
PAIR_KNIFE_K = 512


def normalize_pair(a, b):
    return (a, b) if a < b else (b, a)


def top2_indices(arr):
    idx = np.argpartition(arr, -2)[-2:]
    idx = idx[np.argsort(arr[idx])[::-1]]
    return int(idx[0]), int(idx[1])


def build_pairwise_knife_dims(class_hv_space, a, b, k=512):
    """
    Uses ONLY class_hv_space.
    Returns top-k dims from S_ab where other classes are most consistent.
    """
    C, D = class_hv_space.shape
    a, b = normalize_pair(a, b)

    # disagreement set
    S = np.where(class_hv_space[a] != class_hv_space[b])[0]
    if S.size == 0:
        return S.astype(np.int32)

    # others consistency score
    others = [c for c in range(C) if c != a and c != b]
    p = class_hv_space[others][:, S].mean(axis=0)  # in [0,1]
    score = np.abs(p - 0.5)  # higher => others near 0 or 1 => cleaner structural bit

    kk = min(int(k), int(S.size))
    top = np.argpartition(score, -kk)[-kk:]
    top = top[np.argsort(score[top])[::-1]]
    knife = S[top].astype(np.int32)
    return knife


def pairwise_arbitrate(sample_hv_space, class_hv_space, a, b, dims):
    ma = np.mean(sample_hv_space[dims] == class_hv_space[a, dims])
    mb = np.mean(sample_hv_space[dims] == class_hv_space[b, dims])
    winner = a if ma >= 0.5 else b
    return winner, float(ma), float(mb)


# ---------------- LOAD DATA ----------------
_, test_loader = get_fashion_mnist(batch_size=batch_size)
print("Test set size:", len(test_loader.dataset))

# ---------------- LOAD HYPERVECTORS ----------------
pos_hvs_amp = np.load("data/pos_hvs_amp_unified.npy")
pos_hvs_phase = np.load("data/pos_hvs_phase_unified.npy")
value_hvs_amp = np.load("data/value_hvs_amp_unified.npy")
value_hvs_phase = np.load("data/value_hvs_phase_unified.npy")
class_hv_full = np.load("data/class_hv_unified.npy").astype(np.uint8)

print("\n=== Class HV ones ratio (unified) ===")
for c in range(NUM_CLASSES):
    print(f"Class {c}: {class_hv_full[c].mean():.4f}")
print("====================================\n")

# ---------------- v3: LOAD PRUNING DIMS (OPTIONAL) ----------------
use_pruning = False
keep_dims = None
class_hv = class_hv_full

try:
    keep_dims = np.load("data/keep_dims_v3a.npy").astype(np.int32)
    if keep_dims.ndim != 1:
        raise ValueError("keep_dims must be 1D indices")
    if keep_dims.size == 0:
        raise ValueError("keep_dims empty")
    if keep_dims.max() >= class_hv_full.shape[1]:
        raise ValueError("keep_dims out of range")

    class_hv = class_hv_full[:, keep_dims]  # pruned prototype space
    use_pruning = True
    print(f"[v3] Loaded keep_dims_v3a.npy: kept {len(keep_dims)}/{class_hv_full.shape[1]} "
          f"({len(keep_dims)/class_hv_full.shape[1]:.3f}) dims")
except Exception as e:
    print(f"[v3] Pruning disabled: {e}")

# ---------------- Precompute knife dims per pair (in current space) ----------------
pair_dims = {}
for (a, b) in CONFUSION_PAIRS:
    aa, bb = normalize_pair(a, b)
    pair_dims[(aa, bb)] = build_pairwise_knife_dims(class_hv, aa, bb, k=PAIR_KNIFE_K)

print("\n[v4] Pairwise sentinel (KNIFE) enabled:", USE_PAIRWISE_SENTINEL)
print("[v4] PAIR_KNIFE_K =", PAIR_KNIFE_K)
print("[v4] pairs:", sorted(pair_dims.keys()))
for k in sorted(pair_dims.keys()):
    print(f"  pair {k}: |knife|={len(pair_dims[k])}")
print()

# ---------------- TEST LOOP ----------------
correct_full = 0
correct_pruned = 0
correct_v4 = 0
total = 0

applied = {k: 0 for k in pair_dims.keys()}
flipped = {k: 0 for k in pair_dims.keys()}

examples = []

for idx, (img, label) in enumerate(test_loader):
    amp_log, phase = fft_lowblock(img, crop=CROP)
    amp_bin = quantize_amp(amp_log, levels=AMP_LEVELS)
    phase_bin = quantize_phase(phase, levels=PHASE_LEVELS)

    sample_hv_float = np.zeros(DIM, dtype=np.float32)

    kpos = 0
    for x in range(2 * CROP):
        for y in range(2 * CROP):
            a_idx = int(amp_bin[x, y])
            p_idx = int(phase_bin[x, y])

            hv_amp = bind(pos_hvs_amp[kpos], value_hvs_amp[a_idx]).astype(np.float32)
            hv_phase = bind(pos_hvs_phase[kpos], value_hvs_phase[p_idx]).astype(np.float32)

            sample_hv_float += ALPHA_AMP * hv_amp + BETA_PHASE * hv_phase
            kpos += 1

    # binarize by sample mean
    thr = float(sample_hv_float.mean())
    sample_hv_full = (sample_hv_float >= thr).astype(np.uint8)

    true_label = int(label.item())

    # ---- v2 baseline (full dims) ----
    sims_full = np.array([similarity(sample_hv_full, class_hv_full[c]) for c in range(NUM_CLASSES)], dtype=np.float64)
    pred_full = int(np.argmax(sims_full))
    correct_full += int(pred_full == true_label)

    # ---- v3 baseline (pruned dims if available) ----
    if use_pruning:
        sample_hv = sample_hv_full[keep_dims]
        sims = np.array([similarity(sample_hv, class_hv[c]) for c in range(NUM_CLASSES)], dtype=np.float64)
        pred_pruned = int(np.argmax(sims))
    else:
        sample_hv = sample_hv_full
        sims = sims_full
        pred_pruned = pred_full

    correct_pruned += int(pred_pruned == true_label)

    # ---- v4: pairwise sentinel arbitration (knife dims) ----
    pred_v4 = pred_pruned

    if USE_PAIRWISE_SENTINEL:
        top1, top2 = top2_indices(sims)
        a, b = normalize_pair(top1, top2)

        if (a, b) in pair_dims:
            margin = float(sims[top1] - sims[top2])

            if (not USE_MARGIN_GATE) or (margin <= MARGIN_THR):
                dims = pair_dims[(a, b)]
                applied[(a, b)] += 1

                winner, ma, mb = pairwise_arbitrate(sample_hv, class_hv, a, b, dims)

                if winner != pred_v4:
                    if (not USE_PAIR_CONF_GATE) or (abs(ma - 0.5) >= PAIR_CONF_THR):
                        pred_v4 = winner
                        flipped[(a, b)] += 1

    correct_v4 += int(pred_v4 == true_label)
    total += 1

    if len(examples) < 5:
        img_np = img.squeeze()
        try:
            img_np = img_np.cpu()
        except Exception:
            pass
        img_np = img_np.numpy().astype(np.float32)
        if img_np.max() > 1.5:
            img_np = img_np / 255.0

        examples.append(
            (img_np, true_label, pred_full, pred_pruned, pred_v4,
             amp_bin.copy(), phase_bin.copy(),
             sims_full.copy(), sims.copy())
        )

acc_full = correct_full / total
acc_pruned = correct_pruned / total
acc_v4 = correct_v4 / total

print(f"🔥 v2 baseline accuracy (full dims): {acc_full:.4f}")
if use_pruning:
    print(f"🔥 v3 pruned accuracy (kept dims):   {acc_pruned:.4f}")
    print(f"⚙️  dims kept: {len(keep_dims)}/{DIM} ({len(keep_dims)/DIM:.3f})")
else:
    print(f"🔥 v3 pruned accuracy (kept dims):   {acc_pruned:.4f}  (pruning disabled)")

print(f"🗡️  v4 pairwise-knife accuracy:       {acc_v4:.4f}")

print("\n=== v4 patch usage (applied / flipped) ===")
for k in sorted(pair_dims.keys()):
    print(f"{k}: applied={applied[k]} flipped={flipped[k]}")
print("=========================================\n")

# ---------------- VISUALIZATION ----------------
print("\nShowing 5 samples with amplitude / phase / similarity...\n")

fig, axes = plt.subplots(5, 4, figsize=(12, 14))
axes = axes.reshape(5, 4)

for i, (img_np, true_label, pred_full, pred_pruned, pred_v4, amp_bin, phase_bin, sims_full, sims) in enumerate(examples):
    axes[i, 0].imshow(img_np, cmap="gray")
    axes[i, 0].set_title(f"T:{true_label}\nF:{pred_full} P:{pred_pruned} V4:{pred_v4}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(amp_bin, cmap="inferno")
    axes[i, 1].set_title(f"FFT Amplitude bins (0..{AMP_LEVELS-1})")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(phase_bin, cmap="twilight")
    axes[i, 2].set_title(f"FFT Phase bins (0..{PHASE_LEVELS-1})")
    axes[i, 2].axis("off")

    axes[i, 3].bar(range(NUM_CLASSES), sims)
    axes[i, 3].set_title("Hamming Similarity (pruned)")
    axes[i, 3].set_xticks(range(NUM_CLASSES))

plt.tight_layout()
plt.show()
