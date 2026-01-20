import argparse
import numpy as np
import matplotlib.pyplot as plt

from load_fashion import get_fashion_mnist
from hdc import bind, similarity


# ---------------- FFT PIPELINE (SAME AS TRAIN) ----------------
def fft_lowblock(img_tensor, crop: int):
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


def quantize_amp(amp_log, levels: int):
    max_amp = float(amp_log.max())
    if max_amp <= 0.0:
        amp_norm = np.zeros_like(amp_log, dtype=np.float32)
    else:
        amp_norm = amp_log / max_amp

    amp_bin = np.floor(amp_norm * levels).astype(np.int32)
    amp_bin = np.clip(amp_bin, 0, levels - 1)
    return amp_bin


def quantize_phase(phase, levels: int):
    phase_norm = (phase + np.pi) / (2 * np.pi)
    phase_bin = np.floor(phase_norm * levels).astype(np.int32)
    phase_bin = np.clip(phase_bin, 0, levels - 1)
    return phase_bin


# ---------------- helpers ----------------
def top2_indices(arr: np.ndarray):
    # returns (best_idx, second_idx) by value
    # safe and fast
    idx = np.argpartition(arr, -2)[-2:]
    idx = idx[np.argsort(arr[idx])[::-1]]
    return int(idx[0]), int(idx[1])


def normalize_pair(a: int, b: int):
    return (a, b) if a < b else (b, a)


def load_pairwise_dims(npz_path: str):
    d = np.load(npz_path)
    # keys are like pair_2_4
    out = {}
    for k in d.files:
        if not k.startswith("pair_"):
            continue
        parts = k.split("_")
        a = int(parts[1]); b = int(parts[2])
        out[(a, b)] = d[k].astype(np.int32)
    return out


def main():
    ap = argparse.ArgumentParser(description="Test unified FFT-HDC with pairwise sentinel patch (top2 gating).")
    ap.add_argument("--config", default="data/config_unified.npz")
    ap.add_argument("--use_keep_dims", action="store_true")
    ap.add_argument("--keep_dims", default="data/keep_dims_v3a.npy")
    ap.add_argument("--pairwise_npz", default="analysis/pairwise_dims_kept.npz")
    ap.add_argument("--patch_mode", default="tie_break", choices=["tie_break", "replace_top2"],
                    help="tie_break: only change if pairwise decides opposite; replace_top2: always overwrite top2 using pairwise.")
    ap.add_argument("--min_global_margin", type=float, default=0.0,
                    help="Only apply patch if global margin (top1-top2) <= this value. 0 = always allow.")
    ap.add_argument("--save_examples", type=int, default=8)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    # ----- Load config -----
    cfg = np.load(args.config)
    DIM = int(cfg["DIM"])
    CROP = int(cfg["CROP"])
    AMP_LEVELS = int(cfg["AMP_LEVELS"])
    PHASE_LEVELS = int(cfg["PHASE_LEVELS"])
    ALPHA_AMP = float(cfg["ALPHA_AMP"])
    BETA_PHASE = float(cfg["BETA_PHASE"])
    NUM_CLASSES = 10

    print("Loaded config:",
          f"DIM={DIM}, block={(2*CROP)}x{(2*CROP)}, AMP_LEVELS={AMP_LEVELS}, PHASE_LEVELS={PHASE_LEVELS}")

    # ----- Load data -----
    _, test_loader = get_fashion_mnist(batch_size=1)
    print("Test size:", len(test_loader.dataset))

    # ----- Load HDC assets -----
    pos_hvs_amp = np.load("data/pos_hvs_amp_unified.npy")
    pos_hvs_phase = np.load("data/pos_hvs_phase_unified.npy")
    value_hvs_amp = np.load("data/value_hvs_amp_unified.npy")
    value_hvs_phase = np.load("data/value_hvs_phase_unified.npy")
    class_hv_full = np.load("data/class_hv_unified.npy").astype(np.uint8)

    keep = None
    if args.use_keep_dims:
        keep = np.load(args.keep_dims).astype(np.int32)
        class_hv = class_hv_full[:, keep]
        print(f"[prune] using keep_dims: {len(keep)}/{DIM}")
    else:
        class_hv = class_hv_full
        print("[prune] using full dims")

    # ----- Load pairwise dims -----
    pair_dims = load_pairwise_dims(args.pairwise_npz)
    print(f"[pairwise] loaded {len(pair_dims)} pairs from {args.pairwise_npz}")

    # ----- Test loop -----
    correct_base = 0
    correct_patch = 0
    total = 0

    # confusion pair stats
    applied = {k: 0 for k in pair_dims.keys()}
    flipped = {k: 0 for k in pair_dims.keys()}

    examples = []

    for idx, (img, label) in enumerate(test_loader):
        true_label = int(label.item())

        # encode sample
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

        thr = float(sample_hv_float.mean())
        sample_hv = (sample_hv_float >= thr).astype(np.uint8)

        if keep is not None:
            sample_eff = sample_hv[keep]
        else:
            sample_eff = sample_hv

        # baseline prediction
        sims = np.array([similarity(sample_eff, class_hv[c]) for c in range(NUM_CLASSES)], dtype=np.float64)
        top1, top2 = top2_indices(sims)
        base_pred = top1

        best = float(sims[top1])
        second = float(sims[top2])
        margin = best - second

        # patched prediction (default = baseline)
        patched_pred = base_pred

        # Apply patch only if top2 pair is in our list and margin condition satisfied
        a, b = normalize_pair(top1, top2)
        key = (a, b)
        if key in pair_dims and (args.min_global_margin <= 0.0 or margin <= args.min_global_margin):
            dims = pair_dims[key]
            applied[key] += 1

            # pairwise similarity on subset dims
            s1 = similarity(sample_eff[dims], class_hv[top1, dims])
            s2 = similarity(sample_eff[dims], class_hv[top2, dims])

            # decide winner within the pair
            pair_winner = top1 if s1 >= s2 else top2

            if args.patch_mode == "replace_top2":
                patched_pred = pair_winner
            else:
                # tie_break: only override if winner != baseline top1
                if pair_winner != base_pred:
                    patched_pred = pair_winner
                    flipped[key] += 1

        correct_base += int(base_pred == true_label)
        correct_patch += int(patched_pred == true_label)
        total += 1

        # save some debug examples: only where patch changed decision or where wrong
        if len(examples) < args.save_examples:
            if patched_pred != base_pred or base_pred != true_label:
                img_np = img.squeeze()
                try:
                    img_np = img_np.cpu()
                except Exception:
                    pass
                img_np = img_np.numpy().astype(np.float32)
                if img_np.max() > 1.5:
                    img_np = img_np / 255.0

                examples.append((img_np, true_label, base_pred, patched_pred, amp_bin.copy(), phase_bin.copy(), sims.copy()))

    acc_base = correct_base / total
    acc_patch = correct_patch / total

    print("\n=== Accuracy ===")
    print(f"baseline: {acc_base:.4f}")
    print(f"patched : {acc_patch:.4f}")
    print("================\n")

    print("=== Patch usage (applied / flipped) ===")
    for k in sorted(pair_dims.keys()):
        print(f"{k}: applied={applied[k]} flipped={flipped[k]}")
    print("======================================\n")

    # Visualization
    if args.show and len(examples) > 0:
        n = len(examples)
        fig, axes = plt.subplots(n, 4, figsize=(12, 3.2 * n))
        if n == 1:
            axes = np.array([axes])

        for i, (img_np, t, bpred, ppred, amp_bin, phase_bin, sims) in enumerate(examples):
            axes[i, 0].imshow(img_np, cmap="gray")
            axes[i, 0].set_title(f"Image\nT:{t} base:{bpred} patch:{ppred}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(amp_bin, cmap="inferno")
            axes[i, 1].set_title("Amp bins")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(phase_bin, cmap="twilight")
            axes[i, 2].set_title("Phase bins")
            axes[i, 2].axis("off")

            axes[i, 3].bar(range(NUM_CLASSES), sims)
            axes[i, 3].set_title("Global similarity")
            axes[i, 3].set_xticks(range(NUM_CLASSES))

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
