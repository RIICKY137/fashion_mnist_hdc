import argparse
import os
import json
import numpy as np

from load_fashion import get_fashion_mnist
from hdc import bind


# ---------------- FFT PIPELINE (SAME AS TRAIN/TEST UNIFIED) ----------------
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


# ---------------- HV ENCODING ----------------
def encode_sample_hv_float(
    img_tensor,
    *,
    DIM: int,
    CROP: int,
    AMP_LEVELS: int,
    PHASE_LEVELS: int,
    ALPHA_AMP: float,
    BETA_PHASE: float,
    pos_hvs_amp: np.ndarray,
    pos_hvs_phase: np.ndarray,
    value_hvs_amp: np.ndarray,
    value_hvs_phase: np.ndarray,
):
    amp_log, phase = fft_lowblock(img_tensor, crop=CROP)
    amp_bin = quantize_amp(amp_log, levels=AMP_LEVELS)
    phase_bin = quantize_phase(phase, levels=PHASE_LEVELS)

    hv_float = np.zeros(DIM, dtype=np.float32)
    k = 0
    for x in range(2 * CROP):
        for y in range(2 * CROP):
            a_idx = int(amp_bin[x, y])
            p_idx = int(phase_bin[x, y])

            hv_amp = bind(pos_hvs_amp[k], value_hvs_amp[a_idx]).astype(np.float32)
            hv_phase = bind(pos_hvs_phase[k], value_hvs_phase[p_idx]).astype(np.float32)

            hv_float += ALPHA_AMP * hv_amp + BETA_PHASE * hv_phase
            k += 1
    return hv_float


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def parse_pairs(pairs_str: str):
    """
    pairs_str example: "2-4,4-6,6-0,7-9,7-5"
    returns list of (a,b) ints with a<b enforced.
    """
    out = []
    for tok in pairs_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" not in tok:
            raise ValueError(f"Bad pair token: {tok} (expected like 2-4)")
        a, b = tok.split("-", 1)
        a = int(a.strip()); b = int(b.strip())
        if a == b:
            continue
        if a > b:
            a, b = b, a
        out.append((a, b))
    # unique preserve order
    seen = set()
    uniq = []
    for ab in out:
        if ab not in seen:
            seen.add(ab)
            uniq.append(ab)
    return uniq


def main():
    ap = argparse.ArgumentParser(
        description="Build pairwise discriminative dimensions (pairwise sentinels) for unified FFT-HDC."
    )
    ap.add_argument("--config", default="data/config_unified.npz", help="config_unified.npz path")
    ap.add_argument("--pairs", default="2-4,4-6,6-0,7-9,7-5",
                    help='Comma list like "2-4,4-6,6-0"')
    ap.add_argument("--n_per_class", type=int, default=1500,
                    help="How many train samples per class used to estimate statistics for each pair.")
    ap.add_argument("--k_dims", type=int, default=256,
                    help="How many discriminative dims to keep per pair.")
    ap.add_argument("--use_keep_dims", action="store_true",
                    help="Apply keep_dims (constant pruning) and build pairwise dims in kept-space.")
    ap.add_argument("--keep_dims", default="data/keep_dims_v3a.npy",
                    help="keep dims file (must align with your test run).")
    ap.add_argument("--out_npz", default="analysis/pairwise_dims_kept.npz",
                    help="Output npz file to store pairwise dims.")
    ap.add_argument("--out_meta", default="analysis/pairwise_dims_kept_meta.json",
                    help="Output json metadata.")
    ap.add_argument("--seed", type=int, default=0, help="random seed for reproducible sampling")
    ap.add_argument("--max_train", type=int, default=0,
                    help="If >0, only scan first N train samples (debug).")
    args = ap.parse_args()

    np.random.seed(args.seed)

    cfg = np.load(args.config)
    DIM = int(cfg["DIM"])
    CROP = int(cfg["CROP"])
    AMP_LEVELS = int(cfg["AMP_LEVELS"])
    PHASE_LEVELS = int(cfg["PHASE_LEVELS"])
    ALPHA_AMP = float(cfg["ALPHA_AMP"])
    BETA_PHASE = float(cfg["BETA_PHASE"])

    pairs = parse_pairs(args.pairs)
    print(f"Loaded config: DIM={DIM}, block={(2*CROP)}x{(2*CROP)}, AMP_LEVELS={AMP_LEVELS}, PHASE_LEVELS={PHASE_LEVELS}")
    print(f"Pairs: {pairs}")
    print(f"n_per_class={args.n_per_class}, k_dims={args.k_dims}, use_keep_dims={args.use_keep_dims}")

    # Load train data
    train_loader, _ = get_fashion_mnist(batch_size=1)
    print("Train size:", len(train_loader.dataset))

    # Load HDC assets
    pos_hvs_amp = np.load("data/pos_hvs_amp_unified.npy")
    pos_hvs_phase = np.load("data/pos_hvs_phase_unified.npy")
    value_hvs_amp = np.load("data/value_hvs_amp_unified.npy")
    value_hvs_phase = np.load("data/value_hvs_phase_unified.npy")

    keep = None
    if args.use_keep_dims:
        keep = np.load(args.keep_dims).astype(np.int32)
        D_eff = len(keep)
        print(f"[prune] building in kept-space: {D_eff}/{DIM} dims")
    else:
        D_eff = DIM
        print("[prune] building in full-space")

    # Collect indices of samples per class (streaming)
    # We store hv_float for selected samples only to keep memory under control.
    target_classes = sorted({c for ab in pairs for c in ab})
    cls_hvs = {c: [] for c in target_classes}

    # Stream through train set and collect up to n_per_class hv_float per needed class
    counts = {c: 0 for c in target_classes}
    done = set()

    for i, (img, label) in enumerate(train_loader):
        if args.max_train > 0 and i >= args.max_train:
            break
        c = int(label.item())
        if c not in cls_hvs:
            continue
        if counts[c] >= args.n_per_class:
            done.add(c)
            if len(done) == len(target_classes):
                break
            continue

        hv_float = encode_sample_hv_float(
            img,
            DIM=DIM, CROP=CROP,
            AMP_LEVELS=AMP_LEVELS, PHASE_LEVELS=PHASE_LEVELS,
            ALPHA_AMP=ALPHA_AMP, BETA_PHASE=BETA_PHASE,
            pos_hvs_amp=pos_hvs_amp,
            pos_hvs_phase=pos_hvs_phase,
            value_hvs_amp=value_hvs_amp,
            value_hvs_phase=value_hvs_phase,
        )
        if keep is not None:
            hv_float = hv_float[keep]

        cls_hvs[c].append(hv_float)
        counts[c] += 1

        if (i + 1) % 2000 == 0:
            msg = " ".join([f"{k}:{counts[k]}" for k in target_classes])
            print(f"Scanned {i+1} train samples... collected {msg}")

    # Convert lists to arrays
    for c in target_classes:
        if counts[c] < 10:
            raise RuntimeError(f"Too few samples collected for class {c}: {counts[c]}")
        cls_hvs[c] = np.stack(cls_hvs[c], axis=0).astype(np.float32)
        print(f"Collected class {c}: {cls_hvs[c].shape}")

    # Build pairwise discriminative dims
    pair_dims = {}
    meta = {
        "config": {
            "DIM_full": DIM,
            "D_eff": int(D_eff),
            "CROP": CROP,
            "AMP_LEVELS": AMP_LEVELS,
            "PHASE_LEVELS": PHASE_LEVELS,
            "ALPHA_AMP": ALPHA_AMP,
            "BETA_PHASE": BETA_PHASE,
            "use_keep_dims": bool(args.use_keep_dims),
            "keep_dims_file": args.keep_dims if args.use_keep_dims else None,
        },
        "pairs": [],
        "n_per_class": int(args.n_per_class),
        "k_dims": int(args.k_dims),
        "seed": int(args.seed),
    }

    eps = 1e-6

    for (a, b) in pairs:
        A = cls_hvs[a]
        B = cls_hvs[b]

        mu_a = A.mean(axis=0)
        mu_b = B.mean(axis=0)
        std_a = A.std(axis=0)
        std_b = B.std(axis=0)

        # Effect size style score: |mu_a - mu_b| / (std_a + std_b + eps)
        score = np.abs(mu_a - mu_b) / (std_a + std_b + eps)

        # Take top-k dims
        k = min(args.k_dims, score.shape[0])
        topk = np.argpartition(score, -k)[-k:]
        topk = topk[np.argsort(score[topk])[::-1]].astype(np.int32)

        key = f"pair_{a}_{b}"
        pair_dims[key] = topk

        meta["pairs"].append({
            "pair": [a, b],
            "key": key,
            "k": int(k),
            "mean_abs_diff_topk": float(np.mean(np.abs(mu_a[topk] - mu_b[topk]))),
            "mean_score_topk": float(np.mean(score[topk])),
            "max_score": float(score[topk[0]]),
        })

        print(f"[ok] {key}: selected {k} dims | max_score={float(score[topk[0]]):.6f} | mean_score_topk={float(np.mean(score[topk])):.6f}")

    # Save outputs
    out_dir = os.path.dirname(args.out_npz)
    if out_dir:
        ensure_dir(out_dir)

    np.savez(args.out_npz, **pair_dims)
    with open(args.out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved pairwise dims to: {args.out_npz}")
    print(f"Saved metadata to:     {args.out_meta}")
    print("✅ build_pairwise_sentinels complete.")


if __name__ == "__main__":
    main()
