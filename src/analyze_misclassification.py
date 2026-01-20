import argparse
import json
import os
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from load_fashion import get_fashion_mnist
from hdc import bind, similarity


# ---------------- FFT PIPELINE (SAME AS YOUR UNIFIED VERSION) ----------------
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


# ---------------- Utilities ----------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b != 0 else 0.0


def load_optional_npz(path: str):
    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        return None


def compute_metrics_from_confusion(conf: np.ndarray):
    C = conf.shape[0]
    tp = np.diag(conf).astype(np.float64)
    support = conf.sum(axis=1).astype(np.float64)          # true count per class
    pred_count = conf.sum(axis=0).astype(np.float64)       # predicted count per class

    recall = np.array([safe_div(tp[i], support[i]) for i in range(C)], dtype=np.float64)
    precision = np.array([safe_div(tp[i], pred_count[i]) for i in range(C)], dtype=np.float64)
    f1 = np.array([
        safe_div(2 * precision[i] * recall[i], precision[i] + recall[i])
        for i in range(C)
    ], dtype=np.float64)

    acc = safe_div(tp.sum(), conf.sum())
    return acc, precision, recall, f1, support, pred_count


@dataclass
class ErrorCase:
    idx: int
    true_label: int
    pred_label: int
    sims_full: np.ndarray
    margin: float
    img_np: np.ndarray
    amp_bin: np.ndarray
    phase_bin: np.ndarray
    # sentinel stats (optional)
    true_sentinel_hit: int | None = None
    true_sentinel_total: int | None = None
    pred_sentinel_hit: int | None = None
    pred_sentinel_total: int | None = None


def main():
    ap = argparse.ArgumentParser(
        description="Misclassification analyzer for unified FFT-HDC (Fashion-MNIST)."
    )
    ap.add_argument("--config", default="data/config_unified.npz", help="config_unified.npz path")
    ap.add_argument("--data_split", default="test", choices=["test", "train"],
                    help="Evaluate on test or train split (usually test).")
    ap.add_argument("--batch_size", type=int, default=1, help="keep 1 for analysis")
    ap.add_argument("--max_samples", type=int, default=0,
                    help="If >0, only process first N samples (debug/fast).")
    ap.add_argument("--use_keep_dims", action="store_true",
                    help="Apply keep_dims (e.g., constant-only pruning) before similarity.")
    ap.add_argument("--keep_dims", default="data/keep_dims_v3.npy",
                    help="keep_dims file used when --use_keep_dims is set.")
    ap.add_argument("--sentinel_npz", default="analysis/sentinel_dims_kept.npz",
                    help="sentinel dims npz from analyze_classhv_features.py (optional).")
    ap.add_argument("--out_dir", default="analysis/misclf", help="output directory")
    ap.add_argument("--save_plots", action="store_true", help="save plots to out_dir/plots")
    ap.add_argument("--show", action="store_true", help="show plots interactively")
    ap.add_argument("--top_pairs", type=int, default=10, help="how many confusion pairs to list")
    ap.add_argument("--save_examples", type=int, default=12,
                    help="how many representative error examples to save as images")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    plots_dir = os.path.join(args.out_dir, "plots")
    if args.save_plots:
        ensure_dir(plots_dir)

    # ----- Load config -----
    cfg = np.load(args.config)
    DIM = int(cfg["DIM"])
    CROP = int(cfg["CROP"])
    AMP_LEVELS = int(cfg["AMP_LEVELS"])
    PHASE_LEVELS = int(cfg["PHASE_LEVELS"])
    ALPHA_AMP = float(cfg["ALPHA_AMP"])
    BETA_PHASE = float(cfg["BETA_PHASE"])

    NUM_CLASSES = 10
    NUM_FEATS = (2 * CROP) * (2 * CROP)

    print("Loaded config:",
          f"DIM={DIM}, block={(2*CROP)}x{(2*CROP)}, AMP_LEVELS={AMP_LEVELS}, PHASE_LEVELS={PHASE_LEVELS}")

    # ----- Load data -----
    train_loader, test_loader = get_fashion_mnist(batch_size=args.batch_size)
    loader = test_loader if args.data_split == "test" else train_loader
    print(f"Using split={args.data_split}, dataset size={len(loader.dataset)}")

    # ----- Load HDC assets -----
    pos_hvs_amp = np.load("data/pos_hvs_amp_unified.npy")
    pos_hvs_phase = np.load("data/pos_hvs_phase_unified.npy")
    value_hvs_amp = np.load("data/value_hvs_amp_unified.npy")
    value_hvs_phase = np.load("data/value_hvs_phase_unified.npy")
    class_hv = np.load("data/class_hv_unified.npy")

    # ----- Optional keep_dims pruning -----
    keep_dims = None
    if args.use_keep_dims:
        keep_dims = np.load(args.keep_dims).astype(np.int32)
        class_hv = class_hv[:, keep_dims]
        print(f"[prune] use_keep_dims enabled: kept {len(keep_dims)}/{DIM} dims")

    D_eff = class_hv.shape[1]

    # ----- Optional sentinel dims -----
    sentinel = load_optional_npz(args.sentinel_npz)
    has_sentinel = sentinel is not None
    if has_sentinel:
        # expect keys like class_0, class_0_pos, class_0_neg, ...
        print(f"[sentinel] loaded: {args.sentinel_npz}")
    else:
        print(f"[sentinel] not found or failed to load: {args.sentinel_npz} (sentinel stats disabled)")

    # ----- Evaluation containers -----
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    margins = []
    correct = 0
    total = 0
    error_cases: list[ErrorCase] = []

    # To save similarity distributions per true class
    # (optional: can be large; keep only mean)
    sims_sum_by_true = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    sims_count_by_true = np.zeros(NUM_CLASSES, dtype=np.int64)

    # ----- Main loop -----
    for idx, (img, label) in enumerate(loader):
        if args.max_samples > 0 and idx >= args.max_samples:
            break

        true_label = int(label.item())

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

        thr = float(sample_hv_float.mean())
        sample_hv = (sample_hv_float >= thr).astype(np.uint8)

        if keep_dims is not None:
            sample_hv_eff = sample_hv[keep_dims]
        else:
            sample_hv_eff = sample_hv

        sims = np.array([similarity(sample_hv_eff, class_hv[c]) for c in range(NUM_CLASSES)], dtype=np.float64)
        pred_label = int(np.argmax(sims))

        # margin = best - second_best
        best = float(sims[pred_label])
        second_best = float(np.partition(sims, -2)[-2])
        margin = best - second_best
        margins.append(margin)

        conf[true_label, pred_label] += 1
        sims_sum_by_true[true_label] += sims
        sims_count_by_true[true_label] += 1

        correct += int(pred_label == true_label)
        total += 1

        if pred_label != true_label:
            # cache a limited set for later plotting (we'll sort by margin)
            img_np = img.squeeze()
            try:
                img_np = img_np.cpu()
            except Exception:
                pass
            img_np = img_np.numpy().astype(np.float32)
            if img_np.max() > 1.5:
                img_np = img_np / 255.0

            ec = ErrorCase(
                idx=idx,
                true_label=true_label,
                pred_label=pred_label,
                sims_full=sims,
                margin=margin,
                img_np=img_np,
                amp_bin=amp_bin.copy(),
                phase_bin=phase_bin.copy()
            )

            if has_sentinel:
                # sentinel dims are in "kept space" if you analyzed kept space.
                # Here we analyze on the same space used for similarity (sample_hv_eff).
                tkey = f"class_{true_label}"
                pkey = f"class_{pred_label}"
                if (tkey in sentinel.files) and (pkey in sentinel.files):
                    t_dims = sentinel[tkey].astype(np.int32)
                    p_dims = sentinel[pkey].astype(np.int32)
                    # if you didn't use keep_dims in this run but sentinel was computed in kept space,
                    # indices won't align. We'll do a sanity guard:
                    if t_dims.size > 0 and int(t_dims.max()) < len(sample_hv_eff) and int(p_dims.max()) < len(sample_hv_eff):
                        ec.true_sentinel_hit = int(sample_hv_eff[t_dims].sum())
                        ec.true_sentinel_total = int(t_dims.size)
                        ec.pred_sentinel_hit = int(sample_hv_eff[p_dims].sum())
                        ec.pred_sentinel_total = int(p_dims.size)

            error_cases.append(ec)

        if (idx + 1) % 2000 == 0:
            print(f"Processed {idx+1} samples...")

    acc = safe_div(correct, total)
    print(f"\n[done] processed {total} samples. accuracy={acc:.4f}  D_eff={D_eff}")

    # ----- Metrics -----
    acc2, precision, recall, f1, support, pred_count = compute_metrics_from_confusion(conf)
    assert abs(acc2 - acc) < 1e-9

    # per-class mean similarity profile (useful for diagnosing confusion structure)
    mean_sims_by_true = np.zeros_like(sims_sum_by_true)
    for c in range(NUM_CLASSES):
        if sims_count_by_true[c] > 0:
            mean_sims_by_true[c] = sims_sum_by_true[c] / float(sims_count_by_true[c])

    # ----- Top confusion pairs -----
    pair_counts = []
    for t in range(NUM_CLASSES):
        for p in range(NUM_CLASSES):
            if t != p and conf[t, p] > 0:
                pair_counts.append((int(conf[t, p]), t, p))
    pair_counts.sort(reverse=True)
    top_pairs = pair_counts[:args.top_pairs]

    # ----- Save numeric outputs -----
    out = {
        "split": args.data_split,
        "accuracy": acc,
        "D_eff": int(D_eff),
        "use_keep_dims": bool(args.use_keep_dims),
        "num_samples": int(total),
        "per_class": []
    }
    for c in range(NUM_CLASSES):
        # most confused-to class for c (excluding itself)
        row = conf[c].copy()
        row[c] = 0
        worst_to = int(np.argmax(row)) if row.sum() > 0 else -1
        worst_to_cnt = int(row[worst_to]) if worst_to >= 0 else 0

        out["per_class"].append({
            "class": c,
            "support": int(support[c]),
            "pred_count": int(pred_count[c]),
            "precision": float(precision[c]),
            "recall": float(recall[c]),
            "f1": float(f1[c]),
            "most_confused_to": worst_to,
            "most_confused_to_count": worst_to_cnt
        })

    out["top_confusion_pairs"] = [{"count": int(cnt), "true": t, "pred": p} for (cnt, t, p) in top_pairs]

    with open(os.path.join(args.out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    np.save(os.path.join(args.out_dir, "confusion_matrix.npy"), conf)
    np.save(os.path.join(args.out_dir, "margins.npy"), np.array(margins, dtype=np.float64))
    np.save(os.path.join(args.out_dir, "mean_sims_by_true.npy"), mean_sims_by_true)

    print(f"[ok] saved: {args.out_dir}/report.json, confusion_matrix.npy, margins.npy, mean_sims_by_true.npy")

    # ----- Plots -----
    def maybe_savefig(name: str):
        if args.save_plots:
            path = os.path.join(plots_dir, name)
            plt.savefig(path, dpi=200)
            print(f"[ok] Saved plot: {path}")

    # Plot confusion matrix (raw counts)
    plt.figure(figsize=(7, 6))
    plt.imshow(conf, interpolation="nearest", aspect="auto")
    plt.title(f"Confusion matrix (counts) | acc={acc:.4f} | D={D_eff}")
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.xticks(range(NUM_CLASSES))
    plt.yticks(range(NUM_CLASSES))
    plt.colorbar()
    plt.tight_layout()
    maybe_savefig("confusion_counts.png")
    if args.show:
        plt.show()
    else:
        plt.close()

    # Plot confusion matrix (row-normalized = recall distribution)
    conf_norm = conf.astype(np.float64)
    row_sums = conf_norm.sum(axis=1, keepdims=True)
    conf_norm = np.divide(conf_norm, np.maximum(row_sums, 1.0))
    plt.figure(figsize=(7, 6))
    plt.imshow(conf_norm, interpolation="nearest", aspect="auto", vmin=0.0, vmax=1.0)
    plt.title("Confusion matrix (row-normalized)")
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.xticks(range(NUM_CLASSES))
    plt.yticks(range(NUM_CLASSES))
    plt.colorbar()
    plt.tight_layout()
    maybe_savefig("confusion_row_normalized.png")
    if args.show:
        plt.show()
    else:
        plt.close()

    # Plot per-class recall
    plt.figure(figsize=(8, 4))
    plt.bar(range(NUM_CLASSES), recall)
    plt.ylim(0, 1.0)
    plt.xticks(range(NUM_CLASSES))
    plt.title("Per-class recall")
    plt.xlabel("class")
    plt.ylabel("recall")
    plt.tight_layout()
    maybe_savefig("per_class_recall.png")
    if args.show:
        plt.show()
    else:
        plt.close()

    # Plot margin histogram
    plt.figure(figsize=(8, 4))
    plt.hist(np.array(margins, dtype=np.float64), bins=60)
    plt.title("Prediction margin histogram (best - second best similarity)")
    plt.xlabel("margin")
    plt.ylabel("count")
    plt.tight_layout()
    maybe_savefig("margin_hist.png")
    if args.show:
        plt.show()
    else:
        plt.close()

    # ----- Save representative error examples -----
    if len(error_cases) > 0 and args.save_examples > 0:
        # pick some "hard" errors (small margin) + some "confident wrong" (large margin)
        error_cases_sorted = sorted(error_cases, key=lambda e: e.margin)
        n = min(args.save_examples, len(error_cases_sorted))

        # Mix: half smallest margins, half largest margins
        half = max(1, n // 2)
        chosen = error_cases_sorted[:half] + error_cases_sorted[-(n - half):]

        for j, ec in enumerate(chosen):
            fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))

            axes[0].imshow(ec.img_np, cmap="gray")
            axes[0].set_title(f"idx={ec.idx}\nT={ec.true_label} P={ec.pred_label}\nmargin={ec.margin:.4f}")
            axes[0].axis("off")

            axes[1].imshow(ec.amp_bin, cmap="inferno")
            axes[1].set_title(f"Amp bins (0..{AMP_LEVELS-1})")
            axes[1].axis("off")

            axes[2].imshow(ec.phase_bin, cmap="twilight")
            axes[2].set_title(f"Phase bins (0..{PHASE_LEVELS-1})")
            axes[2].axis("off")

            axes[3].bar(range(NUM_CLASSES), ec.sims_full)
            axes[3].set_xticks(range(NUM_CLASSES))
            axes[3].set_title("Similarity")

            # annotate sentinel if available
            if ec.true_sentinel_hit is not None and ec.pred_sentinel_hit is not None:
                axes[3].text(
                    0.02, 0.98,
                    f"sentinel T: {ec.true_sentinel_hit}/{ec.true_sentinel_total}\n"
                    f"sentinel P: {ec.pred_sentinel_hit}/{ec.pred_sentinel_total}",
                    transform=axes[3].transAxes,
                    va="top", ha="left", fontsize=9
                )

            plt.tight_layout()

            if args.save_plots:
                outpath = os.path.join(plots_dir, f"error_example_{j+1}_idx{ec.idx}_T{ec.true_label}_P{ec.pred_label}.png")
                plt.savefig(outpath, dpi=200)
                print(f"[ok] Saved error example: {outpath}")

            if args.show:
                plt.show()
            else:
                plt.close()

    # ----- Print top pairs to console -----
    print("\n=== Top confusion pairs (true -> pred) ===")
    for cnt, t, p in top_pairs:
        print(f"{t} -> {p}: {cnt}")
    print("=========================================\n")

    # ----- Sentinel aggregate report (optional) -----
    if has_sentinel and len(error_cases) > 0:
        # Aggregate: for each confusion pair, average sentinel ratio for true vs pred
        pair_stats = {}  # (t,p) -> dict
        for ec in error_cases:
            if ec.true_sentinel_hit is None or ec.pred_sentinel_hit is None:
                continue
            key = (ec.true_label, ec.pred_label)
            if key not in pair_stats:
                pair_stats[key] = {
                    "n": 0,
                    "true_hit": 0,
                    "true_total": 0,
                    "pred_hit": 0,
                    "pred_total": 0
                }
            ps = pair_stats[key]
            ps["n"] += 1
            ps["true_hit"] += ec.true_sentinel_hit
            ps["true_total"] += ec.true_sentinel_total
            ps["pred_hit"] += ec.pred_sentinel_hit
            ps["pred_total"] += ec.pred_sentinel_total

        # Save as json
        sentinel_report = []
        for (t, p), ps in pair_stats.items():
            sentinel_report.append({
                "true": t,
                "pred": p,
                "n": int(ps["n"]),
                "true_sentinel_ratio": safe_div(ps["true_hit"], ps["true_total"]),
                "pred_sentinel_ratio": safe_div(ps["pred_hit"], ps["pred_total"]),
            })
        sentinel_report.sort(key=lambda x: x["n"], reverse=True)

        with open(os.path.join(args.out_dir, "sentinel_error_report.json"), "w", encoding="utf-8") as f:
            json.dump(sentinel_report, f, indent=2)
        print(f"[ok] saved: {args.out_dir}/sentinel_error_report.json (pairs with sentinel stats)")

    print("[done] misclassification analysis complete.")


if __name__ == "__main__":
    main()
