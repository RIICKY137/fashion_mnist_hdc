import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def entropy_from_count(ones: np.ndarray, C: int, eps: float = 1e-12) -> np.ndarray:
    """
    Binary entropy H(p) with p = ones/C. Range: [0, 1] bits if using log2.
    """
    p = ones.astype(np.float32) / float(C)
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def load_class_hv(path: str) -> np.ndarray:
    hv = np.load(path)
    if hv.ndim != 2:
        raise ValueError(f"class_hv must be 2D (C,D), got shape {hv.shape}")
    return hv.astype(np.uint8)


def load_keep_dims(path: str, D: int) -> np.ndarray:
    keep = np.load(path)
    if keep.ndim != 1:
        raise ValueError("keep_dims must be a 1D array of indices.")
    if keep.size == 0:
        raise ValueError("keep_dims is empty.")
    if int(keep.min()) < 0 or int(keep.max()) >= D:
        raise ValueError("keep_dims contains out-of-range indices.")
    return keep.astype(np.int32)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(
        description="Analyze class_hv columns: ones-count, entropy, sentinel columns, and produce plots + npz outputs."
    )
    ap.add_argument("--class_hv", default="data/class_hv_unified.npy",
                    help="Path to class_hv .npy (shape: C x D).")
    ap.add_argument("--use_keep_dims", action="store_true",
                    help="If set, apply keep_dims first (e.g., constant-only pruning) to analyze kept-space.")
    ap.add_argument("--keep_dims", default="data/keep_dims_v3.npy",
                    help="Path to keep_dims .npy (1D indices). Used if --use_keep_dims.")
    ap.add_argument("--out_dir", default="analysis",
                    help="Output directory for npz + plots.")
    ap.add_argument("--topk_entropy", type=int, default=512,
                    help="How many highest-entropy dims to export as a list.")
    ap.add_argument("--window", type=int, default=32,
                    help="Window width for 'best windows' search/plot.")
    ap.add_argument("--top_windows", type=int, default=6,
                    help="How many top windows to export/plot.")
    ap.add_argument("--min_sentinel_per_class", type=int, default=0,
                    help="Warn if any class has fewer than this many sentinels.")
    ap.add_argument("--save_plots", action="store_true",
                    help="If set, save plots as PNG into out_dir/plots.")
    ap.add_argument("--show", action="store_true",
                    help="If set, show plots interactively.")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    plots_dir = os.path.join(args.out_dir, "plots")
    if args.save_plots:
        ensure_dir(plots_dir)

    class_hv_full = load_class_hv(args.class_hv)
    C, D = class_hv_full.shape

    space = "full"
    keep_dims_fullspace = None
    class_hv = class_hv_full
    if args.use_keep_dims:
        keep_dims_fullspace = load_keep_dims(args.keep_dims, D)
        class_hv = class_hv_full[:, keep_dims_fullspace]
        space = "kept"
    C2, D2 = class_hv.shape
    assert C2 == C

    print(f"[info] Loaded class_hv: C={C}, D_full={D}")
    print(f"[info] Analysis space: {space}, D_space={D2}")

    # ---------------- Column statistics ----------------
    ones_count = class_hv.sum(axis=0).astype(np.int32)  # shape: (D2,)
    entropy = entropy_from_count(ones_count, C=C)       # shape: (D2,)

    # constant columns (in current space)
    const_mask = (ones_count == 0) | (ones_count == C)
    n_const = int(const_mask.sum())

    # sentinel columns: exactly one class differs
    # Type A: ones_count == 1  -> only one class has 1
    # Type B: ones_count == C-1 -> only one class has 0 (equivalently one class differs)
    sentinel_pos_mask = (ones_count == 1)
    sentinel_neg_mask = (ones_count == C - 1)
    sentinel_mask = sentinel_pos_mask | sentinel_neg_mask
    n_sentinel = int(sentinel_mask.sum())

    # For each sentinel column, determine which class it points to:
    # - if ones_count==1: the class with bit=1 is the "owner"
    # - if ones_count==C-1: the class with bit=0 is the "owner" (anti-sentinel)
    sentinel_owner = -np.ones(D2, dtype=np.int32)    # -1 means not sentinel
    sentinel_sign = np.zeros(D2, dtype=np.int8)      # +1 for "only this class has 1", -1 for "only this class has 0"
    # compute owners efficiently:
    if n_sentinel > 0:
        # for ones_count==1
        idx_pos = np.where(sentinel_pos_mask)[0]
        if idx_pos.size > 0:
            # argmax gives index of 1 since only one 1 exists
            owners_pos = np.argmax(class_hv[:, idx_pos], axis=0).astype(np.int32)
            sentinel_owner[idx_pos] = owners_pos
            sentinel_sign[idx_pos] = 1

        # for ones_count==C-1
        idx_neg = np.where(sentinel_neg_mask)[0]
        if idx_neg.size > 0:
            # argmin gives index of 0 since only one 0 exists
            owners_neg = np.argmin(class_hv[:, idx_neg], axis=0).astype(np.int32)
            sentinel_owner[idx_neg] = owners_neg
            sentinel_sign[idx_neg] = -1

    # group sentinel dims per class
    sentinel_dims_by_class = {}
    sentinel_pos_by_class = {}
    sentinel_neg_by_class = {}
    for c in range(C):
        dims_c = np.where(sentinel_owner == c)[0]
        sentinel_dims_by_class[str(c)] = dims_c.astype(np.int32)

        dims_pos = np.where((sentinel_owner == c) & (sentinel_sign == 1))[0]
        dims_neg = np.where((sentinel_owner == c) & (sentinel_sign == -1))[0]
        sentinel_pos_by_class[str(c)] = dims_pos.astype(np.int32)
        sentinel_neg_by_class[str(c)] = dims_neg.astype(np.int32)

    # Optional warning: too few sentinels
    if args.min_sentinel_per_class > 0:
        for c in range(C):
            k = sentinel_dims_by_class[str(c)].size
            if k < args.min_sentinel_per_class:
                print(f"[warn] class {c} has only {k} sentinel dims (< {args.min_sentinel_per_class})")

    # ---------------- Export dimension lists ----------------
    # top entropy dims (most "balanced" columns)
    topk = min(args.topk_entropy, D2)
    top_entropy_dims = np.argsort(-entropy)[:topk].astype(np.int32)

    # columns by ones_count buckets (useful for coarse grouping)
    buckets = {}
    for s in range(C + 1):
        buckets[f"ones_{s}"] = np.where(ones_count == s)[0].astype(np.int32)

    # Map "space dim index" -> "full dim index" (if keep_dims used)
    if keep_dims_fullspace is None:
        space_to_full = np.arange(D2, dtype=np.int32)
    else:
        space_to_full = keep_dims_fullspace.astype(np.int32)

    # save analysis package
    out_npz = os.path.join(args.out_dir, f"classhv_features_{space}.npz")
    np.savez(
        out_npz,
        C=C,
        D_space=D2,
        D_full=D,
        space=space,
        ones_count=ones_count,
        entropy=entropy,
        const_dims=np.where(const_mask)[0].astype(np.int32),
        sentinel_dims=np.where(sentinel_mask)[0].astype(np.int32),
        sentinel_owner=sentinel_owner,
        sentinel_sign=sentinel_sign,
        top_entropy_dims=top_entropy_dims,
        space_to_full=space_to_full,
    )
    print(f"[ok] Saved features npz: {out_npz}")

    # save sentinel per class as separate npz for easy consumption
    out_sentinel = os.path.join(args.out_dir, f"sentinel_dims_{space}.npz")
    np.savez(out_sentinel, **{f"class_{k}": v for k, v in sentinel_dims_by_class.items()},
             **{f"class_{k}_pos": v for k, v in sentinel_pos_by_class.items()},
             **{f"class_{k}_neg": v for k, v in sentinel_neg_by_class.items()})
    print(f"[ok] Saved sentinel dims npz: {out_sentinel}")

    # ---------------- Print summary ----------------
    print("\n=== Summary ===")
    print(f"space: {space}")
    print(f"C={C}, D_space={D2}, D_full={D}")
    print(f"constant columns in this space: {n_const} ({n_const / D2:.3f})")
    print(f"sentinel columns (ones==1 or ones==C-1): {n_sentinel} ({n_sentinel / D2:.3f})")
    print("sentinel per class:")
    for c in range(C):
        k = sentinel_dims_by_class[str(c)].size
        kp = sentinel_pos_by_class[str(c)].size
        kn = sentinel_neg_by_class[str(c)].size
        print(f"  class {c}: {k}  (pos:{kp}, neg:{kn})")
    print("===============\n")

    # ---------------- Plots ----------------
    def maybe_savefig(name: str):
        if args.save_plots:
            path = os.path.join(plots_dir, name)
            plt.savefig(path, dpi=200)
            print(f"[ok] Saved plot: {path}")

    # Plot 1: ones_count histogram
    plt.figure(figsize=(8, 4))
    plt.hist(ones_count, bins=np.arange(-0.5, C + 1.5, 1), rwidth=0.9)
    plt.xticks(range(C + 1))
    plt.xlabel("ones_count (sum over classes)")
    plt.ylabel("num dimensions")
    plt.title(f"ones_count distribution | space={space} | D={D2}")
    plt.tight_layout()
    maybe_savefig(f"ones_count_hist_{space}.png")
    if args.show:
        plt.show()
    else:
        plt.close()

    # Plot 2: entropy histogram
    plt.figure(figsize=(8, 4))
    plt.hist(entropy, bins=50)
    plt.xlabel("binary entropy H(p)")
    plt.ylabel("num dimensions")
    plt.title(f"entropy distribution | space={space} | D={D2}")
    plt.tight_layout()
    maybe_savefig(f"entropy_hist_{space}.png")
    if args.show:
        plt.show()
    else:
        plt.close()

    # Plot 3: sentinel owners bar chart
    sentinel_owner_valid = sentinel_owner[sentinel_owner >= 0]
    if sentinel_owner_valid.size > 0:
        counts = np.bincount(sentinel_owner_valid, minlength=C)
        plt.figure(figsize=(8, 4))
        plt.bar(range(C), counts)
        plt.xticks(range(C))
        plt.xlabel("class")
        plt.ylabel("#sentinel dims owned")
        plt.title(f"sentinel count per class | space={space}")
        plt.tight_layout()
        maybe_savefig(f"sentinel_per_class_{space}.png")
        if args.show:
            plt.show()
        else:
            plt.close()

    # Plot 4: find and plot top windows by "structure score"
    # score = (#sentinel in window) + 0.5*(mean entropy in window)
    W = int(args.window)
    if W > 0 and D2 >= W:
        sentinel_float = sentinel_mask.astype(np.float32)
        # prefix sums for O(D) window sums
        ps_sentinel = np.concatenate([[0.0], np.cumsum(sentinel_float)])
        ps_entropy = np.concatenate([[0.0], np.cumsum(entropy.astype(np.float32))])

        scores = np.zeros(D2 - W + 1, dtype=np.float32)
        for i in range(D2 - W + 1):
            sent_sum = ps_sentinel[i + W] - ps_sentinel[i]
            ent_mean = (ps_entropy[i + W] - ps_entropy[i]) / float(W)
            scores[i] = sent_sum + 0.5 * ent_mean

        topw = min(args.top_windows, scores.size)
        top_idx = np.argsort(-scores)[:topw]

        # export windows
        out_windows = os.path.join(args.out_dir, f"top_windows_{space}.npz")
        np.savez(out_windows,
                 window=W,
                 starts=top_idx.astype(np.int32),
                 scores=scores[top_idx])
        print(f"[ok] Saved top windows npz: {out_windows}")

        # plot each window heatmap (no cell text, just structure)
        for rank, start in enumerate(top_idx):
            sub = class_hv[:, start:start + W]
            plt.figure(figsize=(max(8, W / 3), 3.5))
            plt.imshow(sub, aspect="auto", interpolation="nearest", cmap="bwr", vmin=0, vmax=1)
            plt.yticks(range(C), [str(i) for i in range(C)])
            plt.xlabel(f"dim index in {space} space")
            plt.ylabel("class")
            plt.title(f"Top window #{rank+1} | start={start} | W={W} | score={scores[start]:.3f}")
            plt.tight_layout()
            maybe_savefig(f"top_window_{space}_{rank+1}_start{start}_W{W}.png")
            if args.show:
                plt.show()
            else:
                plt.close()

    print("[done] Analysis complete.")


if __name__ == "__main__":
    main()
