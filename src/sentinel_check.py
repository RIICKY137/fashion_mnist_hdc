import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="analysis/classhv_features_kept.npz",
                    help="NPZ produced by analyze_classhv_features.py (kept space).")
    ap.add_argument("--class_hv", default="data/class_hv_unified.npy",
                    help="Full-space class_hv (10 x DIM).")
    ap.add_argument("--keep_dims", default="data/keep_dims_v3.npy",
                    help="Keep dims used to define the kept-space. MUST match --features.")
    args = ap.parse_args()

    feat = np.load(args.features, allow_pickle=True)
    ones = feat["ones_count"].astype(np.int32)  # shape (D_kept,)
    C = int(feat["C"]) if "C" in feat else 10
    D_kept = int(feat["D"]) if "D" in feat else int(ones.shape[0])

    keep = np.load(args.keep_dims).astype(np.int32)
    class_hv_full = np.load(args.class_hv).astype(np.uint8)

    # Build kept-space class_hv and sanity check dimensions
    class_hv_kept = class_hv_full[:, keep]
    if class_hv_kept.shape[1] != ones.shape[0]:
        raise ValueError(
            f"Space mismatch:\n"
            f"  ones_count length = {ones.shape[0]}\n"
            f"  class_hv_full[:, keep_dims] length = {class_hv_kept.shape[1]}\n"
            f"Fix: make sure --features and --keep_dims come from the SAME run "
            f"(same keep_dims file)."
        )

    print(f"[ok] aligned kept space: C={C}, D_kept={ones.shape[0]}")

    # Strict sentinel: ones_count==1 with class bit=1  OR  ones_count==C-1 with class bit=0
    sentinel = {}
    for c in range(C):
        pos = np.where((ones == 1) & (class_hv_kept[c] == 1))[0]
        neg = np.where((ones == C - 1) & (class_hv_kept[c] == 0))[0]
        dims = np.concatenate([pos, neg]).astype(np.int32)
        sentinel[f"class_{c}"] = dims

    print("\n=== strict sentinel counts (kept space) ===")
    total = 0
    for c in range(C):
        n = int(len(sentinel[f"class_{c}"]))
        total += n
        print(f"{c}: {n}")
    print(f"TOTAL strict sentinel dims (sum over classes, duplicates impossible here): {total}")
    print("=========================================\n")

    # Optional: show distribution of ones_count globally
    vals, cnts = np.unique(ones, return_counts=True)
    print("ones_count distribution (kept space):")
    for v, k in zip(vals.tolist(), cnts.tolist()):
        print(f"  ones={v}: {k}")

    # Save strict sentinel file
    out_path = "analysis/strict_sentinel_dims_kept.npz"
    np.savez(out_path, **sentinel)
    print(f"\nSaved strict sentinel dims to: {out_path}")


if __name__ == "__main__":
    main()
