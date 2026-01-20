import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def load_class_hv(class_path: str) -> np.ndarray:
    class_hv = np.load(class_path)
    if class_hv.ndim != 2:
        raise ValueError(f"class_hv must be 2D (C,D), got shape {class_hv.shape}")
    return class_hv.astype(np.uint8)


def load_keep_dims(keep_dims_path: str | None, D: int) -> np.ndarray | None:
    if keep_dims_path is None:
        return None
    keep_dims = np.load(keep_dims_path)
    if keep_dims.ndim != 1:
        raise ValueError("keep_dims must be a 1D array of indices.")
    if keep_dims.size == 0:
        raise ValueError("keep_dims is empty.")
    if int(keep_dims.min()) < 0 or int(keep_dims.max()) >= D:
        raise ValueError("keep_dims contains out-of-range indices.")
    return keep_dims.astype(np.int32)


def main():
    p = argparse.ArgumentParser(
        description="Visualize class prototype hypervectors (class_hv) in a dimension range [start:end)."
    )
    p.add_argument("--class_hv", default="data/class_hv_unified.npy",
                   help="Path to class_hv .npy (shape: 10 x DIM).")
    p.add_argument("--start", type=int, default=0,
                   help="Start dimension (inclusive) in the chosen space.")
    p.add_argument("--end", type=int, default=256,
                   help="End dimension (exclusive) in the chosen space.")
    p.add_argument("--use_keep_dims", action="store_true",
                   help="If set, apply keep_dims to drop constant dims (or any pruning index list).")
    p.add_argument("--keep_dims", default="data/keep_dims_v3.npy",
                   help="Path to keep_dims .npy (1D indices). Only used when --use_keep_dims is set.")
    p.add_argument("--title", default="class prototypes",
                   help="Plot title.")
    p.add_argument("--save", default="",
                   help="If provided, save figure to this path (e.g., plots/proto_0_256.png).")
    p.add_argument("--show", action="store_true",
                   help="If set, show interactive window.")
    p.add_argument("--cell_text", action="store_true",
                   help="If set, draw 0/1 text in each cell (use for small ranges like <= 64 columns).")
    p.add_argument("--fig_w", type=float, default=14.0,
                   help="Figure width.")
    p.add_argument("--fig_h", type=float, default=4.0,
                   help="Figure height.")

    args = p.parse_args()

    class_hv = load_class_hv(args.class_hv)
    C, D = class_hv.shape

    keep_dims = None
    space_name = "full"
    if args.use_keep_dims:
        keep_dims = load_keep_dims(args.keep_dims, D)
        class_hv = class_hv[:, keep_dims]
        space_name = "kept"
    D2 = class_hv.shape[1]

    start = max(0, args.start)
    end = min(args.end, D2)
    if end <= start:
        raise ValueError(f"Invalid range: start={start}, end={end}, D={D2}")

    sub = class_hv[:, start:end]  # (10, N)
    N = sub.shape[1]

    plt.figure(figsize=(args.fig_w, args.fig_h))
    # binary heatmap: 0->blue, 1->red (using bwr). You can change cmap if you want.
    plt.imshow(sub, aspect="auto", interpolation="nearest", cmap="bwr", vmin=0, vmax=1)

    plt.yticks(range(C), [str(i) for i in range(C)])
    plt.xticks(range(0, N, max(1, N // 16)), [str(start + i) for i in range(0, N, max(1, N // 16))])

    plt.xlabel(f"Dimension index in {space_name} space")
    plt.ylabel("Class")
    plt.title(f"{args.title}  |  {space_name} dims  |  range [{start}:{end})  (N={N})")

    plt.colorbar(label="bit")

    if args.cell_text:
        if N > 128:
            print("[warn] cell_text on a large range will be unreadable. Consider end-start <= 64.")
        for r in range(C):
            for c in range(N):
                plt.text(c, r, str(int(sub[r, c])),
                         ha="center", va="center", fontsize=7, color="white")

    plt.tight_layout()

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        plt.savefig(args.save, dpi=200)
        print(f"Saved to: {args.save}")

    if args.show:
        plt.show()
    else:
        # avoid hanging in non-interactive environments
        plt.close()


if __name__ == "__main__":
    main()
