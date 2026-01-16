import numpy as np

def compute_keep_dims(class_hv: np.ndarray, *, remove_constant=True, dedup_signatures=True):
    """
    class_hv: shape (C, D), uint8 {0,1}
    returns:
      keep_dims: sorted np.ndarray of kept dimension indices
      info: dict with stats
    """
    assert class_hv.ndim == 2, "class_hv must be (C, D)"
    C, D = class_hv.shape

    keep_mask = np.ones(D, dtype=bool)

    # ---- 1) remove constant columns (all-0 or all-1 across classes) ----
    if remove_constant:
        col_sum = class_hv.sum(axis=0)  # (D,)
        constant = (col_sum == 0) | (col_sum == C)
        keep_mask &= ~constant
    else:
        constant = np.zeros(D, dtype=bool)

    kept_after_constant = np.where(keep_mask)[0]

    # ---- 2) deduplicate identical signatures (columns in class_hv) ----
    # signature = class_hv[:, k] for each k. We keep one representative per unique signature.
    if dedup_signatures:
        sigs = class_hv[:, kept_after_constant].T  # (D_kept, C)
        # np.unique returns unique rows; keep first occurrence indices
        _, first_idx = np.unique(sigs, axis=0, return_index=True)
        reps = kept_after_constant[np.sort(first_idx)]
        keep_dims = np.array(reps, dtype=np.int32)
    else:
        keep_dims = kept_after_constant.astype(np.int32)

    keep_dims.sort()

    info = {
        "C": int(C),
        "D": int(D),
        "removed_constant": int(constant.sum()),
        "kept_after_constant": int(len(kept_after_constant)),
        "kept_final": int(len(keep_dims)),
        "compression_ratio": float(len(keep_dims) / D),
    }
    return keep_dims, info


def apply_prune_to_hv(hv: np.ndarray, keep_dims: np.ndarray) -> np.ndarray:
    """Slice 1D hypervector hv by keep_dims."""
    return hv[keep_dims]


def apply_prune_to_class_hv(class_hv: np.ndarray, keep_dims: np.ndarray) -> np.ndarray:
    """Slice class_hv (C,D) by keep_dims -> (C, D')"""
    return class_hv[:, keep_dims]
