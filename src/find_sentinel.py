import numpy as np
import os

# ---------------- CONFIG ----------------
CLASS_HV_PATH = "data/class_hv_unified.npy"     # shape: (C, D)
KEEP_DIMS_PATH = "data/keep_dims_v3a.npy"       # optional
OUT_DIR = "analysis"
NUM_CLASSES = 10

USE_KEEP_DIMS = True     # False = full space


# ---------------- LOAD ----------------
class_hv_full = np.load(CLASS_HV_PATH).astype(np.uint8)

if USE_KEEP_DIMS:
    keep = np.load(KEEP_DIMS_PATH).astype(np.int32)
    class_hv = class_hv_full[:, keep]
    print(f"[info] using kept space: {class_hv.shape}")
else:
    class_hv = class_hv_full
    keep = None
    print(f"[info] using full space: {class_hv.shape}")

C, D = class_hv.shape
assert C == NUM_CLASSES


# ---------------- STRICT SENTINEL ----------------
ones_count = class_hv.sum(axis=0)

strict_sentinel = {}
for c in range(C):
    pos = np.where((ones_count == 1) & (class_hv[c] == 1))[0]
    neg = np.where((ones_count == C - 1) & (class_hv[c] == 0))[0]
    strict_sentinel[f"class_{c}"] = np.concatenate([pos, neg]).astype(np.int32)

print("\n=== strict sentinel counts ===")
for c in range(C):
    print(f"{c}: {len(strict_sentinel[f'class_{c}'])}")
print("================================\n")


# ---------------- PAIRWISE SENTINEL ----------------
pairwise_sentinel = {}

for a in range(C):
    for b in range(a + 1, C):
        dims = np.where(class_hv[a] != class_hv[b])[0]
        pairwise_sentinel[f"pair_{a}_{b}"] = dims.astype(np.int32)

print("=== pairwise sentinel counts (examples) ===")
for k in ["pair_2_4", "pair_4_6", "pair_7_9"]:
    if k in pairwise_sentinel:
        print(f"{k}: {len(pairwise_sentinel[k])}")
print("==========================================\n")


# ---------------- SAVE ----------------
os.makedirs(OUT_DIR, exist_ok=True)

np.savez(
    os.path.join(OUT_DIR, "strict_sentinel_from_classhv.npz"),
    **strict_sentinel
)

np.savez(
    os.path.join(OUT_DIR, "pairwise_sentinel_from_classhv.npz"),
    **pairwise_sentinel
)

print("✅ Sentinel construction from class_hv complete.")
