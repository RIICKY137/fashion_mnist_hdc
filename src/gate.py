import os
import json
import numpy as np
from load_fashion import get_fashion_mnist
from hdc import bind, similarity

# ---------------- USER SETTINGS ----------------
CONFUSION_PAIRS = {(0, 6), (2, 4), (4, 6)}   # 你当前关心的走廊
PAIR_KNIFE_K = 512                          # 256/512/1024 都可扫；先固定一个再调
USE_KEEP_DIMS = True                        # 自动加载 data/keep_dims_v3a.npy
MAX_TEST_SAMPLES = None                     # None=全量1万；想快点就填 3000 之类

# 扫描密度（越大越细但越慢）
M_STEPS = 25
C_STEPS = 15

OUT_DIR = "analysis/gate_tuning"
os.makedirs(OUT_DIR, exist_ok=True)

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

print("Loaded config:",
      f"DIM={DIM}, block={(2*CROP)}x{(2*CROP)}, AMP_LEVELS={AMP_LEVELS}, PHASE_LEVELS={PHASE_LEVELS}")
print("Pairs:", sorted([tuple(sorted(p)) for p in CONFUSION_PAIRS]), "PAIR_KNIFE_K=", PAIR_KNIFE_K)

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

# ---------------- Helpers ----------------
def normalize_pair(a, b):
    return (a, b) if a < b else (b, a)

def top2_indices(arr):
    # return (best, second)
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

    S = np.where(class_hv_space[a] != class_hv_space[b])[0]
    if S.size == 0:
        return S.astype(np.int32)

    others = [c for c in range(C) if c != a and c != b]
    p = class_hv_space[others][:, S].mean(axis=0)  # [0,1]
    score = np.abs(p - 0.5)                        # bigger => cleaner structural bit

    kk = min(int(k), int(S.size))
    top = np.argpartition(score, -kk)[-kk:]
    top = top[np.argsort(score[top])[::-1]]
    knife = S[top].astype(np.int32)
    return knife

def pairwise_ma_mb(sample_hv_space, class_hv_space, a, b, dims):
    ma = np.mean(sample_hv_space[dims] == class_hv_space[a, dims])
    mb = np.mean(sample_hv_space[dims] == class_hv_space[b, dims])
    return float(ma), float(mb)

# ---------------- LOAD DATA ----------------
_, test_loader = get_fashion_mnist(batch_size=batch_size)
print("Test set size:", len(test_loader.dataset))

# ---------------- LOAD HYPERVECTORS ----------------
pos_hvs_amp = np.load("data/pos_hvs_amp_unified.npy")
pos_hvs_phase = np.load("data/pos_hvs_phase_unified.npy")
value_hvs_amp = np.load("data/value_hvs_amp_unified.npy")
value_hvs_phase = np.load("data/value_hvs_phase_unified.npy")
class_hv_full = np.load("data/class_hv_unified.npy").astype(np.uint8)

# ---------------- LOAD keep_dims (optional) ----------------
use_pruning = False
keep_dims = None
class_hv = class_hv_full

if USE_KEEP_DIMS:
    try:
        keep_dims = np.load("data/keep_dims_v3a.npy").astype(np.int32)
        if keep_dims.ndim != 1 or keep_dims.size == 0:
            raise ValueError("keep_dims invalid")
        if keep_dims.max() >= class_hv_full.shape[1]:
            raise ValueError("keep_dims out of range")

        class_hv = class_hv_full[:, keep_dims]
        use_pruning = True
        print(f"[prune] using keep_dims: {len(keep_dims)}/{DIM} ({len(keep_dims)/DIM:.3f})")
    except Exception as e:
        print("[prune] disabled:", e)

# ---------------- Precompute knife dims (in current space) ----------------
pair_dims = {}
for (a, b) in CONFUSION_PAIRS:
    aa, bb = normalize_pair(a, b)
    pair_dims[(aa, bb)] = build_pairwise_knife_dims(class_hv, aa, bb, k=PAIR_KNIFE_K)

print("[knife] dims per pair:")
for k in sorted(pair_dims.keys()):
    print(" ", k, "->", len(pair_dims[k]))

# ---------------- PASS 1: collect candidate stats ----------------
records = []  # list of dicts for candidates where top2 is in CONFUSION_PAIRS

baseline_correct = 0
total = 0

pair_stats = {k: {"cand": 0, "base_ok": 0, "base_bad": 0,
                  "winner_ok_if_forced": 0, "winner_bad_if_forced": 0}
              for k in pair_dims.keys()}

for idx, (img, label) in enumerate(test_loader):
    if MAX_TEST_SAMPLES is not None and total >= int(MAX_TEST_SAMPLES):
        break

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
    sample_full = (sample_hv_float >= thr).astype(np.uint8)

    true = int(label.item())

    # baseline sims in the chosen space
    if use_pruning:
        sample = sample_full[keep_dims]
        sims = np.array([similarity(sample, class_hv[c]) for c in range(NUM_CLASSES)], dtype=np.float64)
    else:
        sample = sample_full
        sims = np.array([similarity(sample, class_hv_full[c]) for c in range(NUM_CLASSES)], dtype=np.float64)

    top1, top2 = top2_indices(sims)
    a, b = normalize_pair(top1, top2)

    pred_base = top1
    base_ok = (pred_base == true)
    baseline_correct += int(base_ok)
    total += 1

    # candidate only if top2 pair matches one of our corridors
    if (a, b) not in pair_dims:
        continue

    margin = float(sims[top1] - sims[top2])

    dims = pair_dims[(a, b)]
    ma, mb = pairwise_ma_mb(sample, class_hv if use_pruning else class_hv_full, a, b, dims)
    winner = a if ma >= 0.5 else b
    conf = abs(ma - 0.5)

    # stats (if we forced winner)
    pair_stats[(a, b)]["cand"] += 1
    if base_ok:
        pair_stats[(a, b)]["base_ok"] += 1
    else:
        pair_stats[(a, b)]["base_bad"] += 1

    forced_ok = (winner == true)
    if forced_ok:
        pair_stats[(a, b)]["winner_ok_if_forced"] += 1
    else:
        pair_stats[(a, b)]["winner_bad_if_forced"] += 1

    records.append({
        "i": idx,
        "true": true,
        "top1": int(top1),
        "top2": int(top2),
        "a": int(a),
        "b": int(b),
        "margin": margin,
        "ma": ma,
        "mb": mb,
        "conf": conf,
        "winner": int(winner),
        "base_ok": bool(base_ok),
        "winner_ok": bool(forced_ok),
    })

acc_base = baseline_correct / total
print(f"\nBaseline accuracy (in this script run): {acc_base:.4f}  on N={total}")
print(f"Candidates collected: {len(records)}")

# Save raw records
with open(os.path.join(OUT_DIR, "candidate_records.json"), "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

pair_stats_json = {f"{k[0]}_{k[1]}": v for k, v in pair_stats.items()}
with open(os.path.join(OUT_DIR, "pair_stats_if_forced.json"), "w", encoding="utf-8") as f:
    json.dump(pair_stats_json, f, ensure_ascii=False, indent=2)


# ---------------- Compute data-driven starting thresholds ----------------
margins = np.array([r["margin"] for r in records], dtype=np.float64)
confs = np.array([r["conf"] for r in records], dtype=np.float64)
base_ok_mask = np.array([r["base_ok"] for r in records], dtype=bool)
base_bad_mask = ~base_ok_mask

if base_bad_mask.sum() == 0:
    print("\n[stop] No baseline-wrong candidates among selected pairs. Nothing to tune.")
    raise SystemExit(0)

m_start = float(np.percentile(margins[base_bad_mask], 5))     # "最小可用 margin" 起点
c_start = float(np.percentile(confs[base_bad_mask], 50))      # "最小可用 conf" 起点（中位数）

m_max = float(np.percentile(margins, 90))
c_max = float(np.percentile(confs, 90))

print("\nData-driven start thresholds:")
print(f"  m_start (P5 of margin on base-wrong candidates)  = {m_start:.6f}")
print(f"  c_start (P50 of conf   on base-wrong candidates) = {c_start:.6f}")
print("Scan ranges:")
print(f"  margin in [{m_start:.6f}, {m_max:.6f}]  steps={M_STEPS}")
print(f"  conf   in [{c_start:.6f}, {c_max:.6f}]  steps={C_STEPS}")

margin_grid = np.linspace(m_start, m_max, M_STEPS)
conf_grid = np.linspace(c_start, c_max, C_STEPS)

# ---------------- Offline replay scan ----------------
# Strategy: only flip when:
#   margin <= m_thr AND conf >= c_thr
# AND winner differs from baseline top1
#
# We compute patched accuracy over FULL test set by:
# - start from baseline correct count
# - for each candidate record, adjust based on whether we flip and whether that changes correctness

baseline_correct_count = baseline_correct
N_total = total

# Precompute baseline correctness for each candidate
# and correctness if we set pred to winner
base_ok_arr = np.array([r["base_ok"] for r in records], dtype=bool)
winner_ok_arr = np.array([r["winner_ok"] for r in records], dtype=bool)
winner_arr = np.array([r["winner"] for r in records], dtype=np.int32)
top1_arr = np.array([r["top1"] for r in records], dtype=np.int32)
margin_arr = margins
conf_arr = confs

# flipping is only meaningful when winner != top1
can_flip = (winner_arr != top1_arr)

results = []
best = None

for m_thr in margin_grid:
    margin_mask = (margin_arr <= m_thr)
    for c_thr in conf_grid:
        conf_mask = (conf_arr >= c_thr)
        flip_mask = can_flip & margin_mask & conf_mask

        # baseline correct count adjusted:
        # For each flipped sample:
        #   if baseline was correct and winner wrong => -1
        #   if baseline was wrong and winner correct => +1
        # else 0
        delta = np.sum((~base_ok_arr & winner_ok_arr & flip_mask).astype(np.int32)) \
              - np.sum((base_ok_arr & ~winner_ok_arr & flip_mask).astype(np.int32))

        patched_correct = baseline_correct_count + int(delta)
        patched_acc = patched_correct / N_total
        flips = int(flip_mask.sum())

        row = {
            "m_thr": float(m_thr),
            "c_thr": float(c_thr),
            "acc": float(patched_acc),
            "flips": flips,
            "delta_correct": int(delta),
        }
        results.append(row)

        if best is None or row["acc"] > best["acc"] or (row["acc"] == best["acc"] and row["flips"] < best["flips"]):
            best = row

# Sort results by accuracy desc, then flips asc
results_sorted = sorted(results, key=lambda r: (-r["acc"], r["flips"]))

print("\nTop 10 threshold settings (acc desc, flips asc):")
for r in results_sorted[:10]:
    print(f"  acc={r['acc']:.4f}  flips={r['flips']:4d}  delta={r['delta_correct']:4d} "
          f"  m_thr={r['m_thr']:.6f}  c_thr={r['c_thr']:.6f}")

print("\nBEST setting:")
print(f"  acc={best['acc']:.4f}  flips={best['flips']}  delta={best['delta_correct']} "
      f"  m_thr={best['m_thr']:.6f}  c_thr={best['c_thr']:.6f}")

# Save outputs
with open(os.path.join(OUT_DIR, "scan_results_top.json"), "w", encoding="utf-8") as f:
    json.dump(results_sorted[:200], f, ensure_ascii=False, indent=2)

with open(os.path.join(OUT_DIR, "best_thresholds.json"), "w", encoding="utf-8") as f:
    json.dump({
        "baseline_acc": float(acc_base),
        "N_total": int(N_total),
        "N_candidates": int(len(records)),
        "m_start": m_start,
        "c_start": c_start,
        "m_max": m_max,
        "c_max": c_max,
        "M_STEPS": int(M_STEPS),
        "C_STEPS": int(C_STEPS),
        "best": best,
        "pairs": sorted([list(p) for p in pair_dims.keys()]),
        "PAIR_KNIFE_K": int(PAIR_KNIFE_K),
        "use_pruning": bool(use_pruning),
        "D_eff": int(class_hv.shape[1]) if use_pruning else int(DIM),
    }, f, ensure_ascii=False, indent=2)

print(f"\nSaved:")
print(f"  {OUT_DIR}/candidate_records.json")
print(f"  {OUT_DIR}/pair_stats_if_forced.json")
print(f"  {OUT_DIR}/scan_results_top.json")
print(f"  {OUT_DIR}/best_thresholds.json")
