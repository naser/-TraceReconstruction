"""
preprocess_elk_workloads.py
─────────────────────────────────────────────────────────────────────────────
Split the ELK trace into two workload sub-datasets for the varying-workload
experiment (RQ-W):

  elk_clean  — events drawn from the *pre-noise* phase (normal operation)
  elk_noisy  — events drawn from the *noise-injection* phase (stressed)

ELK experiment timeline (from KernelTracing/README.md):
  0 – 1200 s  : normal workload (light/heavy queries, no injected noise)
  1200 – 1860 s: four injected noise bursts
    CPU   noise: 1200–1320 s
    I/O   noise: 1380–1500 s
    Network noise: 1560–1680 s
    Memory noise: 1740–1860 s
  1860 – end  : post-noise recovery (excluded from both splits)

Because the raw sequence (elk_syscalls.txt) is ordered chronologically but
carries no timestamps in this file, we approximate the phase boundaries
from the time fractions of the total experiment duration:

  total  =  1920 s  (assumed, 60 s post-noise recovery)
  clean  =  1200 s  →  fraction 0.625  →  first 62.5 % of events
  noisy  =  660 s   →  fraction 0.344  →  next  34.4 % of events
  tail   =  60 s    →  fraction 0.031  →  last  3.1 % excluded

Vocabulary policy: a SHARED vocabulary is built from ALL events in
elk_syscalls.txt before the split.  This ensures that event IDs are
identical across elk_clean and elk_noisy, so cross-workload performance
differences reflect *sequence pattern* differences rather than vocabulary
mapping artefacts.

Output directories (created under TraceReconstruction-main/Datasets/):
  elk_clean/sequence_length_{50,100,150,200}/   train.npy  test.npy
  elk_noisy/sequence_length_{50,100,150,200}/   train.npy  test.npy

Array format: int32, shape (N, seq_len, 1) — identical to all other
dataset arrays in the paper.
"""

import os
import sys
import numpy as np
from collections import Counter

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.dirname(SCRIPT_DIR)
ELK_TXT      = os.path.join(REPO_ROOT,
                             "TraceReconstruction-main", "Datasets", "elk",
                             "elk_syscalls.txt")
OUT_BASE     = os.path.join(REPO_ROOT, "TraceReconstruction-main", "Datasets")

# ── Configuration ─────────────────────────────────────────────────────────────
# Temporal fractions derived from the experiment timeline described above.
CLEAN_FRAC   = 0.625   # 0 – 1200 s out of assumed 1920 s total
NOISY_FRAC   = 0.344   # 1200 – 1860 s  (noise injection window)
# The remaining ~3.1 % is excluded (post-noise recovery).

MAX_TRAIN    = 10000
MAX_TEST     = 500
SEQ_LENGTHS  = [50, 100, 150, 200]
STRIDE_FRAC  = 0.5     # matching preprocess_elk.py


def load_syscalls(path: str):
    with open(path, "r") as fh:
        return [line.strip() for line in fh if line.strip()]


def sliding_windows(ids: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    """Return (N, seq_len) matrix of overlapping windows."""
    n = len(ids)
    if n < seq_len:
        return np.empty((0, seq_len), dtype=np.int32)
    starts = range(0, n - seq_len + 1, stride)
    return np.stack([ids[s:s + seq_len] for s in starts])


def save_split(name: str, events_ids: np.ndarray, vocab_size: int,
               oob_id: int) -> None:
    """Save all sequence-length variants for one workload split."""
    total = len(events_ids)
    split_idx = int(total * 0.8)
    train_ids = events_ids[:split_idx]
    test_ids  = events_ids[split_idx:]

    print(f"\n  [{name}]  total={total:,}  "
          f"train_pool={len(train_ids):,}  test_pool={len(test_ids):,}")

    for seq_len in SEQ_LENGTHS:
        out_dir = os.path.join(OUT_BASE, name, f"sequence_length_{seq_len}")
        os.makedirs(out_dir, exist_ok=True)

        stride = max(1, int(seq_len * STRIDE_FRAC))

        train_wins = sliding_windows(train_ids, seq_len, stride)
        test_wins  = sliding_windows(test_ids,  seq_len, stride)

        n_train = min(MAX_TRAIN, len(train_wins))
        n_test  = min(MAX_TEST,  len(test_wins))

        train_arr = train_wins[:n_train]
        test_arr  = test_wins[:n_test]

        np.save(os.path.join(out_dir, "train.npy"),
                train_arr[:, :, np.newaxis])
        np.save(os.path.join(out_dir, "test.npy"),
                test_arr[:, :,  np.newaxis])

        # Plain-text copies (matching other datasets)
        with open(os.path.join(out_dir, "training"), "w") as fout:
            for row in train_arr:
                fout.write(",".join(map(str, row)) + "\n")
        with open(os.path.join(out_dir, "testing"), "w") as fout:
            for row in test_arr:
                fout.write(",".join(map(str, row)) + "\n")

        print(f"    seq_len={seq_len}: "
              f"train{train_arr.shape}  test{test_arr.shape}")


# ── Main ──────────────────────────────────────────────────────────────────────
print("Loading ELK syscall sequence …")
all_events = load_syscalls(ELK_TXT)
total      = len(all_events)
print(f"Total events: {total:,}")

# ── Build shared vocabulary from ALL events ───────────────────────────────────
print("Building shared vocabulary from all events …")
freq          = Counter(all_events)
sorted_calls  = sorted(freq.keys(), key=lambda c: (-freq[c], c))
syscall_to_id = {sc: idx + 1 for idx, sc in enumerate(sorted_calls)}
vocab_size    = len(syscall_to_id)
oob_id        = vocab_size + 1      # out-of-vocabulary sentinel
print(f"Shared vocabulary size: {vocab_size}")

# Encode the full stream once
all_ids = np.array([syscall_to_id.get(e, oob_id) for e in all_events],
                   dtype=np.int32)

# ── Split into clean / noisy / tail ──────────────────────────────────────────
clean_end = int(total * CLEAN_FRAC)
noisy_end = clean_end + int(total * NOISY_FRAC)

clean_ids = all_ids[:clean_end]
noisy_ids = all_ids[clean_end:noisy_end]
tail_ids  = all_ids[noisy_end:]               # excluded from both splits

print(f"\nSplit boundaries (approximate, based on temporal fractions):")
print(f"  Clean  (0 – 1200 s): events [        0 : {clean_end:>6,}]"
      f"  ({len(clean_ids):,} events, {len(clean_ids)/total*100:.1f} %)")
print(f"  Noisy  (1200 – 1860 s): events [{clean_end:>6,} : {noisy_end:>6,}]"
      f"  ({len(noisy_ids):,} events, {len(noisy_ids)/total*100:.1f} %)")
print(f"  Tail   (excl.)       : events [{noisy_end:>6,} : {total:>6,}]"
      f"  ({len(tail_ids):,} events, {len(tail_ids)/total*100:.1f} %)")

# Show vocabulary coverage in each split
clean_calls = set(all_events[:clean_end])
noisy_calls = set(all_events[clean_end:noisy_end])
print(f"\nDistinct syscalls in clean: {len(clean_calls)}")
print(f"Distinct syscalls in noisy: {len(noisy_calls)}")
print(f"Syscalls exclusive to clean: {len(clean_calls - noisy_calls)}")
print(f"Syscalls exclusive to noisy: {len(noisy_calls - clean_calls)}")
print(f"Shared syscalls: {len(clean_calls & noisy_calls)}")

# ── Write datasets ─────────────────────────────────────────────────────────────
save_split("elk_clean", clean_ids, vocab_size, oob_id)
save_split("elk_noisy", noisy_ids, vocab_size, oob_id)

print("\nDone.  Datasets written to:")
print(f"  {os.path.join(OUT_BASE, 'elk_clean')}/")
print(f"  {os.path.join(OUT_BASE, 'elk_noisy')}/")
