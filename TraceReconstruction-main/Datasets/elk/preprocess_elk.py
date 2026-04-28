# Preprocess ELK raw system call text data into TraceReconstruction format.
# Input:  elk_syscalls.txt (extracted with babeltrace2, see README.md)
# Output: TraceReconstruction-main/Datasets/elk/sequence_length_{N}/
#             train.npy, test.npy, training, testing
# Unlike other datasets, ELK includes segments with injected noise.

import os
import numpy as np
from collections import Counter

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SYSCALLS_TXT = os.path.join(SCRIPT_DIR, "elk_syscalls.txt")
OUT_BASE     = SCRIPT_DIR

MAX_TRAIN   = 10000   # cap; actual count may be lower if data is insufficient
MAX_TEST    = 500
SEQ_LENGTHS = [50, 100, 150, 200]
STRIDE_FRAC = 0.5     # sliding window stride = seq_len * STRIDE_FRAC

def load_syscalls(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def sliding_windows(ids, seq_len, stride):
    """Return an (N, seq_len) array of overlapping windows."""
    n = len(ids)
    if n < seq_len:
        return np.empty((0, seq_len), dtype=np.int32)
    starts = range(0, n - seq_len + 1, stride)
    return np.stack([ids[s:s + seq_len] for s in starts])

print("Loading ELK syscall sequence...")
all_events = load_syscalls(SYSCALLS_TXT)
total = len(all_events)
print(f"Total events: {total:,}")

# 80/20 split for train/test pools
split_idx  = int(total * 0.8)
train_pool = all_events[:split_idx]
test_pool  = all_events[split_idx:]

print("Building vocabulary from training data...")
freq = Counter(train_pool)
sorted_calls = sorted(freq.keys(), key=lambda c: (-freq[c], c))
syscall_to_id = {sc: idx + 1 for idx, sc in enumerate(sorted_calls)}
vocab_size = len(syscall_to_id)
print(f"Vocabulary size: {vocab_size}")

train_ids = np.array([syscall_to_id.get(e, vocab_size + 1) for e in train_pool], dtype=np.int32)
test_ids  = np.array([syscall_to_id.get(e, vocab_size + 1) for e in test_pool],  dtype=np.int32)

for seq_len in SEQ_LENGTHS:
    out_dir = os.path.join(OUT_BASE, f"sequence_length_{seq_len}")
    os.makedirs(out_dir, exist_ok=True)

    stride = max(1, int(seq_len * STRIDE_FRAC))

    train_all = sliding_windows(train_ids, seq_len, stride)
    test_all  = sliding_windows(test_ids,  seq_len, stride)

    train_n = min(MAX_TRAIN, len(train_all))
    test_n  = min(MAX_TEST,  len(test_all))

    train_flat = train_all[:train_n]
    test_flat  = test_all[:test_n]

    np.save(os.path.join(out_dir, "train.npy"), train_flat[:, :, np.newaxis])
    np.save(os.path.join(out_dir, "test.npy"),  test_flat[:, :,  np.newaxis])

    with open(os.path.join(out_dir, "training"), "w") as fout:
        for row in train_flat:
            fout.write(",".join(map(str, row)) + "\n")
    with open(os.path.join(out_dir, "testing"), "w") as fout:
        for row in test_flat:
            fout.write(",".join(map(str, row)) + "\n")

    print(f"  seq_len={seq_len}: train{train_flat.shape}  test{test_flat.shape}")

print("Done.")
