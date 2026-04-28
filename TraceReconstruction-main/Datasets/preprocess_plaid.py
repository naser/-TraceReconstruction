# Preprocess PLAID raw system call data into TraceReconstruction dataset format.
# Input:  plaid-dataset/data/PLAID.tar.xz
# Output: TraceReconstruction-main/Datasets/plaid/sequence_length_{N}/
#             train.npy, test.npy, training, testing
# The paper uses PLAID baseline data only (excluding attack traces).
# Preprocessing: extract distinct system calls, sort by frequency in training
# data, assign integer IDs starting at 1.

import tarfile
import tempfile
import shutil
import os
import numpy as np
from collections import Counter

PLAID_TAR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
              "..", "plaid-dataset", "data", "PLAID.tar.xz")
OUT_BASE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plaid")
TRAIN_SIZE = 10000
TEST_SIZE  = 500
SEQ_LENGTHS = [50, 100, 150, 200]

import tempfile, shutil

print("Extracting PLAID archive to temp directory (faster than streaming)...")
tmp_dir = tempfile.mkdtemp()
try:
    with tarfile.open(PLAID_TAR, "r:xz") as tar:
        tar.extractall(tmp_dir)
    print(f"  Extracted to {tmp_dir}")

    baseline_dir = os.path.join(tmp_dir, "PLAID", "baseline")
    all_events = []
    file_count = 0
    for root, dirs, files in os.walk(baseline_dir):
        dirs.sort()
        for fname in sorted(files):
            if fname.endswith(".txt"):
                with open(os.path.join(root, fname), "r") as fh:
                    content = fh.read().strip()
                all_events.extend(content.split())
                file_count += 1
    print(f"  Read {file_count} baseline files")
finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)

total = len(all_events)
print(f"Total events: {total:,}")

# Split into training pool (first 80%) and test pool (last 20%)
split_idx = int(total * 0.8)
train_pool = all_events[:split_idx]
test_pool  = all_events[split_idx:]

# Build frequency-based vocabulary from training data
print("Building vocabulary from training data...")
freq = Counter(train_pool)
# Sort by descending frequency; ties broken alphabetically for reproducibility
sorted_calls = sorted(freq.keys(), key=lambda c: (-freq[c], c))
# Assign IDs starting at 1 (0 is typically reserved/unused)
syscall_to_id = {sc: idx + 1 for idx, sc in enumerate(sorted_calls)}
vocab_size = len(syscall_to_id)
print(f"Vocabulary size: {vocab_size} distinct system calls")

# Encode pools
train_ids = np.array([syscall_to_id.get(e, vocab_size + 1) for e in train_pool], dtype=np.int32)
test_ids  = np.array([syscall_to_id.get(e, vocab_size + 1) for e in test_pool],  dtype=np.int32)

for seq_len in SEQ_LENGTHS:
    out_dir = os.path.join(OUT_BASE, f"sequence_length_{seq_len}")
    os.makedirs(out_dir, exist_ok=True)

    needed_train = TRAIN_SIZE * seq_len
    needed_test  = TEST_SIZE  * seq_len

    if len(train_ids) < needed_train:
        raise ValueError(f"Not enough training events: have {len(train_ids)}, need {needed_train}")
    if len(test_ids) < needed_test:
        raise ValueError(f"Not enough test events: have {len(test_ids)}, need {needed_test}")

    # Slice non-overlapping windows
    train_flat = train_ids[:needed_train].reshape(TRAIN_SIZE, seq_len)
    test_flat  = test_ids[:needed_test].reshape(TEST_SIZE,  seq_len)

    # Shape required by SSSD models: (N, seq_len, 1)
    train_arr = train_flat[:, :, np.newaxis]
    test_arr  = test_flat[:, :,  np.newaxis]

    np.save(os.path.join(out_dir, "train.npy"), train_arr)
    np.save(os.path.join(out_dir, "test.npy"),  test_arr)

    # Also write the CSV text versions (matching existing dataset format)
    with open(os.path.join(out_dir, "training"), "w") as fout:
        for row in train_flat:
            fout.write(",".join(map(str, row)) + "\n")
    with open(os.path.join(out_dir, "testing"), "w") as fout:
        for row in test_flat:
            fout.write(",".join(map(str, row)) + "\n")

    print(f"  seq_len={seq_len}: train{train_arr.shape}  test{test_arr.shape}  -> {out_dir}")

print("Done.")
