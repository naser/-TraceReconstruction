# Preprocess Alibaba Microservice Trace data into TraceReconstruction format.
#
# Source: Luo et al., "Characterizing Microservice Dependency and Performance:
#         Alibaba Trace Analysis", SoCC 2021.
#         https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2021
#
# Input:  One or more MSCallGraph_*.csv files in alibaba-trace/raw/
#         (MSCallGraph_0.csv was downloaded; contains ~6M call records)
#
# The "call type" for each event is a compound key  "<rpctype>:<dm>"
# (communication paradigm + downstream microservice), which is always present
# and gives the closest analogue to a system-call name.  Calls are sorted by
# timestamp within each trace, then by rpcid depth to break ties, producing an
# ordered call sequence for every traceid.
#
# Split strategy: TRACE-LEVEL RANDOM 80/20 SPLIT (seed 42).
# All trace IDs are shuffled randomly; the first 80% of traces form the train
# pool and the remaining 20% form the test pool.  No individual trace is split
# between pools, which prevents the temporal-concentration artefact that occurs
# with a sequential event-position split (end-of-day traffic in a 12-hour
# trace skews the test pool toward a small set of dominant call types).
#
# Output: alibaba-trace/  (raw data stays here)
#         TraceReconstruction-main/Datasets/alibaba/
#             alibaba_calls.txt               (all call events, one per line)
#             sequence_length_{N}/
#                 train.npy, test.npy         (N x seq_len x 1, int32)
#                 training, testing           (comma-separated integer rows)

import csv
import os
import glob
import numpy as np
from collections import Counter, defaultdict

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
WORKSPACE   = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "alibaba-trace", "raw"))
OUT_BASE    = SCRIPT_DIR
CALLS_TXT   = os.path.join(OUT_BASE, "alibaba_calls.txt")

# ── Hyper-parameters ─────────────────────────────────────────────────────────
MAX_TRAIN   = 10_000
MAX_TEST    = 500
SEQ_LENGTHS = [50, 100, 150, 200]
STRIDE_FRAC = 0.5   # sliding window stride = seq_len × STRIDE_FRAC

# ────────────────────────────────────────────────────────────────────────────

def rpcid_depth(rpcid: str) -> tuple:
    """Convert dotted rpcid '0.1.2.3' to a sortable tuple of ints."""
    try:
        return tuple(int(x) for x in rpcid.split("."))
    except ValueError:
        return (0,)


def load_call_sequences(csv_paths: list) -> tuple:
    """
    Read MSCallGraph CSV files and return (traces_dict, trace_order).

    traces_dict  : {traceid: [(timestamp, rpcid_depth_tuple, token), ...]}
    trace_order  : list of traceids in CSV arrival order (for reproducibility)

    Calls are NOT flattened here; the caller performs the train/test split at
    trace granularity before flattening.
    """
    traces: dict = defaultdict(list)
    trace_order: list = []

    total_rows = 0
    for csv_path in csv_paths:
        print(f"  Reading {os.path.basename(csv_path)} …")
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                tid  = row["traceid"]
                ts   = int(row["timestamp"]) if row["timestamp"].strip() else 0
                rid  = rpcid_depth(row.get("rpcid", "0"))
                dm   = row["dm"].strip()
                rt   = row["rpctype"].strip() or "rpc"

                if not dm:          # skip rows with missing downstream service
                    continue

                token = f"{rt}:{dm}"  # e.g. "rpc:abc123..." or "mc:def456..."

                if tid not in traces:
                    trace_order.append(tid)
                traces[tid].append((ts, rid, token))
                total_rows += 1

    print(f"  Loaded {total_rows:,} valid call records from {len(trace_order):,} traces")
    return traces, trace_order


def flatten_traces(tid_list: list, traces_dict: dict) -> list:
    """Sort calls within each trace and flatten the list of traces to tokens."""
    tokens: list = []
    for tid in tid_list:
        calls = sorted(traces_dict[tid], key=lambda x: (x[0], x[1]))
        tokens.extend(c[2] for c in calls)
    return tokens


def sliding_windows(ids: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    n = len(ids)
    if n < seq_len:
        return np.empty((0, seq_len), dtype=np.int32)
    starts = range(0, n - seq_len + 1, stride)
    return np.stack([ids[s:s + seq_len] for s in starts])


# ── Main ─────────────────────────────────────────────────────────────────────

print("Locating MSCallGraph CSV files …")
csv_files = sorted(glob.glob(os.path.join(WORKSPACE, "MSCallGraph_*.csv")))
if not csv_files:
    raise FileNotFoundError(
        f"No MSCallGraph_*.csv files found in:\n  {WORKSPACE}\n"
        "Please download at least MSCallGraph_0.tar.gz from:\n"
        "  http://aliopentrace.oss-cn-beijing.aliyuncs.com/"
        "v2021MicroservicesTraces/MSCallGraph/MSCallGraph_0.tar.gz\n"
        "and extract it into that folder."
    )
print(f"Found {len(csv_files)} file(s): {[os.path.basename(f) for f in csv_files]}")

print("\nLoading call sequences …")
traces, trace_order = load_call_sequences(csv_files)
print(f"Total traces: {len(trace_order):,}")

# ── Random trace-level 80/20 split (seed 42 for reproducibility) ─────────────
# Each trace stays entirely in one pool; this prevents temporal-concentration
# artefacts that arise when a sequential event-position boundary lands in a
# temporally skewed region of the trace (e.g. end-of-day high-frequency calls).
rng = np.random.default_rng(42)
shuffled_order = list(trace_order)
rng.shuffle(shuffled_order)
n_train_traces = int(len(shuffled_order) * 0.8)
train_tids = shuffled_order[:n_train_traces]
test_tids  = shuffled_order[n_train_traces:]
print(f"Split: {n_train_traces:,} train traces / {len(test_tids):,} test traces")

train_pool = flatten_traces(train_tids, traces)
test_pool  = flatten_traces(test_tids,  traces)
total = len(train_pool) + len(test_pool)
print(f"Total call events: {total:,}  (train {len(train_pool):,} / test {len(test_pool):,})")

# Write full call sequence for reference / reproducibility
all_events = flatten_traces(trace_order, traces)
print(f"\nWriting {CALLS_TXT} …")
with open(CALLS_TXT, "w") as fout:
    for token in all_events:
        fout.write(token + "\n")

print("\nBuilding vocabulary from training data …")
freq = Counter(train_pool)
sorted_calls = sorted(freq.keys(), key=lambda c: (-freq[c], c))
call_to_id   = {c: idx + 1 for idx, c in enumerate(sorted_calls)}
vocab_size   = len(call_to_id)
print(f"Vocabulary size: {vocab_size} distinct call types")

# OOV id = vocab_size + 1 (for test tokens not seen in training)
train_ids = np.array([call_to_id[e]                    for e in train_pool], dtype=np.int32)
test_ids  = np.array([call_to_id.get(e, vocab_size + 1) for e in test_pool],  dtype=np.int32)

print()
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

print("\nDone — Alibaba trace dataset created successfully.")
