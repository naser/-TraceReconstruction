"""
preprocess_pts_shared_vocab.py
──────────────────────────────────────────────────────────────────────────────
Build PTS datasets with a SHARED vocabulary per functional family so that
cross-workload experiments use consistent token IDs.

  Memory-bandwidth family: stream + ramspeed
  CPU-benchmark family:    pybench + phpbench

The bug this fixes:
  Previously each dataset built its own frequency-ranked vocab from its own
  training data.  So ID 10 in stream meant a DIFFERENT syscall than ID 10 in
  ramspeed.  A model trained on stream and tested on ramspeed data was
  comparing incompatible token spaces, making cross-workload "transfer" results
  meaningless.

This script:
  1. Downloads a single all-events run from Zenodo 437170 for each benchmark
     (stream.zip → run0, ramspeed.zip → run0, pybench.zip → run0,
      phpbench.zip → run0).
  2. Uses the pure-Python ctf_reader.py to extract syscall name sequences from
     the LTTng CTF binary traces (no babeltrace dependency).
  3. Merges the two raw name sequences per family and builds ONE shared
     syscall→ID mapping from the combined corpus.
  4. Concatenates run0…run15 (first half) from each benchmark as training data
     and run16…run31 (second half) as a test pool, re-encoding with the shared
     vocab.
  5. Slides windows and saves as train.npy / test.npy in the existing dataset
     directories, replacing the per-dataset-vocab files.
  6. Saves vocab JSON files so the mapping is reproducible.

Usage (run on a login or compute node with internet access):
  python3 varying-workload/preprocess_pts_shared_vocab.py

Or submit via the provided SLURM script:
  sbatch varying-workload/build_pts_shared_vocab.slurm

Outputs written to:
  TraceReconstruction-main/Datasets/stream/   (overwritten with shared-vocab)
  TraceReconstruction-main/Datasets/ramspeed/
  TraceReconstruction-main/Datasets/pybench/
  TraceReconstruction-main/Datasets/phpbench/
  varying-workload/pts_mem_vocab.json          (stream+ramspeed shared vocab)
  varying-workload/pts_cpu_vocab.json          (pybench+phpbench shared vocab)
"""

import io
import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent
CTF_READER  = REPO_ROOT / "ctf_reader.py"
DATASETS    = REPO_ROOT / "TraceReconstruction-main" / "Datasets"
VOCAB_DIR   = SCRIPT_DIR         # save vocab JSONs alongside this script
ZENODO_CACHE = Path(os.environ.get("SCRATCH", REPO_ROOT)) / "zenodo-pts-cache"

# ── Dataset config ─────────────────────────────────────────────────────────────
# output_names are the NEW dataset directory names written under Datasets/.
# These deliberately differ from the canonical single-vocab names (stream,
# ramspeed, …) so the original TraceReconstruction datasets are never overwritten.
FAMILIES = {
    "mem": {
        "datasets":     ["stream",          "ramspeed"],
        "output_names": ["stream_sharedmem", "ramspeed_sharedmem"],
        "zenodo_urls": {
            "stream":   "https://zenodo.org/records/437170/files/stream.zip",
            "ramspeed": "https://zenodo.org/records/437170/files/ramspeed.zip",
        },
        "vocab_file": VOCAB_DIR / "pts_mem_vocab.json",
    },
    "cpu": {
        "datasets":     ["pybench",          "phpbench"],
        "output_names": ["pybench_sharedcpu", "phpbench_sharedcpu"],
        "zenodo_urls": {
            "pybench":  "https://zenodo.org/records/437170/files/pybench.zip",
            "phpbench": "https://zenodo.org/records/437170/files/phpbench.zip",
        },
        "vocab_file": VOCAB_DIR / "pts_cpu_vocab.json",
    },
}

# All 32 runs per benchmark (run0..run31): extract all, then chunk-split
ALL_RUNS = list(range(32))

# Chunk-based train/test split (same method as ELK v2)
CHUNK_SIZE = 500   # events per chunk
TEST_EVERY = 5     # every 5th chunk → test (≈20%, uniformly spread)

# Dataset generation parameters (match paper)
SEQ_LENGTHS = [50, 100, 150, 200]
STRIDE_FRAC = 0.1   # stride = seq_len * 0.1 = 20 for seq_len=200
MAX_TRAIN   = 10_000
MAX_TEST    = 500

# Maximum raw events to extract per dataset before stopping.
# At STRIDE_FRAC=0.1 and seq_len=200 (stride=20) we need at most:
#   train budget: MAX_TRAIN * stride + seq_len = 10000*20+200 = 200_200
#   test  budget: MAX_TEST  * stride + seq_len =   500*20+200 =  10_200
# Multiply by 1.5× safety margin and align to CHUNK_SIZE boundary.
# This prevents OOM on dense benchmarks (phpbench: 9M events/run).
MAX_EVENTS_PER_DS = ((200_200 + 10_200) * 2 // CHUNK_SIZE + 1) * CHUNK_SIZE  # ≈ 422_000

# ── Lazy-import ctf_reader functions ──────────────────────────────────────────
sys.path.insert(0, str(REPO_ROOT))
from ctf_reader import parse_metadata, PKT_HDR_SIZE, PKT_CTX_SIZE, CTF_MAGIC
from ctf_reader import _read_event_header_compact, _read_event_header_large, _skip_fields
import struct as _struct


def _extract_pts_syscalls(channel_path: str, event_map: dict, stream_hdrs: dict,
                           budget: int) -> tuple[list[str], int]:
    """
    Extract syscall names from one CTF channel file of a PTS LTTng trace.
    Stops early when `budget` remaining events reaches 0.

    PTS kernel traces have NO per-event stream context (no _procname/_pid/_tid),
    unlike the ELK trace.  Omitting the 24-byte context skip is the critical
    difference from ctf_reader.extract_events_from_channel.

    Returns (names, remaining_budget).
    """
    names: list[str] = []
    with open(channel_path, "rb") as f:
        raw = bytearray(f.read())

    n = len(raw)
    pos = 0

    while pos + PKT_HDR_SIZE + PKT_CTX_SIZE <= n:
        if budget <= 0:
            break
        if _struct.unpack_from("<I", raw, pos)[0] != CTF_MAGIC:
            nxt = raw.find(_struct.pack("<I", CTF_MAGIC), pos + 1)
            if nxt == -1:
                break
            pos = nxt
            continue

        pkt_start = pos
        pkt_stream_id = _struct.unpack_from("<I", raw, pos + 20)[0]
        pos += PKT_HDR_SIZE

        content_bits = _struct.unpack_from("<Q", raw, pos + 16)[0]
        packet_bits  = _struct.unpack_from("<Q", raw, pos + 24)[0]
        pos += PKT_CTX_SIZE

        if not content_bits:
            pos = pkt_start + (packet_bits >> 3 if packet_bits else 4096)
            continue

        content_end = pkt_start + (content_bits >> 3)
        pkt_end     = pkt_start + (packet_bits >> 3 if packet_bits else content_bits >> 3)
        if content_end > n:
            content_end = n

        hdr_type = stream_hdrs.get(pkt_stream_id, "compact")

        while pos < content_end:
            if pos + 4 > n or budget <= 0:
                break
            try:
                if hdr_type == "large":
                    event_id, pos = _read_event_header_large(raw, pos)
                else:
                    event_id, pos = _read_event_header_compact(raw, pos)
            except Exception:
                break

            # NOTE: no per-event context skip (PTS has none, unlike ELK)

            key = (pkt_stream_id, event_id)
            entry = event_map.get(key)
            if entry is None:
                pos = content_end
                break

            name, fields = entry
            if fields:
                pos = _skip_fields(raw, pos, fields)

            if "syscall" in name:
                clean = (name
                         .replace("syscall_entry_", "")
                         .replace("syscall_exit_",  "")
                         .replace("compat_syscall_entry_", "")
                         .replace("compat_syscall_exit_",  ""))
                names.append(clean)
                budget -= 1

        pos = pkt_end

    return names, budget


# ── Download helper ───────────────────────────────────────────────────────────
def download_zip(url: str, dest: Path) -> Path:
    """Download url to dest/filename if not already present; return path."""
    dest.mkdir(parents=True, exist_ok=True)
    fname = dest / url.split("/")[-1]
    if fname.exists():
        print(f"  [cache] {fname.name}", flush=True)
        return fname
    print(f"  Downloading {fname.name} …", flush=True)
    import urllib.request
    urllib.request.urlretrieve(url, str(fname))
    print(f"  Done: {fname}", flush=True)
    return fname


# ── Syscall extraction from zip ───────────────────────────────────────────────
def _ensure_run_extracted(zip_path: Path, run_idx: int,
                           trace_dirs: dict) -> Path | None:
    """
    Extract run_{run_idx} from zip_path into ZENODO_CACHE/tmp_{stem}_run{N}/.
    Uses a sentinel file (.complete) to validate prior extractions:
      - If .complete exists: directory is intact, reuse it.
      - If directory exists but .complete is missing: prior extraction was
        interrupted; delete and re-extract.
      - If directory does not exist: extract now.
    Returns the kernel/ Path or None on error.
    """
    tmp_dir  = ZENODO_CACHE / f"tmp_{zip_path.stem}_run{run_idx}"
    sentinel = tmp_dir / ".complete"

    if tmp_dir.exists() and not sentinel.exists():
        # Partial or legacy extraction — remove and redo
        import shutil
        print(f"      Removing incomplete tmp dir: {tmp_dir.name}", flush=True)
        shutil.rmtree(tmp_dir)

    if not tmp_dir.exists():
        tmp_dir.mkdir(parents=True, exist_ok=True)
        kernel_members = list(trace_dirs.get(run_idx, set()))
        with zipfile.ZipFile(str(zip_path)) as zf:
            for member in kernel_members:
                try:
                    zf.extract(member, tmp_dir)
                except Exception as e:
                    print(f"      Warning extracting {member}: {e}", flush=True)
        sentinel.touch()  # mark extraction as complete

    # Locate kernel dir
    kernel_dir = None
    for md in tmp_dir.rglob("metadata"):
        kernel_dir = md.parent
        break
    return kernel_dir


def extract_syscalls_from_zip(zip_path: Path, runs: list[int],
                              max_events: int = MAX_EVENTS_PER_DS) -> list[str]:
    """
    Open a Zenodo PTS zip and extract syscall names from the 'all-events'
    kernel traces for the specified run indices.

    Actual zip structure (Zenodo 437170):
      <benchmark>/<benchmark>-all-events-run<N>/kernel/<channel_files>
      <benchmark>/<benchmark>-all-events-run<N>/kernel/metadata

    Stops early once max_events syscalls have been collected, to prevent
    OOM on dense benchmarks (phpbench: ~9M events/run).
    Returns a flat list of syscall name strings (len ≤ max_events).
    """
    # Build member index for requested runs
    trace_dirs: dict[int, set[str]] = {}
    with zipfile.ZipFile(str(zip_path)) as zf:
        for member in zf.infolist():
            parts = Path(member.filename).parts
            if len(parts) >= 3 and "all-events-run" in parts[1] and parts[2] == "kernel":
                run_str = parts[1].split("all-events-run")[-1]
                try:
                    run_idx = int(run_str)
                except ValueError:
                    continue
                if run_idx in runs:
                    trace_dirs.setdefault(run_idx, set()).add(member.filename)

    syscalls: list[str] = []
    budget = max_events

    for run_idx in sorted(trace_dirs):
        if budget <= 0:
            break
        print(f"    run{run_idx}: extracting … (budget={budget:,})", flush=True)

        kernel_dir = _ensure_run_extracted(zip_path, run_idx, trace_dirs)
        if kernel_dir is None or not (kernel_dir / "metadata").exists():
            print(f"      ERROR: metadata not found", flush=True)
            continue

        event_map, stream_hdrs = parse_metadata(str(kernel_dir / "metadata"))
        channel_files = sorted(
            f for f in os.listdir(kernel_dir)
            if os.path.isfile(kernel_dir / f)
            and f != "metadata"
            and not f.endswith(".idx")
            and not f.startswith(".")
        )
        for ch in channel_files:
            if budget <= 0:
                break
            names, budget = _extract_pts_syscalls(
                str(kernel_dir / ch), event_map, stream_hdrs, budget
            )
            syscalls.extend(names)

        print(f"      Total so far: {len(syscalls):,}  (budget remaining: {budget:,})",
              flush=True)

    return syscalls


# ── Sliding window encoder ─────────────────────────────────────────────────────
def sliding_windows(ids: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    n = len(ids)
    if n < seq_len:
        return np.empty((0, seq_len), dtype=np.int32)
    starts = range(0, n - seq_len + 1, stride)
    return np.stack([ids[s:s + seq_len] for s in starts])


def write_dataset(source_name: str, output_name: str,
                  train_ids: np.ndarray, test_ids: np.ndarray,
                  vocab_size: int) -> None:
    """Encode and write train/test npy files for all sequence lengths."""
    out_base = DATASETS / output_name
    print(f"\n  Writing {source_name} → {output_name} …")
    oob = vocab_size + 1  # out-of-vocabulary sentinel

    for seq_len in SEQ_LENGTHS:
        out_dir = out_base / f"sequence_length_{seq_len}"
        out_dir.mkdir(parents=True, exist_ok=True)
        stride = max(1, int(seq_len * STRIDE_FRAC))

        tr_wins = sliding_windows(train_ids, seq_len, stride)
        te_wins = sliding_windows(test_ids,  seq_len, stride)

        n_tr = min(MAX_TRAIN, len(tr_wins))
        n_te = min(MAX_TEST,  len(te_wins))
        tr = tr_wins[:n_tr]
        te = te_wins[:n_te]

        np.save(out_dir / "train.npy", tr[:, :, np.newaxis])
        np.save(out_dir / "test.npy",  te[:, :, np.newaxis])

        with open(out_dir / "training", "w") as f:
            for row in tr:
                f.write(",".join(map(str, row)) + "\n")
        with open(out_dir / "testing", "w") as f:
            for row in te:
                f.write(",".join(map(str, row)) + "\n")

        print(f"    seq_len={seq_len}: train{tr.shape}  test{te.shape}")


# ── Main ──────────────────────────────────────────────────────────────────────
def process_family(family_name: str, cfg: dict) -> None:
    datasets  = cfg["datasets"]
    urls      = cfg["zenodo_urls"]
    vocab_file = cfg["vocab_file"]

    print(f"\n{'='*60}")
    print(f"  Family: {family_name}  ({' + '.join(datasets)})")
    print(f"{'='*60}")

    # ── Step 1: Download zips ──────────────────────────────────────────────
    zip_paths = {}
    for ds in datasets:
        zip_paths[ds] = download_zip(urls[ds], ZENODO_CACHE)

    # ── Step 2: Extract ALL runs combined, then chunk-split ────────────────
    print("\nExtracting syscall sequences (all 32 runs each) …")
    raw_all: dict[str, list[str]] = {}
    for ds in datasets:
        print(f"\n  [{ds}]")
        names = extract_syscalls_from_zip(zip_paths[ds], ALL_RUNS)
        raw_all[ds] = names
        print(f"  {ds}: {len(names):,} total syscalls")

    # Chunk-based train/test split for each dataset (identical to ELK v2)
    def chunk_split(events: list[str]) -> tuple[list[str], list[str]]:
        train, test = [], []
        n_chunks = len(events) // CHUNK_SIZE
        for i in range(n_chunks):
            chunk = events[i * CHUNK_SIZE: (i + 1) * CHUNK_SIZE]
            if (i % TEST_EVERY) == (TEST_EVERY - 1):
                test.extend(chunk)
            else:
                train.extend(chunk)
        train.extend(events[n_chunks * CHUNK_SIZE:])  # remainder → train
        return train, test

    raw: dict[str, dict[str, list[str]]] = {}
    for ds in datasets:
        tr, te = chunk_split(raw_all[ds])
        raw[ds] = {"train": tr, "test": te}
        print(f"  {ds}: train={len(tr):,}  test={len(te):,}")

    # ── Step 3: Build shared vocabulary from combined TRAIN corpus ─────────
    print("\nBuilding shared vocabulary …")
    all_train_names: list[str] = []
    for ds in datasets:
        all_train_names.extend(raw[ds]["train"])

    freq = Counter(all_train_names)
    sorted_calls = sorted(freq.keys(), key=lambda c: (-freq[c], c))
    syscall_to_id = {sc: idx + 1 for idx, sc in enumerate(sorted_calls)}
    vocab_size = len(syscall_to_id)
    oob = vocab_size + 1

    print(f"  Shared vocabulary size: {vocab_size}")
    for ds in datasets:
        ds_calls = set(raw[ds]["train"])
        print(f"  {ds}: {len(ds_calls)} distinct syscalls  "
              f"(coverage {len(ds_calls)/vocab_size*100:.1f}% of shared vocab)")

    # Persist vocab
    vocab_file.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_file, "w") as f:
        json.dump({"family": family_name, "datasets": datasets,
                   "vocab_size": vocab_size, "oob_id": oob,
                   "syscall_to_id": syscall_to_id}, f, indent=2)
    print(f"  Vocab saved → {vocab_file}")

    # ── Step 4: Encode and write datasets ──────────────────────────────────
    output_names = cfg["output_names"]
    for ds, out_name in zip(datasets, output_names):
        train_ids = np.array(
            [syscall_to_id.get(e, oob) for e in raw[ds]["train"]], dtype=np.int32
        )
        test_ids = np.array(
            [syscall_to_id.get(e, oob) for e in raw[ds]["test"]], dtype=np.int32
        )
        write_dataset(ds, out_name, train_ids, test_ids, vocab_size)

    print(f"\n  Family {family_name} done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["mem", "cpu", "all"], default="all",
                        help="Which family to process (default: all)")
    args = parser.parse_args()

    families_to_run = list(FAMILIES.keys()) if args.family == "all" else [args.family]
    for fname in families_to_run:
        process_family(fname, FAMILIES[fname])

    print("\nAll done. Re-run cross-workload SLURM jobs to get corrected results.")
