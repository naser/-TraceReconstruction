"""
preprocess_elk_workloads_v2.py
──────────────────────────────────────────────────────────────────────────────
Redesigned ELK workload split that fixes two bugs from v1:

Bug 1 — Approximate event-count boundary instead of real timestamps.
  Fix:  Reads the raw LTTng CTF binary trace from elk-extracted/elktracelong/
        and extracts per-PACKET timestamps.  Uses the known experiment timeline
        (0–1200 s = clean, 1200–1860 s = noisy) to assign events to phases.
        Falls back to event-count fractions if timestamp extraction fails.

Bug 2 — "First 80% train, last 20% test" inside each phase.
  Fix:  Chunk-based interleaved sampling.  The event list is divided into
        CHUNK_SIZE-event blocks; every TEST_EVERY-th block goes to the test
        pool, the rest to train.  This ensures both pools sample uniformly
        across the phase, not just from the end.

Additionally:
  - Reports a majority-token baseline for each split (predict the single most
    common token for all missing positions) so the paper can report this
    trivial lower bound alongside the model.
  - Saves a metadata JSON alongside each dataset split.

Outputs (same format as v1, compatible with all downstream scripts):
  TraceReconstruction-main/Datasets/elk_clean/sequence_length_{N}/
      train.npy  test.npy  training  testing  metadata.json
  TraceReconstruction-main/Datasets/elk_noisy/sequence_length_{N}/  (same)

The shared vocabulary over ALL events is used (identical to v1).

Usage:
  python3 varying-workload/preprocess_elk_workloads_v2.py
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent
ELK_TXT     = REPO_ROOT / "TraceReconstruction-main" / "Datasets" / "elk" / "elk_syscalls.txt"
ELK_TRACE   = REPO_ROOT / "elk-extracted" / "elktracelong" / "kernel"
OUT_BASE    = REPO_ROOT / "TraceReconstruction-main" / "Datasets"

sys.path.insert(0, str(REPO_ROOT))

# ── Configuration ─────────────────────────────────────────────────────────────
# Experiment timeline (from ELK dataset README):
#   0    – 1200 s  : clean (normal queries, no stress)
#   1200 – 1860 s  : noisy (4 × stress-ng bursts: CPU/IO/Net/Mem)
#   1860 – ~1920 s : post-noise tail (excluded)
CLEAN_END_SEC  = 1200.0
NOISY_END_SEC  = 1860.0

# Chunk-based train/test split
CHUNK_SIZE  = 500    # events per chunk
TEST_EVERY  = 5      # every 5th chunk → test (≈20% test, uniformly spread)

# Dataset parameters (match the paper)
MAX_TRAIN   = 10_000
MAX_TEST    = 500
SEQ_LENGTHS = [50, 100, 150, 200]
STRIDE_FRAC = 0.5

# ELK clean phase generates only ~12 events/s (normal ELK queries) vs stress-ng's
# ~646 events/s.  Timestamp-based split correctly yields only ~14,737 clean events.
# Dense stride-1 windowing extracts ~11,600 overlapping training windows from those
# events, reaching MAX_TRAIN=10,000.  This is the standard approach in time-series
# imputation and is more faithful than mixing clean+noisy events via an event-count
# fraction approximation.
STRIDE_ELK  = 1      # dense windowing for timestamp-split phases


# ── Step 1: Try to extract events with per-packet timestamps ──────────────────
def extract_with_timestamps(trace_dir: Path):
    """
    Read the LTTng CTF binary trace and return a list of (pkt_ns, syscall_name)
    tuples, where pkt_ns is the packet start time in nanoseconds.

    Timestamps are obtained from the packet context's timestamp_begin field,
    converted to nanoseconds using the clock frequency in the CTF metadata.
    Returns None if extraction fails.
    """
    from ctf_reader import parse_metadata, extract_events_from_channel, \
        PKT_HDR_SIZE, PKT_CTX_SIZE, CTF_MAGIC
    import struct, re

    meta_path = trace_dir / "metadata"
    if not meta_path.exists():
        return None

    # Parse clock frequency from metadata text
    text = meta_path.read_text(encoding="utf-8", errors="replace")
    freq_m = re.search(r'freq\s*=\s*(\d+)', text)
    offset_m = re.search(r'offset\s*=\s*(\d+)', text)
    freq   = int(freq_m.group(1)) if freq_m else 1_000_000_000
    offset = int(offset_m.group(1)) if offset_m else 0

    def ticks_to_ns(ticks: int) -> float:
        return (ticks - offset) / freq * 1e9

    event_map, stream_hdrs = parse_metadata(str(meta_path))

    channel_files = sorted(
        f for f in os.listdir(trace_dir)
        if os.path.isfile(trace_dir / f)
        and f != "metadata"
        and not f.endswith(".idx")
        and not f.startswith(".")
    )

    events: list[tuple[float, str]] = []   # (ns, name)

    for ch in channel_files:
        ch_path = str(trace_dir / ch)
        with open(ch_path, "rb") as f:
            raw = bytearray(f.read())

        n = len(raw)
        pos = 0

        while pos + PKT_HDR_SIZE + PKT_CTX_SIZE <= n:
            if struct.unpack_from("<I", raw, pos)[0] != CTF_MAGIC:
                nxt = raw.find(struct.pack("<I", CTF_MAGIC), pos + 1)
                if nxt == -1:
                    break
                pos = nxt
                continue

            pkt_start = pos
            pkt_stream_id = struct.unpack_from("<I", raw, pos + 20)[0]
            pos += PKT_HDR_SIZE

            # Packet context: timestamp_begin at offset 0 (8 bytes), ts_end at 8
            pkt_ts_begin = struct.unpack_from("<Q", raw, pos)[0]
            content_bits = struct.unpack_from("<Q", raw, pos + 16)[0]
            packet_bits  = struct.unpack_from("<Q", raw, pos + 24)[0]
            pos += PKT_CTX_SIZE

            if not content_bits:
                pos = pkt_start + (packet_bits >> 3 if packet_bits else 4096)
                continue

            content_end = pkt_start + (content_bits >> 3)
            pkt_end     = pkt_start + (packet_bits >> 3 if packet_bits else content_bits >> 3)
            if content_end > n:
                content_end = n

            # Nanoseconds for this packet's start
            pkt_ns = ticks_to_ns(pkt_ts_begin)

            hdr_type    = stream_hdrs.get(pkt_stream_id, "compact")
            is_stream0  = (pkt_stream_id == 0)
            STREAM0_EVT_CTX_SIZE = 24

            # Yield all syscalls in this packet with pkt_ns as their timestamp
            from ctf_reader import _read_event_header_compact, _read_event_header_large, _skip_fields
            while pos < content_end:
                if pos + 4 > n:
                    break
                try:
                    if hdr_type == "large":
                        event_id, pos = _read_event_header_large(raw, pos)
                    else:
                        event_id, pos = _read_event_header_compact(raw, pos)
                except Exception:
                    break

                if is_stream0:
                    pos += STREAM0_EVT_CTX_SIZE

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
                    events.append((pkt_ns, clean))

            pos = pkt_end

    return events if events else None


# ── Step 2: Fall back to event-count fraction method ─────────────────────────
def load_from_txt(path: Path) -> list[str]:
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


# ── Step 3: Chunk-based interleaved train/test split ──────────────────────────
def chunk_split(events: list) -> tuple[list, list]:
    """
    Divide events into CHUNK_SIZE blocks; every TEST_EVERY-th block goes to
    test, the rest to train.  Both pools sample uniformly across the phase.
    """
    train, test = [], []
    n_chunks = len(events) // CHUNK_SIZE
    for i in range(n_chunks):
        chunk = events[i * CHUNK_SIZE: (i + 1) * CHUNK_SIZE]
        if (i % TEST_EVERY) == (TEST_EVERY - 1):
            test.extend(chunk)
        else:
            train.extend(chunk)
    # Remainder goes to train
    remainder = events[n_chunks * CHUNK_SIZE:]
    train.extend(remainder)
    return train, test


# ── Step 4: Windowing + dataset write ─────────────────────────────────────────
def sliding_windows(ids: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    n = len(ids)
    if n < seq_len:
        return np.empty((0, seq_len), dtype=np.int32)
    return np.stack([ids[s:s + seq_len] for s in range(0, n - seq_len + 1, stride)])


def majority_baseline(ids: np.ndarray, missing_k: int = 10, test_n: int = MAX_TEST,
                      seq_len: int = 200) -> dict:
    """
    Trivial baseline: always predict the single most-common token.
    Returns accuracy and perfect_rate on the test windows using centered masking.
    """
    if len(ids) < seq_len:
        return {}
    stride = max(1, seq_len // 2)
    wins = sliding_windows(ids, seq_len, stride)[:test_n]
    if len(wins) == 0:
        return {}

    # Most common token overall
    freq = Counter(ids.tolist())
    majority = max(freq, key=freq.get)

    # Centered mask: middle `missing_k` positions
    mask_start = (seq_len - missing_k) // 2
    mask_end   = mask_start + missing_k

    correct = 0
    perfect = 0
    total   = len(wins) * missing_k

    for w in wins:
        masked = w[mask_start:mask_end]
        hits = np.sum(masked == majority)
        correct += hits
        if hits == missing_k:
            perfect += 1

    return {
        "majority_token":      int(majority),
        "majority_accuracy":   round(correct / total * 100, 2),
        "majority_perfect_rate": round(perfect / len(wins) * 100, 2),
    }


def save_split(name: str, train_names: list[str], test_names: list[str],
               syscall_to_id: dict, vocab_size: int,
               split_method: str, meta_extra: dict,
               stride_override: int | None = None) -> None:
    oob = vocab_size + 1
    train_ids = np.array([syscall_to_id.get(e, oob) for e in train_names], dtype=np.int32)
    test_ids  = np.array([syscall_to_id.get(e, oob) for e in test_names],  dtype=np.int32)

    print(f"\n  [{name}]  train={len(train_ids):,}  test={len(test_ids):,}  "
          f"method={split_method}")

    for seq_len in SEQ_LENGTHS:
        out_dir = OUT_BASE / name / f"sequence_length_{seq_len}"
        out_dir.mkdir(parents=True, exist_ok=True)

        stride    = stride_override if stride_override is not None else max(1, int(seq_len * STRIDE_FRAC))
        tr_wins   = sliding_windows(train_ids, seq_len, stride)
        te_wins   = sliding_windows(test_ids, seq_len, stride)

        n_tr = min(MAX_TRAIN, len(tr_wins))
        n_te = min(MAX_TEST,  len(te_wins))
        tr   = tr_wins[:n_tr]
        te   = te_wins[:n_te]

        np.save(out_dir / "train.npy", tr[:, :, np.newaxis])
        np.save(out_dir / "test.npy",  te[:, :, np.newaxis])

        with open(out_dir / "training", "w") as f:
            for row in tr: f.write(",".join(map(str, row)) + "\n")
        with open(out_dir / "testing", "w") as f:
            for row in te: f.write(",".join(map(str, row)) + "\n")

        # Majority-token baseline for this seq_len
        maj = majority_baseline(test_ids, missing_k=10, test_n=MAX_TEST, seq_len=seq_len)

        meta = {
            "dataset":       name,
            "seq_len":       seq_len,
            "vocab_size":    vocab_size,
            "oob_id":        oob,
            "split_method":  split_method,
            "n_train_events": len(train_ids),
            "n_test_events":  len(test_ids),
            "n_train_seqs":  int(n_tr),
            "n_test_seqs":   int(n_te),
            "majority_baseline": maj,
            **meta_extra,
        }
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"    seq_len={seq_len}: train{tr.shape}  test{te.shape}  "
              f"majority_acc={maj.get('majority_accuracy','?')}%")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading ELK syscall sequence from text …")
    all_events_txt = load_from_txt(ELK_TXT)
    total = len(all_events_txt)
    print(f"  Total events (text file): {total:,}")

    # ── Build shared vocab from ALL events (same as v1) ──────────────────
    print("Building shared vocabulary …")
    freq = Counter(all_events_txt)
    sorted_calls  = sorted(freq.keys(), key=lambda c: (-freq[c], c))
    syscall_to_id = {sc: idx + 1 for idx, sc in enumerate(sorted_calls)}
    vocab_size    = len(syscall_to_id)
    print(f"  Shared vocab size: {vocab_size}")

    # ── Timestamp-based phase split (always; no fallback) ────────────────
    # The clean phase genuinely produces only ~14,737 events over 1200 s
    # (~12 events/s from normal ELK queries) because the kernel is nearly idle
    # during low-load Elasticsearch operation.  This is the real experiment:
    # the model is trained on quiet ELK syscalls and tested on stress-ng
    # saturated syscalls (or vice versa).  Dense stride-1 windowing (STRIDE_ELK)
    # extracts ~11,600 overlapping training windows from 14K events, which is
    # sufficient for SSSD.  A fraction-based fallback would contaminate the clean
    # pool with noisy events and defeat the purpose of the workload-shift test.
    split_method = "timestamp"

    print(f"\nExtracting timestamps from {ELK_TRACE} …")
    if not ELK_TRACE.exists():
        sys.exit(f"ERROR: CTF trace not found at {ELK_TRACE}")

    try:
        ts_events = extract_with_timestamps(ELK_TRACE)
    except Exception as e:
        sys.exit(f"ERROR: CTF timestamp extraction failed: {e}")

    if not ts_events or len(ts_events) < 1000:
        sys.exit(f"ERROR: extracted only {len(ts_events or [])} events — "
                 f"check {ELK_TRACE}")

    times_ns = [t for t, _ in ts_events]
    t0 = min(times_ns)
    clean_names: list[str] = []
    noisy_names: list[str] = []
    for rel_ns, name in ts_events:
        rel_s = (rel_ns - t0) / 1e9
        if rel_s < CLEAN_END_SEC:
            clean_names.append(name)
        elif rel_s < NOISY_END_SEC:
            noisy_names.append(name)
        # tail (≥ NOISY_END_SEC) excluded

    ts_meta = {
        "trace_start_epoch_ns": int(t0),
        "clean_boundary_s":     CLEAN_END_SEC,
        "noisy_boundary_s":     NOISY_END_SEC,
        "clean_events":         len(clean_names),
        "noisy_events":         len(noisy_names),
        "clean_rate_per_sec":   round(len(clean_names) / CLEAN_END_SEC, 1),
        "noisy_rate_per_sec":   round(len(noisy_names) / (NOISY_END_SEC - CLEAN_END_SEC), 1),
        "stride_elk":           STRIDE_ELK,
        "note": ("Clean phase is intentionally sparse (~12 events/s = idle ELK queries). "
                 "Dense stride-1 windowing extracts enough training windows from 14K events."),
    }
    print(f"  Timestamp split: clean={len(clean_names):,}  noisy={len(noisy_names):,}")
    print(f"  Clean rate: {ts_meta['clean_rate_per_sec']:.1f} ev/s  "
          f"Noisy rate: {ts_meta['noisy_rate_per_sec']:.1f} ev/s")

    # ── Vocabulary coverage report ────────────────────────────────────────
    for phase_name, phase_names in [("clean", clean_names), ("noisy", noisy_names)]:
        distinct = set(phase_names)
        print(f"  {phase_name}: {len(phase_names):,} events, "
              f"{len(distinct)} distinct syscalls "
              f"({len(distinct)/vocab_size*100:.1f}% of shared vocab)")

    # ── Chunk-based train/test split ──────────────────────────────────────
    print(f"\nChunk-based train/test split "
          f"(chunk={CHUNK_SIZE}, test_every={TEST_EVERY}) …")
    clean_train, clean_test_raw = chunk_split(clean_names)
    noisy_train, noisy_test_raw = chunk_split(noisy_names)

    # Cap test pool
    clean_test = clean_test_raw[:MAX_TEST * 200]   # generous cap; windowing trims later
    noisy_test = noisy_test_raw[:MAX_TEST * 200]

    print(f"  elk_clean: train_pool={len(clean_train):,}  test_pool={len(clean_test):,}")
    print(f"  elk_noisy: train_pool={len(noisy_train):,}  test_pool={len(noisy_test):,}")

    # ── Generate and save both splits ─────────────────────────────────────
    clean_meta = {
        "phase": "clean",
        "chunk_size": CHUNK_SIZE,
        "test_every_nth_chunk": TEST_EVERY,
        **ts_meta,
    }
    noisy_meta = {
        "phase": "noisy",
        "chunk_size": CHUNK_SIZE,
        "test_every_nth_chunk": TEST_EVERY,
        **ts_meta,
    }

    save_split("elk_clean", clean_train, clean_test, syscall_to_id,
               vocab_size, split_method, clean_meta, stride_override=STRIDE_ELK)
    save_split("elk_noisy", noisy_train, noisy_test, syscall_to_id,
               vocab_size, split_method, noisy_meta, stride_override=STRIDE_ELK)

    print("\nDone. Datasets written to:")
    print(f"  {OUT_BASE / 'elk_clean'}/")
    print(f"  {OUT_BASE / 'elk_noisy'}/")


if __name__ == "__main__":
    main()
