"""
generate_varwl_report.py
──────────────────────────────────────────────────────────────────────────────
Automatically generates the RQ-W results report from the actual metrics.json
files written by evaluate_metrics.py.

Fixes:
  - No hand-filled tables: every number comes directly from a metrics.json.
  - Sanity-checks that the evaluated original0.npy belongs to the intended
    TEST_DS by comparing its shape / token distribution against the test.npy
    in the Datasets directory.
  - Computes cross-workload degradation against the correct in-workload baseline
    (dynamically loaded from the same results run, not from a hard-coded table).
  - Loads majority-token baselines from the ELK metadata.json files if available.

Usage:
  python3 varying-workload/generate_varwl_report.py \
      --results-base $SCRATCH/varying-workload-results \
      [--seq-len 200] [--missing-k 10] [--output varying-workload/VARWL_RESULTS.txt]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from textwrap import dedent

import numpy as np


# ── Config ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS  = REPO_ROOT / "TraceReconstruction-main" / "Datasets"

CONDITIONS_ORDERED = [
    # (condition_label, train_ds, test_ds)
    ("elk-clean-baseline",      "elk_clean",         "elk_clean"),
    ("elk-noisy-baseline",      "elk_noisy",         "elk_noisy"),
    ("elk-clean-to-noisy",      "elk_clean",         "elk_noisy"),
    ("elk-noisy-to-clean",      "elk_noisy",         "elk_clean"),
    ("pts-stream-baseline",     "stream_sharedmem",  "stream_sharedmem"),
    ("pts-ramspeed-baseline",   "ramspeed_sharedmem","ramspeed_sharedmem"),
    ("pts-stream-to-ramspeed",  "stream_sharedmem",  "ramspeed_sharedmem"),
    ("pts-ramspeed-to-stream",  "ramspeed_sharedmem","stream_sharedmem"),
    ("pts-pybench-baseline",    "pybench_sharedcpu", "pybench_sharedcpu"),
    ("pts-phpbench-baseline",   "phpbench_sharedcpu","phpbench_sharedcpu"),
    ("pts-pybench-to-phpbench", "pybench_sharedcpu", "phpbench_sharedcpu"),
    ("pts-phpbench-to-pybench", "phpbench_sharedcpu","pybench_sharedcpu"),
]

FAMILIES = {
    "ELK Workload":          ["elk-clean-baseline",   "elk-noisy-baseline",
                               "elk-clean-to-noisy",   "elk-noisy-to-clean"],
    "PTS Memory-Bandwidth":  ["pts-stream-baseline",  "pts-ramspeed-baseline",
                               "pts-stream-to-ramspeed", "pts-ramspeed-to-stream"],
    "PTS CPU-Benchmark":     ["pts-pybench-baseline", "pts-phpbench-baseline",
                               "pts-pybench-to-phpbench", "pts-phpbench-to-pybench"],
}

# For degradation: (cross_condition, baseline_condition)
CROSS_PAIRS = [
    ("elk-clean-to-noisy",      "elk-noisy-baseline"),
    ("elk-noisy-to-clean",      "elk-clean-baseline"),
    ("pts-stream-to-ramspeed",  "pts-ramspeed-baseline"),
    ("pts-ramspeed-to-stream",  "pts-stream-baseline"),
    ("pts-pybench-to-phpbench", "pts-phpbench-baseline"),
    ("pts-phpbench-to-pybench", "pts-pybench-baseline"),
]


def run_dir(results_base: Path, train_ds: str, test_ds: str,
            seq_len: int, k: int) -> Path:
    """Return the exact output directory used by make_trace_config.py."""
    if train_ds != test_ds:
        label = f"{train_ds}_to_{test_ds}_seq{seq_len}_k{k}"
    else:
        label = f"{train_ds}_seq{seq_len}_k{k}"
    return results_base / "SSSDS4" / label / "T200_beta00.0001_betaT0.02"


def load_metrics(results_base: Path, train_ds: str, test_ds: str,
                 seq_len: int, k: int) -> dict | None:
    d = run_dir(results_base, train_ds, test_ds, seq_len, k)
    mf = d / "metrics.json"
    if mf.exists():
        m = json.loads(mf.read_text())
        m["_dir"] = str(d)
        return m
    return None


# ── Sanity check ──────────────────────────────────────────────────────────────
def sanity_check_test_ds(results_base: Path, train_ds: str, test_ds: str,
                          seq_len: int, k: int) -> dict:
    """
    Verify that original0.npy in the inference output directory matches the
    test.npy from the TEST dataset (not train).

    Returns a dict: {"ok": bool, "message": str, ...}
    """
    d = run_dir(results_base, train_ds, test_ds, seq_len, k)
    orig_path = d / "original0.npy"
    test_path = DATASETS / test_ds / f"sequence_length_{seq_len}" / "test.npy"

    if not orig_path.exists():
        return {"ok": False, "message": f"original0.npy not found in {d}"}
    if not test_path.exists():
        return {"ok": False, "message": f"test.npy not found at {test_path}"}

    # Collect ALL original batches in the inference output dir, concatenate.
    # SSSD saves original{i}.npy as channels-first (N, 1, L).
    # test.npy is channels-last (N, L, 1).
    batch_files = sorted(d.glob("original*.npy"),
                         key=lambda p: int(p.stem.replace("original", "")))
    if not batch_files:
        return {"ok": False, "message": f"No original*.npy found in {d}"}

    try:
        batches = [np.load(str(p)) for p in batch_files]
        # (total, 1, L) channels-first → (total, L) flat
        orig_cf = np.concatenate(batches, axis=0)           # (N_inf, 1, L)
        if orig_cf.ndim == 3:
            orig_rows = orig_cf[:, 0, :]                    # (N_inf, L)
        else:
            orig_rows = orig_cf.reshape(orig_cf.shape[0], -1)
    except Exception as e:
        return {"ok": False, "message": f"Failed loading original batches: {e}"}

    test_arr = np.load(str(test_path))                       # (N_test, L, 1) channels-last
    if test_arr.ndim == 3:
        test_rows = test_arr[:, :, 0]                        # (N_test, L)
    else:
        test_rows = test_arr.reshape(test_arr.shape[0], -1)

    n_inf   = orig_rows.shape[0]
    n_test  = test_rows.shape[0]
    seq_inf = orig_rows.shape[1]

    if seq_inf != seq_len:
        return {"ok": False,
                "message": f"Sequence length mismatch: orig={seq_inf} expected={seq_len}"}

    if n_inf > n_test:
        return {"ok": False,
                "message": f"Inference produced {n_inf} sequences but test.npy only has {n_test}"}

    # Exact element-wise equality against the first n_inf rows of test.npy
    expected = test_rows[:n_inf].astype(np.int64)
    actual   = orig_rows.astype(np.int64)
    if not np.array_equal(actual, expected):
        # Report first mismatch to aid debugging
        diff_mask = actual != expected
        bad_seq, bad_pos = np.argwhere(diff_mask)[0]
        return {
            "ok": False,
            "message": (
                f"MISMATCH: original*.npy != test.npy[:n_inf]. "
                f"First diff at seq={bad_seq} pos={bad_pos}: "
                f"got {actual[bad_seq, bad_pos]} expected {expected[bad_seq, bad_pos]}. "
                f"Check that inference used TEST_DS='{test_ds}' data."
            ),
            "n_mismatches": int(diff_mask.sum()),
        }

    return {
        "ok": True,
        "message": "OK",
        "n_sequences_checked": int(n_inf),
        "exact_match": True,
    }


# ── Majority-token baseline loader ────────────────────────────────────────────
def load_majority_baseline(test_ds: str, seq_len: int) -> dict | None:
    meta_path = (DATASETS / test_ds / f"sequence_length_{seq_len}" / "metadata.json")
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    return meta.get("majority_baseline")


# ── Formatting helpers ────────────────────────────────────────────────────────
def fmt_metric(v) -> str:
    if v is None:
        return "[MISSING]"
    return f"{v:.2f}"


def sign_str(delta) -> str:
    if delta is None:
        return "  [?]  "
    if abs(delta) < 0.05:
        return "  ≈0   "
    s = f"{delta:+.2f}"
    return s.ljust(7)


# ── Report generation ─────────────────────────────────────────────────────────
def generate_report(results_base: Path, seq_len: int, k: int,
                    output_path: Path | None) -> str:

    # Build condition → (label, train_ds, test_ds)
    cond_map = {c: (c, tr, te) for c, tr, te in CONDITIONS_ORDERED}

    # Load all available metrics
    metrics: dict[str, dict | None] = {}
    for cond, train_ds, test_ds in CONDITIONS_ORDERED:
        metrics[cond] = load_metrics(results_base, train_ds, test_ds, seq_len, k)

    # Run sanity checks
    sanity: dict[str, dict] = {}
    for cond, train_ds, test_ds in CONDITIONS_ORDERED:
        if metrics[cond] is not None:
            sanity[cond] = sanity_check_test_ds(
                results_base, train_ds, test_ds, seq_len, k)

    # Build degradation table
    degradation: dict[str, dict | None] = {}
    for cross_cond, baseline_cond in CROSS_PAIRS:
        cm = metrics.get(cross_cond)
        bm = metrics.get(baseline_cond)
        if cm and bm:
            degradation[cross_cond] = {
                "acc_delta":        round(bm["accuracy"]     - cm["accuracy"],     2),
                "perf_delta":       round(bm["perfect_rate"] - cm["perfect_rate"], 2),
                "rouge_delta":      round(bm["rouge_l"]      - cm["rouge_l"],      2),
                "baseline":         baseline_cond,
            }
        else:
            degradation[cross_cond] = None

    lines = []
    W = 78

    def hdr(title: str):
        lines.append("═" * W)
        lines.append(f"  {title}")
        lines.append("─" * W)

    lines.append("=" * W)
    lines.append("  RQ-W VARYING-WORKLOAD RESULTS  (auto-generated)")
    lines.append(f"  seq_len={seq_len}  blackout_k={k}")
    lines.append(f"  results_base: {results_base}")
    lines.append("=" * W)

    # ── Sanity check summary ──────────────────────────────────────────────
    lines.append("")
    hdr("SANITY CHECKS (TEST_DS verification)")
    problems = [(c, s) for c, s in sanity.items() if not s.get("ok", True)]
    if not problems:
        lines.append("  All evaluated conditions: original0.npy matches intended TEST_DS. ✓")
    else:
        for cond, s in problems:
            lines.append(f"  ⚠  {cond}: {s['message']}")
    lines.append("")

    # ── Results tables by family ──────────────────────────────────────────
    for family_name, family_conds in FAMILIES.items():
        lines.append("")
        hdr(family_name)
        header = f"{'Condition':<30} {'Acc%':>7} {'PerfR%':>7} {'ROUGER':>7} {'N':>5}  {'Majority%':>9}"
        lines.append(header)
        lines.append("-" * W)

        for cond in family_conds:
            _, train_ds, test_ds = cond_map[cond]
            m = metrics.get(cond)
            is_cross = (train_ds != test_ds)
            prefix = "→ " if is_cross else "  "

            if m is None:
                lines.append(f"{prefix}{cond:<28} {'[MISSING]':>7} {'[MISSING]':>7} {'[MISSING]':>7} {'?':>5}  {'?':>9}")
                continue

            maj = load_majority_baseline(test_ds, seq_len)
            maj_str = f"{maj['majority_accuracy']:.1f}" if maj else "?"

            sanity_flag = ""
            s = sanity.get(cond, {})
            if not s.get("ok", True):
                sanity_flag = " ⚠"

            lines.append(
                f"{prefix}{cond:<28} "
                f"{fmt_metric(m.get('accuracy')):>7} "
                f"{fmt_metric(m.get('perfect_rate')):>7} "
                f"{fmt_metric(m.get('rouge_l')):>7} "
                f"{m.get('n_sequences', '?'):>5}"
                f"  {maj_str:>9}"
                f"{sanity_flag}"
            )

        lines.append("")

    # ── Cross-workload degradation ─────────────────────────────────────────
    lines.append("")
    hdr("CROSS-WORKLOAD DEGRADATION (Baseline − Cross; positive = worse)")
    lines.append(
        f"{'Transfer pair':<32} {'Acc Δ':>8} {'PerfR Δ':>8} {'ROUGE Δ':>8}  Baseline"
    )
    lines.append("-" * W)

    all_missing = False
    for cross_cond, baseline_cond in CROSS_PAIRS:
        d = degradation.get(cross_cond)
        tag = "⚠ MISSING" if d is None else ""
        if d is None:
            lines.append(f"  {cross_cond:<30} {'?':>8} {'?':>8} {'?':>8}  ← {baseline_cond}  {tag}")
            all_missing = True
        else:
            acc_s  = sign_str(d["acc_delta"])
            perf_s = sign_str(d["perf_delta"])
            roug_s = sign_str(d["rouge_delta"])
            lines.append(
                f"  {cross_cond:<30} {acc_s:>8} {perf_s:>8} {roug_s:>8}  ← {baseline_cond}"
            )
    lines.append("")
    lines.append("  Note: positive Δ = cross-workload condition is WORSE than baseline.")
    lines.append("        negative Δ = cross-workload is BETTER than baseline.")

    # ── ELK interpretation note ────────────────────────────────────────────
    lines.append("")
    lines.append("  ELK note: The ELK split uses real CTF timestamps (no fallback).")
    lines.append("  Clean phase: ~12 ev/s (14,737 events, STRIDE=1 dense windows).")
    lines.append("  Noisy phase: ~647 ev/s (426,690 events). Because stress-ng tokens")
    lines.append("  dominate the noisy vocabulary, the noisy majority baseline is ~99.5%;")
    lines.append("  the model trivially achieves ~100% on noisy test sets by predicting")
    lines.append("  the dominant class. Clean majority baseline is ~38%, giving meaningful")
    lines.append("  signal. ELK is treated as supporting evidence; headline cross-workload")
    lines.append("  results come from the PTS shared-vocab experiments.")

    # ── Raw metrics dump ───────────────────────────────────────────────────
    lines.append("")
    hdr("RAW METRICS (from metrics.json files)")
    for cond, train_ds, test_ds in CONDITIONS_ORDERED:
        m = metrics.get(cond)
        if m:
            src = m.get("_dir", "?")
            lines.append(f"  {cond}")
            lines.append(f"    dir  : {src}")
            lines.append(f"    acc  : {m.get('accuracy')}  perf: {m.get('perfect_rate')}  "
                         f"rouge: {m.get('rouge_l')}  n={m.get('n_sequences')}")
        else:
            lines.append(f"  {cond}: [NOT YET COMPUTED]")

    lines.append("")
    lines.append("=" * W)

    report = "\n".join(lines)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report + "\n")
        print(f"Report written → {output_path}", flush=True)
    else:
        print(report)

    return report


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RQ-W results report.")
    parser.add_argument("--results-base", required=True,
                        help="Path to $SCRATCH/varying-workload-results")
    parser.add_argument("--seq-len",  type=int, default=200)
    parser.add_argument("--missing-k", type=int, default=10)
    parser.add_argument("--output",
                        help="Output file path (default: print to stdout)")
    args = parser.parse_args()

    generate_report(
        results_base = Path(args.results_base),
        seq_len      = args.seq_len,
        k            = args.missing_k,
        output_path  = Path(args.output) if args.output else None,
    )
