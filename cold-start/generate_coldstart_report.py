"""
generate_coldstart_report.py
──────────────────────────────────────────────────────────────────────────────
Generate the RQ-CS (Training-Data-Size / Cold-Start) results report from
metrics.json files produced by the cold-start SLURM jobs.

Experiment design (loaded from cold-start-results):
  Model   : SSSD^S4 (best model across all RQ-1–RQ-6 experiments)
  Dataset : phpbench_sharedcpu  (84-token shared CPU-benchmark vocabulary)
  Sizes   : 100, 500, 1 000, 2 500, 5 000  (scratch training)
  +Reference: 10 000 sequences  ← from existing RQ-W pts-phpbench-baseline run

SHARED-VOCABULARY DISCLOSURE
  phpbench_sharedcpu uses the shared 84-token CPU-benchmark vocabulary built
  by varying-workload/preprocess_pts_shared_vocab.py (shared with pybench_sharedcpu).
  The paper's default preprocessing (§III-B) assigns each dataset its own
  frequency-ranked vocabulary.  The shared vocabulary is required here so that
  the cold-start and transfer-learning experiments use consistent token IDs and
  are directly comparable.  This deviation must be disclosed in the paper if
  these results are reported.

SUBSAMPLING METHOD
  Subsets are drawn by UNIFORM RANDOM SAMPLING WITHOUT REPLACEMENT (seed=42)
  from the 10 000-sequence training pool.  This is NOT stratified sampling.

Metrics reported (over masked positions only):
  • Accuracy (%)       — fraction of missing events exactly matched
  • Perfect Rate (%)   — fraction of sequences 100 % correctly reconstructed
  • ROUGE-L (%)        — LCS-based token overlap

Additional analysis:
  • Sample-efficiency curve: how quickly each metric saturates with more data
  • Relative gap to full-data (10k) performance at each size

Usage:
  python3 cold-start/generate_coldstart_report.py \\
      --cold-start-base $SCRATCH/cold-start-results \\
      --varwl-base      $SCRATCH/varying-workload-results \\
      [--seq-len 200] [--missing-k 10] \\
      [--output cold-start/COLDSTART_RESULTS.txt]
"""

import argparse
import json
import sys
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
DATASETS    = REPO_ROOT / "TraceReconstruction-main" / "Datasets"

DATASET     = "phpbench_sharedcpu"
SOURCE_DS   = "pybench_sharedcpu"          # source for fine-tuning (used in Exp 2 report)
TRAIN_SIZES = [100, 500, 1000, 2500, 5000]  # scratch-trained sizes
FULL_SIZE   = 10000                          # full-data reference from RQ-W

W = 72   # report column width


# ── Path helpers ─────────────────────────────────────────────────────────────
def cold_start_run_dir(cold_base: Path, n: int, seq_len: int, k: int) -> Path:
    """Output dir for scratch training on phpbench_sharedcpu_n{N}."""
    # make_trace_config sets run_label = "{sub_ds}_to_{test_ds}_seq{L}_k{K}"
    # because SUBSAMPLED_DS != DATASET (test-dataset differs from dataset)
    sub_ds   = f"{DATASET}_n{n}"
    run_label = f"{sub_ds}_to_{DATASET}_seq{seq_len}_k{k}"
    return cold_base / "SSSDS4" / run_label / "T200_beta00.0001_betaT0.02"


def varwl_run_dir(varwl_base: Path, ds: str, seq_len: int, k: int) -> Path:
    """Output dir for a full-dataset RQ-W baseline run."""
    run_label = f"{ds}_seq{seq_len}_k{k}"
    return varwl_base / "SSSDS4" / run_label / "T200_beta00.0001_betaT0.02"


def load_metrics(d: Path) -> dict | None:
    mf = d / "metrics.json"
    if mf.exists():
        m = json.loads(mf.read_text())
        m["_dir"] = str(d)
        return m
    return None


def load_majority_baseline(ds: str, seq_len: int) -> dict | None:
    meta = DATASETS / ds / f"sequence_length_{seq_len}" / "metadata.json"
    if not meta.exists():
        return None
    return json.loads(meta.read_text()).get("majority_baseline")


# ── Formatting ────────────────────────────────────────────────────────────────
def fmt(v, decimals=2) -> str:
    if v is None:
        return "[MISSING]"
    return f"{v:.{decimals}f}"


def gap_str(val, ref) -> str:
    """How far val is below the ref (full-data) performance."""
    if val is None or ref is None:
        return "  [?]"
    d = val - ref
    if abs(d) < 0.05:
        return "  ≈0 "
    return f"{d:+.2f}"


# ── Report generation ─────────────────────────────────────────────────────────
def generate_report(cold_base: Path, varwl_base: Path,
                    seq_len: int, k: int,
                    output_path: Path | None = None) -> str:
    lines = []

    def hdr(title: str) -> None:
        lines.append("─" * W)
        lines.append(f"  {title}")
        lines.append("─" * W)

    lines.append("=" * W)
    lines.append("  RQ-CS: Training-Data-Size / Cold-Start Experiment  [ADDED]")
    lines.append("=" * W)
    lines.append(f"  Model      : SSSD^S4 (best model, paper §IV)")
    lines.append(f"  Dataset    : {DATASET}  (84-token shared CPU-benchmark vocab)")
    lines.append(f"  Seq len    : {seq_len}   Blackout k: {k}   Masking: center (cm)")
    lines.append(f"  Paper default: seq=200, k=10, masking=cm, T=200 diffusion steps")
    lines.append(f"  Subsampling: uniform random without replacement (seed=42)")
    lines.append("")
    lines.append("  METHODOLOGICAL NOTE — Shared vocabulary")
    lines.append("  ────────────────────────────────────────")
    lines.append(f"  {DATASET} uses a SHARED 84-token vocabulary built across")
    lines.append(f"  pybench_sharedcpu + phpbench_sharedcpu by")
    lines.append(f"  varying-workload/preprocess_pts_shared_vocab.py.")
    lines.append(f"  The paper's default preprocessing (§III-B) assigns each dataset")
    lines.append(f"  its own frequency-ranked vocabulary.  The shared vocabulary is")
    lines.append(f"  required for cross-size comparability and for the transfer-learning")
    lines.append(f"  experiment (RQ-TL).  Disclose this deviation in any paper text.")
    lines.append("")

    # ── Load all metrics ──────────────────────────────────────────────────
    metrics = {}

    # Scratch training at each subsampled size
    for n in TRAIN_SIZES:
        d = cold_start_run_dir(cold_base, n, seq_len, k)
        metrics[n] = load_metrics(d)

    # Full-data reference from RQ-W
    d_full = varwl_run_dir(varwl_base, DATASET, seq_len, k)
    metrics[FULL_SIZE] = load_metrics(d_full)

    # Majority baseline
    majority = load_majority_baseline(DATASET, seq_len)

    # ── Summary table ─────────────────────────────────────────────────────
    lines.append("")
    hdr("SAMPLE-EFFICIENCY TABLE  (scratch training from random init)")
    lines.append(
        f"{'N_train':>8}  {'Accuracy':>9}  {'PerfRate':>9}  {'ROUGE-L':>9}"
        f"  {'ΔAcc↑':>7}  {'ΔPerf↑':>7}  {'ΔRouge↑':>7}"
    )
    lines.append(
        f"{'':>8}  {'(%)':>9}  {'(%)':>9}  {'(%)':>9}"
        f"  {'vs 10k':>7}  {'vs 10k':>7}  {'vs 10k':>7}"
    )
    lines.append("-" * W)

    ref = metrics.get(FULL_SIZE)
    ref_acc   = ref.get("accuracy")   if ref else None
    ref_perf  = ref.get("perfect_rate") if ref else None
    ref_rouge = ref.get("rouge_l")    if ref else None

    all_sizes = TRAIN_SIZES + [FULL_SIZE]
    for n in all_sizes:
        m = metrics.get(n)
        acc   = m.get("accuracy")     if m else None
        perf  = m.get("perfect_rate") if m else None
        rouge = m.get("rouge_l")      if m else None

        ref_marker = "  ← reference (RQ-W)" if n == FULL_SIZE else ""
        missing    = "  ⚠ MISSING"           if m is None     else ""

        lines.append(
            f"{n:>8}  {fmt(acc):>9}  {fmt(perf):>9}  {fmt(rouge):>9}"
            f"  {gap_str(acc, ref_acc):>7}  {gap_str(perf, ref_perf):>7}"
            f"  {gap_str(rouge, ref_rouge):>7}"
            f"{ref_marker}{missing}"
        )

    lines.append("")
    if majority:
        maj_acc   = majority.get("accuracy")
        maj_rouge = majority.get("rouge_l")
        lines.append(
            f"  Majority baseline (always predict most-frequent token):"
            f"  acc={fmt(maj_acc)}%  rouge={fmt(maj_rouge)}%"
        )

    # ── Sample-efficiency curve (ASCII) ──────────────────────────────────
    lines.append("")
    hdr("SAMPLE-EFFICIENCY CURVE  (Accuracy % vs Training Size)")

    available  = [(n, metrics[n]) for n in all_sizes if metrics.get(n)]
    if len(available) >= 2:
        max_acc = max(m["accuracy"] for _, m in available if m.get("accuracy"))
        for n, m in available:
            acc = m.get("accuracy") or 0.0
            bar_len = int(round(acc / max_acc * 40))
            bar = "█" * bar_len
            lines.append(f"  {n:>6}  {bar:<40}  {fmt(acc)}%")
    else:
        lines.append("  [Not enough data points to render curve]")

    lines.append("")
    lines.append("  Interpretation:")
    lines.append("  • The curve shows how quickly SSSD^S4 saturates as training")
    lines.append("    data grows.  A steep rise at small N indicates the model")
    lines.append("    can learn useful patterns with few sequences (good for")
    lines.append("    cold-start scenarios).  A flat tail at large N means")
    lines.append("    performance is data-efficient and collection of large")
    lines.append("    datasets is not required.")

    # ── Minimum viable training size ─────────────────────────────────────
    lines.append("")
    hdr("MINIMUM-VIABLE-TRAINING-SIZE ANALYSIS")
    thresholds = [0.90, 0.95, 0.99]   # fraction of full-data accuracy
    if ref_acc:
        for t in thresholds:
            target = ref_acc * t
            for n in all_sizes:
                m = metrics.get(n)
                if m and m.get("accuracy") is not None and m["accuracy"] >= target:
                    lines.append(
                        f"  {int(t*100):3}% of full-data accuracy ({target:.2f}%) "
                        f"reached at N={n}  (acc={fmt(m['accuracy'])}%)"
                    )
                    break
            else:
                lines.append(
                    f"  {int(t*100):3}% of full-data accuracy ({target:.2f}%) "
                    f"not reached at any tested training size."
                )
    else:
        lines.append("  [Full-data reference (N=10k) not available; "
                     "load from RQ-W results]")

    # ── Missing results ───────────────────────────────────────────────────
    lines.append("")
    missing_runs = [n for n in all_sizes if metrics.get(n) is None]
    if missing_runs:
        hdr("INCOMPLETE / MISSING RUNS")
        for n in missing_runs:
            d = cold_start_run_dir(cold_base, n, seq_len, k) if n != FULL_SIZE \
                else varwl_run_dir(varwl_base, DATASET, seq_len, k)
            lines.append(f"  N={n:<6}  expected: {d}")
        lines.append("")
        lines.append("  Re-submit missing jobs or wait for SLURM completion.")

    # ── Raw metrics dump ──────────────────────────────────────────────────
    lines.append("")
    hdr("RAW METRICS  (from metrics.json files)")
    for n in all_sizes:
        m = metrics.get(n)
        src = m.get("_dir", "?") if m else "NOT FOUND"
        if m:
            lines.append(f"  N={n}")
            lines.append(f"    dir  : {src}")
            lines.append(
                f"    acc={m.get('accuracy')}  perf={m.get('perfect_rate')}"
                f"  rouge={m.get('rouge_l')}  n_seq={m.get('n_sequences')}"
            )
        else:
            lines.append(f"  N={n}: [NOT YET COMPUTED]")

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
    parser = argparse.ArgumentParser(
        description="Generate RQ-CS training-data-size report."
    )
    parser.add_argument("--cold-start-base", required=True,
                        help="Path to $SCRATCH/cold-start-results")
    parser.add_argument("--varwl-base", required=True,
                        help="Path to $SCRATCH/varying-workload-results "
                             "(contains pts-phpbench-baseline reference)")
    parser.add_argument("--seq-len",   type=int, default=200)
    parser.add_argument("--missing-k", type=int, default=10)
    parser.add_argument("--output",
                        help="Output file path (default: print to stdout)")
    args = parser.parse_args()

    generate_report(
        cold_base   = Path(args.cold_start_base),
        varwl_base  = Path(args.varwl_base),
        seq_len     = args.seq_len,
        k           = args.missing_k,
        output_path = Path(args.output) if args.output else None,
    )
