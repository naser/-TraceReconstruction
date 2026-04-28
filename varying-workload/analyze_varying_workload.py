"""
analyze_varying_workload.py
─────────────────────────────────────────────────────────────────────────────
Aggregate RQ-W (varying-workload) experiment results and compute the three
key statistics reported in the paper:

  • Accuracy      (% of imputed events exactly matching ground truth)
  • Perfect Rate  (% of sequences where every missing event is correct)
  • ROUGE-L       (LCS-based sequence similarity)

and additionally:

  • Cross-workload Degradation: relative drop vs. in-workload baseline

Usage:
    python varying-workload/analyze_varying_workload.py \
        --results-base /scratch/ghazalkh/varying-workload-results \
        [--existing-results-dir TraceReconstruction-main/Results/SSSD_S4] \
        [--save-report varying-workload/VARYING_WORKLOAD_RESULTS.json]

The script reads:
  1. JSON files from the SLURM jobs' summaries/ directory
     (format: <condition>_seq200_k10.json)
  2. Optionally, raw imputation*.npy files for any conditions not yet
     summarised.

It outputs a comparison table to stdout and writes a structured JSON report.
"""

import argparse
import glob
import json
import os
import sys
import numpy as np

# ── Reuse evaluate_metrics logic inline ──────────────────────────────────────
def lcs_length(a, b):
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def rouge_l(original, reconstructed):
    n = len(original)
    if n == 0:
        return 100.0
    return 100.0 * lcs_length(list(original), list(reconstructed)) / n


def evaluate_from_npy(results_dir: str) -> dict:
    """Compute metrics directly from imputation*.npy files."""
    imp_files  = sorted(glob.glob(os.path.join(results_dir, "imputation*.npy")))
    orig_files = sorted(glob.glob(os.path.join(results_dir, "original*.npy")))
    mask_files = sorted(glob.glob(os.path.join(results_dir, "mask*.npy")))

    if not imp_files:
        return None

    imps, origs, masks = [], [], []
    for i, o, m in zip(imp_files, orig_files, mask_files):
        imps.append(np.load(i))
        origs.append(np.load(o))
        masks.append(np.load(m))

    imp  = np.concatenate(imps,  axis=0)[:, 0, :]
    orig = np.concatenate(origs, axis=0)[:, 0, :]
    mask = np.concatenate(masks, axis=0)[:, 0, :]

    imp_round = np.round(imp).astype(np.int32)
    orig_int  = orig.astype(np.int32)
    miss_mask = (mask == 0)

    per_acc, per_rl, perfect = [], [], 0
    for i in range(orig.shape[0]):
        idx = np.where(miss_mask[i])[0]
        if len(idx) == 0:
            continue
        o_seg = orig_int[i, idx]
        p_seg = imp_round[i, idx]
        acc = float(np.sum(o_seg == p_seg)) / len(o_seg)
        per_acc.append(acc)
        if acc == 1.0:
            perfect += 1
        per_rl.append(rouge_l(o_seg, p_seg))

    n = len(per_acc)
    if n == 0:
        return {"accuracy": 0.0, "perfect_rate": 0.0, "rouge_l": 0.0, "n_sequences": 0}
    return {
        "accuracy":     round(100.0 * np.mean(per_acc), 2),
        "perfect_rate": round(100.0 * perfect / n,       2),
        "rouge_l":      round(np.mean(per_rl),            2),
        "n_sequences":  n,
    }


# ── Result loading ────────────────────────────────────────────────────────────
def load_summary_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def find_metrics(results_base: str, condition: str,
                 seq: int = 200, k: int = 10) -> dict | None:
    """Try to load metrics for a condition, falling back to raw npy eval."""
    # 1. Summaries directory (produced by varying_workload.slurm)
    json_path = os.path.join(results_base, "summaries",
                             f"{condition}_seq{seq}_k{k}.json")
    if os.path.exists(json_path):
        m = load_summary_json(json_path)
        m["source"] = "summary_json"
        return m

    # 2. Raw imputation files anywhere under results_base
    pattern = os.path.join(results_base, "**", "imputation0.npy")
    candidates = glob.glob(pattern, recursive=True)
    # Prefer directories whose path contains the condition label
    for c in candidates:
        if condition.replace("-", "_") in c or condition in c:
            m = evaluate_from_npy(os.path.dirname(c))
            if m:
                m["source"] = "raw_npy"
                return m

    return None


def load_existing_pts_baseline(existing_results_dir: str,
                                dataset: str, k: int = 10) -> dict | None:
    """Load from TraceReconstruction-main/Results/SSSD_S4/results_<ds>_<k>/"""
    rdir = os.path.join(existing_results_dir, f"results_{dataset}_{k}")
    if os.path.exists(rdir):
        m = evaluate_from_npy(rdir)
        if m:
            m["source"] = "existing_results"
        return m
    return None


# ── Degradation computation ───────────────────────────────────────────────────
def degradation(baseline: float, cross: float) -> float:
    """Relative degradation (%) of cross vs baseline.  Positive = worse."""
    if baseline == 0:
        return float("nan")
    return round(100.0 * (baseline - cross) / baseline, 1)


# ── Pretty table ──────────────────────────────────────────────────────────────
def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    header = (f"{'Condition':<32}  {'Acc %':>7}  {'PerfRate %':>10}  "
              f"{'ROUGE-L':>8}  {'N':>6}  {'Source'}")
    print(header)
    print("-" * 80)
    for r in rows:
        m = r.get("metrics")
        if m:
            print(f"  {r['condition']:<30}  {m['accuracy']:>7.2f}  "
                  f"{m['perfect_rate']:>10.2f}  {m['rouge_l']:>8.2f}  "
                  f"{m.get('n_sequences', '?'):>6}  {m.get('source','?')}")
        else:
            print(f"  {r['condition']:<30}  {'PENDING':>7}  "
                  f"{'PENDING':>10}  {'PENDING':>8}  {'?':>6}")
    print("-" * 80)


def print_degradation_table(title: str, rows: list[dict]) -> None:
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    header = (f"{'Pair':<38}  {'Acc↓%':>7}  {'PR↓%':>7}  {'RL↓%':>7}")
    print(header)
    print("-" * 80)
    for r in rows:
        print(f"  {r['pair']:<36}  {r['acc_deg']:>7}  "
              f"{r['pr_deg']:>7}  {r['rl_deg']:>7}")
    print("-" * 80)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-base", default=None,
                        help="Base directory for SLURM job outputs "
                             "(default: $SCRATCH/varying-workload-results)")
    parser.add_argument("--existing-results-dir", default=None,
                        help="TraceReconstruction-main/Results/SSSD_S4 for "
                             "pre-existing PTS baselines")
    parser.add_argument("--save-report",
                        default="varying-workload/VARYING_WORKLOAD_RESULTS.json",
                        help="Path for the JSON results report")
    args = parser.parse_args()

    results_base = args.results_base or os.path.expandvars(
        os.path.join("$SCRATCH", "varying-workload-results"))

    existing = args.existing_results_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "TraceReconstruction-main", "Results", "SSSD_S4")

    # ── Define all 12 experiment conditions ──────────────────────────────────
    conditions = [
        # ELK block
        dict(condition="elk-clean-baseline",   train="elk_clean", test="elk_clean",  family="ELK"),
        dict(condition="elk-noisy-baseline",   train="elk_noisy", test="elk_noisy",  family="ELK"),
        dict(condition="elk-clean-to-noisy",   train="elk_clean", test="elk_noisy",  family="ELK"),
        dict(condition="elk-noisy-to-clean",   train="elk_noisy", test="elk_clean",  family="ELK"),
        # PTS memory-bandwidth block
        dict(condition="pts-stream-baseline",       train="stream",   test="stream",   family="PTS-mem"),
        dict(condition="pts-ramspeed-baseline",      train="ramspeed", test="ramspeed", family="PTS-mem"),
        dict(condition="pts-stream-to-ramspeed",     train="stream",   test="ramspeed", family="PTS-mem"),
        dict(condition="pts-ramspeed-to-stream",     train="ramspeed", test="stream",   family="PTS-mem"),
        # PTS CPU-benchmark block
        dict(condition="pts-pybench-baseline",       train="pybench",  test="pybench",  family="PTS-cpu"),
        dict(condition="pts-phpbench-baseline",      train="phpbench", test="phpbench", family="PTS-cpu"),
        dict(condition="pts-pybench-to-phpbench",    train="pybench",  test="phpbench", family="PTS-cpu"),
        dict(condition="pts-phpbench-to-pybench",    train="phpbench", test="pybench",  family="PTS-cpu"),
    ]

    # Load metrics for each condition
    all_results = []
    for cond in conditions:
        c = cond["condition"]
        m = find_metrics(results_base, c)

        # Fallback to existing pre-computed baselines for PTS in-workload
        if m is None and cond["train"] == cond["test"] and cond["family"].startswith("PTS"):
            m = load_existing_pts_baseline(existing, cond["train"])

        all_results.append({**cond, "metrics": m})

    # ── Print per-family tables ───────────────────────────────────────────────
    for family, label in [("ELK", "ELK Workload Experiment"),
                          ("PTS-mem", "PTS Memory-Bandwidth Family (stream vs ramspeed)"),
                          ("PTS-cpu", "PTS CPU-Benchmark Family (pybench vs phpbench)")]:
        rows = [r for r in all_results if r["family"] == family]
        print_table(label, rows)

    # ── Degradation table ─────────────────────────────────────────────────────
    degradation_rows = []
    cross_pairs = [
        ("elk-clean-baseline",    "elk-clean-to-noisy",    "ELK: clean→noisy  (A→B)"),
        ("elk-noisy-baseline",    "elk-noisy-to-clean",    "ELK: noisy→clean  (B→A)"),
        ("pts-stream-baseline",   "pts-stream-to-ramspeed","PTS-mem: stream→ramspeed"),
        ("pts-ramspeed-baseline", "pts-ramspeed-to-stream","PTS-mem: ramspeed→stream"),
        ("pts-pybench-baseline",  "pts-pybench-to-phpbench","PTS-cpu: pybench→phpbench"),
        ("pts-phpbench-baseline", "pts-phpbench-to-pybench","PTS-cpu: phpbench→pybench"),
    ]

    for base_cond, cross_cond, label in cross_pairs:
        bm = next((r["metrics"] for r in all_results
                   if r["condition"] == base_cond), None)
        cm = next((r["metrics"] for r in all_results
                   if r["condition"] == cross_cond), None)
        if bm and cm:
            degradation_rows.append({
                "pair": label,
                "acc_deg": f"{degradation(bm['accuracy'], cm['accuracy']):+.1f}%",
                "pr_deg":  f"{degradation(bm['perfect_rate'], cm['perfect_rate']):+.1f}%",
                "rl_deg":  f"{degradation(bm['rouge_l'], cm['rouge_l']):+.1f}%",
            })
        else:
            degradation_rows.append({
                "pair": label, "acc_deg": "PENDING", "pr_deg": "PENDING", "rl_deg": "PENDING"
            })

    print_degradation_table(
        "Cross-Workload Degradation vs In-Workload Baseline (positive = worse)",
        degradation_rows)

    # ── Save JSON report ──────────────────────────────────────────────────────
    report = {
        "experiment": "RQ-W: Varying-Workload Robustness",
        "model": "SSSD^S4",
        "seq_len": 200,
        "blackout_k": 10,
        "masking": "cm (centered)",
        "train_iters": 10000,
        "conditions": all_results,
        "degradation": degradation_rows,
    }

    out_path = os.path.abspath(args.save_report)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved → {out_path}")


if __name__ == "__main__":
    main()
