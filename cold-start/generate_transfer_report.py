"""
generate_transfer_report.py
──────────────────────────────────────────────────────────────────────────────
Generate the RQ-TL (Realistic Transfer-Learning) results report from
metrics.json files produced by the transfer_learning.slurm jobs and the
cold-start scratch-training jobs.

FRAMING NOTE
  This is an ADDED experiment to address the reviewer cold-start concern.
  It is NOT a replacement or reframing of existing RQ5 / RQ6:
    RQ5 (current paper): zero-shot cross-application transfer.
    RQ6 (current paper): one model trained on multiple applications simultaneously.
  RQ-TL adds a third scenario: pretrain on a related source, fine-tune on a
  small target set.  If cited in the paper, frame it as an added experiment.

SHARED-VOCABULARY DISCLOSURE
  Both pybench_sharedcpu (source) and phpbench_sharedcpu (target) use a shared
  84-token CPU-benchmark vocabulary produced by
  varying-workload/preprocess_pts_shared_vocab.py.  The paper's default
  preprocessing (§III-B) builds a separate frequency-ranked vocabulary per
  dataset.  A shared vocabulary is required here so that pretrained embedding
  weights transfer meaningfully across corpora.  This deviation from the paper's
  default protocol must be disclosed when reporting these results.

Experiment design:
  Source (pretrain) : pybench_sharedcpu  — full 10k training run, 84-token
                      shared CPU-benchmark vocabulary.
  Target (fine-tune): phpbench_sharedcpu — small splits at 5 training sizes.
  Comparison        : fine-tuned model vs. scratch-trained model at the same
                      training size (from cold-start-results) and vs. the
                      full-data (10k) reference from RQ-W.

Research question (added, not a reframing of existing RQs):
  "Does pretraining on a related trace source help when target data is scarce?"

Metrics (over masked positions only):
  • Accuracy (%)
  • Perfect Rate (%)
  • ROUGE-L (%)

Analysis:
  • Per-size table: scratch vs fine-tuned (Δ = fine-tune − scratch)
  • Fine-tuning benefit curve
  • Cold-start multiplier: how many fewer sequences needed to match scratch-N

Usage:
  python3 cold-start/generate_transfer_report.py \\
      --cold-start-base $SCRATCH/cold-start-results \\
      --tl-base         $SCRATCH/transfer-learning-results \\
      --varwl-base      $SCRATCH/varying-workload-results \\
      [--seq-len 200] [--missing-k 10] \\
      [--output cold-start/TRANSFER_RESULTS.txt]
"""

import argparse
import json
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
DATASETS    = REPO_ROOT / "TraceReconstruction-main" / "Datasets"

TARGET_DS   = "phpbench_sharedcpu"
SOURCE_DS   = "pybench_sharedcpu"
TRAIN_SIZES = [100, 500, 1000, 2500, 5000]
FULL_SIZE   = 10000

W = 76   # report column width


# ── Path helpers ─────────────────────────────────────────────────────────────
def scratch_dir(cold_base: Path, n: int, seq_len: int, k: int) -> Path:
    """cold-start scratch run dir for phpbench_sharedcpu_n{N}."""
    sub_ds    = f"{TARGET_DS}_n{n}"
    run_label = f"{sub_ds}_to_{TARGET_DS}_seq{seq_len}_k{k}"
    return cold_base / "SSSDS4" / run_label / "T200_beta00.0001_betaT0.02"


def finetune_dir(tl_base: Path, n: int, seq_len: int, k: int) -> Path:
    """Fine-tuned run dir (transfer-learning-results)."""
    sub_ds    = f"{TARGET_DS}_n{n}"
    run_label = f"{sub_ds}_to_{TARGET_DS}_seq{seq_len}_k{k}"
    return tl_base / "SSSDS4" / run_label / "T200_beta00.0001_betaT0.02"


def fulldata_dir(varwl_base: Path, ds: str, seq_len: int, k: int) -> Path:
    """Full-10k RQ-W baseline dir."""
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


def delta_str(ft_val, sc_val) -> str:
    """Signed Δ = fine-tune − scratch.  Positive = fine-tuning helps."""
    if ft_val is None or sc_val is None:
        return "  [?]  "
    d = ft_val - sc_val
    if abs(d) < 0.05:
        return "  ≈0   "
    s = f"{d:+.2f}"
    return s.ljust(7)


def ref_gap(val, ref) -> str:
    if val is None or ref is None:
        return "  [?]"
    d = val - ref
    if abs(d) < 0.05:
        return "  ≈0 "
    return f"{d:+.2f}"


# ── Report generation ─────────────────────────────────────────────────────────
def generate_report(cold_base: Path, tl_base: Path, varwl_base: Path,
                    seq_len: int, k: int,
                    output_path: Path | None = None) -> str:
    lines = []

    def hdr(title: str) -> None:
        lines.append("─" * W)
        lines.append(f"  {title}")
        lines.append("─" * W)

    lines.append("=" * W)
    lines.append("  RQ-TL: Realistic Transfer-Learning Experiment  [ADDED]")
    lines.append("=" * W)
    lines.append(f"  FRAMING: This is a NEW added experiment, NOT a reframing of RQ5/RQ6.")
    lines.append(f"    RQ5 (current paper): zero-shot cross-application transfer.")
    lines.append(f"    RQ6 (current paper): one model trained on multiple apps simultaneously.")
    lines.append(f"    RQ-TL (new):         pretrain on related source, fine-tune on small target.")
    lines.append("")
    lines.append(f"  Source (pretrain) : {SOURCE_DS}  (full {FULL_SIZE} seqs, 84-token shared vocab)")
    lines.append(f"  Target (fine-tune): {TARGET_DS}")
    lines.append(f"  Seq len : {seq_len}   Blackout k: {k}   Masking: center (cm)")
    lines.append(f"  Fine-tune init    : warm-start (source 10000.pkl weights, reset optimizer)")
    lines.append(f"  Fine-tune iters   : 10 000  (equal budget to scratch training)")
    lines.append("")
    lines.append("  METHODOLOGICAL NOTE — Shared vocabulary")
    lines.append("  ────────────────────────────────────────")
    lines.append(f"  Both {SOURCE_DS} and {TARGET_DS} use a SHARED")
    lines.append(f"  84-token CPU-benchmark vocabulary built by")
    lines.append(f"  varying-workload/preprocess_pts_shared_vocab.py.")
    lines.append(f"  The paper's default preprocessing (§III-B) assigns each dataset")
    lines.append(f"  its own frequency-ranked vocabulary.  The shared vocabulary is a")
    lines.append(f"  deliberate and necessary deviation: without it the pretrained")
    lines.append(f"  embedding weights would encode different syscalls on source vs")
    lines.append(f"  target, making warm-start weights meaningless.  Disclose this in")
    lines.append(f"  any paper text.")
    lines.append("")
    lines.append(f"  Research question (added): Does pre-training on a RELATED trace")
    lines.append(f"  source reduce the number of TARGET sequences needed for high")
    lines.append(f"  reconstruction quality?")
    lines.append("")

    # ── Load all metrics ──────────────────────────────────────────────────
    scratch_m  = {}
    finetune_m = {}
    for n in TRAIN_SIZES:
        scratch_m[n]  = load_metrics(scratch_dir(cold_base, n, seq_len, k))
        finetune_m[n] = load_metrics(finetune_dir(tl_base,  n, seq_len, k))

    full_ref  = load_metrics(fulldata_dir(varwl_base, TARGET_DS, seq_len, k))
    majority  = load_majority_baseline(TARGET_DS, seq_len)

    ref_acc   = full_ref.get("accuracy")     if full_ref else None
    ref_perf  = full_ref.get("perfect_rate") if full_ref else None
    ref_rouge = full_ref.get("rouge_l")      if full_ref else None

    # ── Comparison table ──────────────────────────────────────────────────
    lines.append("")
    hdr("SCRATCH vs FINE-TUNED COMPARISON TABLE")
    lines.append(
        f"{'N_train':>8}  "
        f"{'──────── Scratch ────────':^26}  "
        f"{'──── Fine-tuned (TL) ────':^26}  "
        f"{'Δ Acc':>6}"
    )
    lines.append(
        f"{'':>8}  {'Acc':>7} {'Perf':>7} {'ROUGE':>7}  "
        f"{'Acc':>7} {'Perf':>7} {'ROUGE':>7}  "
        f"{'ft−sc':>6}"
    )
    lines.append("-" * W)

    for n in TRAIN_SIZES:
        sc = scratch_m.get(n)
        ft = finetune_m.get(n)

        sc_acc   = sc.get("accuracy")     if sc else None
        sc_perf  = sc.get("perfect_rate") if sc else None
        sc_rouge = sc.get("rouge_l")      if sc else None

        ft_acc   = ft.get("accuracy")     if ft else None
        ft_perf  = ft.get("perfect_rate") if ft else None
        ft_rouge = ft.get("rouge_l")      if ft else None

        d_acc    = delta_str(ft_acc, sc_acc)

        flag  = ""
        if sc is None:
            flag += " ⚠sc"
        if ft is None:
            flag += " ⚠ft"

        lines.append(
            f"{n:>8}  "
            f"{fmt(sc_acc):>7} {fmt(sc_perf):>7} {fmt(sc_rouge):>7}  "
            f"{fmt(ft_acc):>7} {fmt(ft_perf):>7} {fmt(ft_rouge):>7}  "
            f"{d_acc:>6}{flag}"
        )

    # Full-data reference row
    lines.append("-" * W)
    lines.append(
        f"{'10000':>8}  "
        f"{fmt(ref_acc):>7} {fmt(ref_perf):>7} {fmt(ref_rouge):>7}  "
        f"{'[ref]':>7} {'[ref]':>7} {'[ref]':>7}  "
        f"{'—':>6}  ← RQ-W full-data baseline"
    )
    lines.append("")
    lines.append(
        "  Δ Acc: positive = fine-tuning helps; negative = scratch is better"
    )
    if majority:
        lines.append(
            f"  Majority baseline: acc={fmt(majority.get('accuracy'))}%"
            f"  rouge={fmt(majority.get('rouge_l'))}%"
        )

    # ── Fine-tuning benefit curve (ASCII) ─────────────────────────────────
    lines.append("")
    hdr("FINE-TUNING BENEFIT CURVE  (Accuracy % vs Training Size)")

    have_sc = [(n, scratch_m[n])  for n in TRAIN_SIZES if scratch_m.get(n)]
    have_ft = [(n, finetune_m[n]) for n in TRAIN_SIZES if finetune_m.get(n)]

    if have_sc or have_ft:
        all_vals = (
            [m["accuracy"] for _, m in have_sc if m.get("accuracy")] +
            [m["accuracy"] for _, m in have_ft if m.get("accuracy")]
        )
        if ref_acc:
            all_vals.append(ref_acc)
        max_acc = max(all_vals) if all_vals else 100.0

        lines.append(f"  {'N':>6}  {'Scratch':>8}  {'Fine-tuned':>11}  {'Δ':>6}")
        lines.append(f"  {'-'*6}  {'-'*8}  {'-'*11}  {'-'*6}")
        for n in TRAIN_SIZES:
            sc = scratch_m.get(n)
            ft = finetune_m.get(n)
            sc_a = sc.get("accuracy") if sc else None
            ft_a = ft.get("accuracy") if ft else None
            delta = f"{ft_a - sc_a:+.2f}" if sc_a is not None and ft_a is not None else "[?]"
            lines.append(
                f"  {n:>6}  {fmt(sc_a):>8}%  {fmt(ft_a):>10}%  {delta:>6}"
            )
        if ref_acc:
            lines.append(
                f"  {'10000':>6}  {fmt(ref_acc):>8}%  "
                f"{'[ref]':>11}  {'—':>6}  ← RQ-W full-data"
            )
    else:
        lines.append("  [No completed runs yet]")

    # ── Cold-start multiplier analysis ───────────────────────────────────
    lines.append("")
    hdr("COLD-START MULTIPLIER ANALYSIS")
    lines.append("  'With N fine-tuned sequences, the model matches scratch at M sequences.'")
    lines.append("  A high multiplier = pretraining is especially beneficial in low-data regime.")
    lines.append("")

    # For each fine-tune N, find the smallest scratch N' with similar accuracy
    ft_accs = {n: finetune_m[n].get("accuracy")
               for n in TRAIN_SIZES if finetune_m.get(n)}
    sc_accs = {n: scratch_m[n].get("accuracy")
               for n in TRAIN_SIZES if scratch_m.get(n)}

    if ft_accs and sc_accs:
        sc_sorted = sorted(sc_accs.items())  # [(n, acc), ...]
        for ft_n, ft_acc in sorted(ft_accs.items()):
            if ft_acc is None:
                continue
            # Find the smallest scratch N whose accuracy >= ft_acc
            equiv_sc = None
            for sc_n, sc_acc in sc_sorted:
                if sc_acc is not None and sc_acc >= ft_acc:
                    equiv_sc = sc_n
                    break
            if equiv_sc is not None:
                mult = equiv_sc / ft_n
                lines.append(
                    f"  Fine-tune N={ft_n:>5} ({fmt(ft_acc)}% acc) ≈ "
                    f"scratch N={equiv_sc:>5} ({fmt(sc_accs[equiv_sc])}% acc)  "
                    f"→ {mult:.1f}× multiplier"
                )
            else:
                lines.append(
                    f"  Fine-tune N={ft_n:>5} ({fmt(ft_acc)}% acc) "
                    f"exceeds all tested scratch sizes."
                )
        lines.append("")
        lines.append("  Higher multiplier = bigger benefit from pretraining.")
        lines.append("  A multiplier > 1 confirms that fine-tuning is especially")
        lines.append("  useful in the cold-start (very low data) regime.")
    else:
        lines.append("  [Insufficient data to compute multipliers]")

    # ── Gap to full-data reference ────────────────────────────────────────
    lines.append("")
    hdr("GAP TO FULL-DATA REFERENCE  (N=10k, pts-phpbench-baseline from RQ-W)")
    if ref_acc:
        lines.append(
            f"  {'N_train':>8}  {'Scratch Δ vs 10k':>17}  {'Fine-tune Δ vs 10k':>20}"
        )
        lines.append("-" * W)
        for n in TRAIN_SIZES:
            sc = scratch_m.get(n)
            ft = finetune_m.get(n)
            sc_a = sc.get("accuracy") if sc else None
            ft_a = ft.get("accuracy") if ft else None
            sc_gap = ref_gap(sc_a, ref_acc)
            ft_gap = ref_gap(ft_a, ref_acc)
            lines.append(
                f"  {n:>8}  {sc_gap:>17}  {ft_gap:>20}"
            )
        lines.append("")
        lines.append(f"  Full-data (N=10k) reference: acc={fmt(ref_acc)}%  "
                     f"perf={fmt(ref_perf)}%  rouge={fmt(ref_rouge)}%")
        lines.append("  Negative Δ = the small-data model runs below full-data performance.")
    else:
        lines.append(
            "  [Full-data reference not found. Check:"
            f"\n   {fulldata_dir(varwl_base, TARGET_DS, seq_len, k)}]"
        )

    # ── Missing results ───────────────────────────────────────────────────
    missing_scratch  = [n for n in TRAIN_SIZES if scratch_m.get(n) is None]
    missing_finetune = [n for n in TRAIN_SIZES if finetune_m.get(n) is None]

    if missing_scratch or missing_finetune or full_ref is None:
        lines.append("")
        hdr("INCOMPLETE / MISSING RUNS")
        for n in missing_scratch:
            lines.append(f"  SCRATCH  N={n}: {scratch_dir(cold_base, n, seq_len, k)}")
        for n in missing_finetune:
            lines.append(f"  FINETUNE N={n}: {finetune_dir(tl_base,  n, seq_len, k)}")
        if full_ref is None:
            lines.append(
                f"  REF 10k : {fulldata_dir(varwl_base, TARGET_DS, seq_len, k)}"
            )
        lines.append("")
        lines.append("  Re-submit missing jobs or wait for SLURM completion.")
        lines.append("  Scratch  jobs: bash alliance-canada/submit_cold_start.sh")
        lines.append("  Finetune jobs: bash alliance-canada/submit_transfer_learning.sh")

    # ── Raw metrics dump ──────────────────────────────────────────────────
    lines.append("")
    hdr("RAW METRICS  (from metrics.json files)")
    for n in TRAIN_SIZES:
        sc = scratch_m.get(n)
        ft = finetune_m.get(n)
        lines.append(f"  N={n}")
        if sc:
            lines.append(f"    scratch  : acc={sc.get('accuracy')}  "
                         f"perf={sc.get('perfect_rate')}  rouge={sc.get('rouge_l')}  "
                         f"n_seq={sc.get('n_sequences')}")
        else:
            lines.append(f"    scratch  : [NOT YET COMPUTED]")
        if ft:
            lines.append(f"    finetune : acc={ft.get('accuracy')}  "
                         f"perf={ft.get('perfect_rate')}  rouge={ft.get('rouge_l')}  "
                         f"n_seq={ft.get('n_sequences')}")
        else:
            lines.append(f"    finetune : [NOT YET COMPUTED]")

    lines.append("")
    if full_ref:
        lines.append(f"  N=10000 (ref, RQ-W pts-phpbench-baseline):")
        lines.append(f"    acc={full_ref.get('accuracy')}  "
                     f"perf={full_ref.get('perfect_rate')}  "
                     f"rouge={full_ref.get('rouge_l')}  "
                     f"n_seq={full_ref.get('n_sequences')}")
    else:
        lines.append(f"  N=10000 (ref): [NOT YET COMPUTED]")

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
        description="Generate RQ-TL transfer-learning report."
    )
    parser.add_argument("--cold-start-base", required=True,
                        help="Path to $SCRATCH/cold-start-results "
                             "(scratch-trained small-data runs)")
    parser.add_argument("--tl-base", required=True,
                        help="Path to $SCRATCH/transfer-learning-results "
                             "(fine-tuned runs)")
    parser.add_argument("--varwl-base", required=True,
                        help="Path to $SCRATCH/varying-workload-results "
                             "(10k full-data reference)")
    parser.add_argument("--seq-len",   type=int, default=200)
    parser.add_argument("--missing-k", type=int, default=10)
    parser.add_argument("--output",
                        help="Output file path (default: print to stdout)")
    args = parser.parse_args()

    generate_report(
        cold_base   = Path(args.cold_start_base),
        tl_base     = Path(args.tl_base),
        varwl_base  = Path(args.varwl_base),
        seq_len     = args.seq_len,
        k           = args.missing_k,
        output_path = Path(args.output) if args.output else None,
    )
