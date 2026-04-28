"""
evaluate_metrics.py  ─  Compute paper metrics from SSSD inference outputs.

Usage:
    python evaluate_metrics.py --results-dir /path/to/results_dir

The results dir must contain: imputation*.npy, original*.npy, mask*.npy
(exactly matching what the SSSD inference.py produces).

Outputs:
    - Prints Accuracy, Perfect Rate, ROUGE-L
    - Saves metrics.json in the results dir
"""

import argparse, json, os, glob
import numpy as np


# ── LCS-based ROUGE-L (identical to paper definition) ──────────────────────
def lcs_length(a, b):
    """O(m*n) DP for LCS length between two integer lists."""
    m, n = len(a), len(b)
    # space-optimised 2-row DP
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
    """ROUGE-L = 100 * LCS(orig, recon) / len(orig)"""
    n = len(original)
    if n == 0:
        return 100.0
    return 100.0 * lcs_length(list(original), list(reconstructed)) / n


# ── Load all batch files in the directory ───────────────────────────────────
def load_all_batches(results_dir):
    imp_files  = sorted(glob.glob(os.path.join(results_dir, "imputation*.npy")))
    orig_files = sorted(glob.glob(os.path.join(results_dir, "original*.npy")))
    mask_files = sorted(glob.glob(os.path.join(results_dir, "mask*.npy")))

    assert len(imp_files) == len(orig_files) == len(mask_files) > 0, (
        f"Missing npy files in {results_dir}")

    imputations, originals, masks = [], [], []
    for i_f, o_f, m_f in zip(imp_files, orig_files, mask_files):
        imputations.append(np.load(i_f))
        originals.append(np.load(o_f))
        masks.append(np.load(m_f))

    return (np.concatenate(imputations, axis=0),
            np.concatenate(originals,   axis=0),
            np.concatenate(masks,        axis=0))


# ── Core evaluation ─────────────────────────────────────────────────────────
def evaluate(results_dir, verbose=True):
    """
    Returns dict with keys: accuracy, perfect_rate, rouge_l
    Expects arrays of shape (N, C, T) where C==1.
    The mask is 1 for OBSERVED positions, 0 for MISSING positions.
    We evaluate only over the missing (masked-out) positions.
    """
    imp, orig, mask = load_all_batches(results_dir)

    # shapes: (N, 1, T)  → squeeze channel dim
    imp  = imp [:, 0, :]   # (N, T)
    orig = orig[:, 0, :]   # (N, T)
    mask = mask[:, 0, :]   # (N, T)  1=observed, 0=missing

    # Round float predictions to nearest integer (event ID)
    imp_round = np.round(imp).astype(np.int32)
    orig_int  = orig.astype(np.int32)

    missing_mask = (mask == 0)   # True where events were removed

    n_seqs = orig.shape[0]
    per_seq_acc   = []
    per_seq_rouge = []
    perfect = 0

    for i in range(n_seqs):
        miss_idx = np.where(missing_mask[i])[0]
        if len(miss_idx) == 0:
            continue

        orig_seg  = orig_int [i, miss_idx]
        pred_seg  = imp_round[i, miss_idx]

        # sequence-level accuracy (fraction correct)
        acc = float(np.sum(orig_seg == pred_seg)) / len(orig_seg)
        per_seq_acc.append(acc)

        # perfect rate: entire segment is correct
        if acc == 1.0:
            perfect += 1

        # ROUGE-L on the missing segment
        rl = rouge_l(orig_seg, pred_seg)
        per_seq_rouge.append(rl)

    n = len(per_seq_acc)
    if n == 0:
        return {"accuracy": 0.0, "perfect_rate": 0.0, "rouge_l": 0.0}

    result = {
        "accuracy":     round(100.0 * np.mean(per_seq_acc), 2),
        "perfect_rate": round(100.0 * perfect / n,           2),
        "rouge_l":      round(np.mean(per_seq_rouge),         2),
        "n_sequences":  n,
    }

    if verbose:
        print(f"  Results dir : {results_dir}")
        print(f"  N sequences : {result['n_sequences']}")
        print(f"  Accuracy    : {result['accuracy']:.2f}%")
        print(f"  Perfect Rate: {result['perfect_rate']:.2f}%")
        print(f"  ROUGE-L     : {result['rouge_l']:.2f}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing imputation*.npy files")
    parser.add_argument("--save-json", action="store_true",
                        help="Save metrics.json inside the results dir")
    args = parser.parse_args()

    metrics = evaluate(args.results_dir)

    if args.save_json:
        out = os.path.join(args.results_dir, "metrics.json")
        with open(out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved → {out}")


if __name__ == "__main__":
    main()
