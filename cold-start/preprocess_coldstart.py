"""
preprocess_coldstart.py
──────────────────────────────────────────────────────────────────────────────
Create subsampled training datasets for the cold-start / training-data-size
experiment.  Reads an existing fully-preprocessed dataset and writes N-sized
sub-versions with different training set sizes.

The test set is always the full 500-sequence test split from the source
dataset.  The vocabulary is identical — no re-encoding is required because we
are simply taking a random subsample of the train.npy rows.

Design choices (aligned with the paper):
  • Source dataset : phpbench_sharedcpu  (84-token shared CPU-benchmark vocab;
                     already used in the RQ-W experiment as pts-phpbench-baseline).
  • Training sizes : 100, 500, 1 000, 2 500, 5 000  (10 000 = full RQ-W run).
  • seq_len        : 200  (paper default for all RQ-2+ experiments).
  • Sampling       : uniform random sampling WITHOUT replacement from the
                     10 000-sequence pool, fixed seed=42 for reproducibility.
                     NOTE: this is NOT stratified sampling — sequences are
                     drawn uniformly at random regardless of content.
  • Output dirs    : TraceReconstruction-main/Datasets/{dataset}_n{N}/
                       sequence_length_200/train.npy   ← N subsampled seqs
                       sequence_length_200/test.npy    ← full 500-seq test set
                       sequence_length_200/metadata.json ← source metadata +
                                                           added n_train field

Usage:
  python3 cold-start/preprocess_coldstart.py [--dataset phpbench_sharedcpu] \
          [--sizes 100 500 1000 2500 5000] [--seq-len 200] [--seed 42]

Run from the workspace root on any machine (CPU-only, fast).
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

# ── Defaults ─────────────────────────────────────────────────────────────────
REPO_ROOT       = Path(__file__).resolve().parent.parent
DATASETS        = REPO_ROOT / "TraceReconstruction-main" / "Datasets"
DEFAULT_DATASET = "phpbench_sharedcpu"
DEFAULT_SIZES   = [100, 500, 1000, 2500, 5000]
DEFAULT_SEQ_LEN = 200
DEFAULT_SEED    = 42


def subsample(train_arr: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Return n sequences chosen uniformly at random (without replacement) from
    train_arr.  If n >= len(train_arr), return the full array.
    """
    total = len(train_arr)
    if n >= total:
        print(f"  Requested {n} >= available {total}; returning full array.")
        return train_arr
    idx = rng.choice(total, size=n, replace=False)
    idx.sort()          # preserve temporal order for reproducibility checks
    return train_arr[idx]


def build_subsampled_dataset(source_ds: str, n: int, seq_len: int,
                             rng: np.random.Generator) -> None:
    src_dir  = DATASETS / source_ds / f"sequence_length_{seq_len}"
    dest_ds  = f"{source_ds}_n{n}"
    dest_dir = DATASETS / dest_ds / f"sequence_length_{seq_len}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # ── Training data ─────────────────────────────────────────────────────
    train_src = src_dir / "train.npy"
    if not train_src.exists():
        raise FileNotFoundError(f"Source training data not found: {train_src}")
    train_all = np.load(str(train_src))        # (10000, seq_len, 1) channels-last
    assert train_all.ndim == 3 and train_all.shape[1] == seq_len, (
        f"Unexpected train.npy shape {train_all.shape}")
    train_sub = subsample(train_all, n, rng)
    np.save(str(dest_dir / "train.npy"), train_sub)
    print(f"  {dest_ds}/train.npy  shape={train_sub.shape}")

    # ── Test data (always full 500-sequence split) ────────────────────────
    test_src = src_dir / "test.npy"
    if not test_src.exists():
        raise FileNotFoundError(f"Source test data not found: {test_src}")
    shutil.copy2(str(test_src), str(dest_dir / "test.npy"))
    test_arr = np.load(str(test_src))
    print(f"  {dest_ds}/test.npy   shape={test_arr.shape}  (full split, copied)")

    # ── Metadata ──────────────────────────────────────────────────────────
    meta_src = src_dir / "metadata.json"
    if meta_src.exists():
        meta = json.loads(meta_src.read_text())
    else:
        meta = {}
    meta["source_dataset"]        = source_ds
    meta["n_train"]               = int(n)
    meta["n_train_source"]        = int(len(train_all))
    meta["seq_len"]               = int(seq_len)
    meta["random_seed"]           = DEFAULT_SEED
    meta["subsampling_method"]    = "uniform_random_without_replacement"
    meta["experiment"]            = "cold-start"
    (dest_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  {dest_ds}/metadata.json written")


def main():
    parser = argparse.ArgumentParser(
        description="Create subsampled training datasets for the cold-start experiment."
    )
    parser.add_argument("--dataset",  default=DEFAULT_DATASET,
                        help="Source dataset name under TraceReconstruction-main/Datasets/")
    parser.add_argument("--sizes",    nargs="+", type=int, default=DEFAULT_SIZES,
                        help="Training set sizes to create (default: 100 500 1000 2500 5000)")
    parser.add_argument("--seq-len",  type=int, default=DEFAULT_SEQ_LEN,
                        help="Sequence length sub-directory to use (default: 200)")
    parser.add_argument("--seed",     type=int, default=DEFAULT_SEED,
                        help="Random seed for reproducible subsampling (default: 42)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Source dataset : {args.dataset}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Training sizes : {args.sizes}")
    print(f"Random seed    : {args.seed}")
    print(f"Output root    : {DATASETS}")
    print()

    src_dir = DATASETS / args.dataset / f"sequence_length_{args.seq_len}"
    if not src_dir.exists():
        raise FileNotFoundError(
            f"Source dataset directory not found: {src_dir}\n"
            "Run varying-workload/preprocess_pts_shared_vocab.py first."
        )

    for n in sorted(args.sizes):
        print(f"Creating {args.dataset}_n{n} ...")
        build_subsampled_dataset(args.dataset, n, args.seq_len, rng)
        print()

    print(f"Done. Created {len(args.sizes)} subsampled datasets.")
    print()
    print("Next steps:")
    print("  1. Submit cold-start training jobs:")
    print("       bash alliance-canada/submit_cold_start.sh")
    print("  2. Submit transfer-learning fine-tuning jobs:")
    print("       bash alliance-canada/submit_transfer_learning.sh")


if __name__ == "__main__":
    main()
