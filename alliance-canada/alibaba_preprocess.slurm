#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# alibaba_preprocess.slurm
#
# Regenerate Alibaba .npy datasets with a random trace-level 80/20 split.
# Must complete BEFORE any training jobs are submitted.
# Run once; all four model jobs depend on its successful completion.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --job-name=alibaba-preprocess
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/3}"
DATASET_DIR="$REPO_DIR/TraceReconstruction-main/Datasets/alibaba"

echo "========================================"
echo "  Regenerating Alibaba .npy datasets"
echo "  Repo dir : $REPO_DIR"
echo "  Dataset  : $DATASET_DIR"
echo "========================================"

mkdir -p "$REPO_DIR/slurm-logs" 2>/dev/null || true

# ── Environment ──────────────────────────────────────────────────────────────
module purge
module load StdEnv/2023 python/3.11 scipy-stack

# ── Remove stale datasets ────────────────────────────────────────────────────
echo "Removing old .npy files to force clean regeneration …"
find "$DATASET_DIR" -name "*.npy" -delete
find "$DATASET_DIR" -name "training" -delete
find "$DATASET_DIR" -name "testing" -delete

# ── Regenerate ───────────────────────────────────────────────────────────────
echo "Running preprocess_alibaba.py (random trace-level split, seed=42) …"
python "$DATASET_DIR/preprocess_alibaba.py"

echo ""
echo "=== Dataset sizes after regeneration ==="
for seq in 50 100 150 200; do
    DIR="$DATASET_DIR/sequence_length_${seq}"
    echo "  seq=$seq:"
    python - <<PY
import numpy as np
tr = np.load("$DIR/train.npy")
te = np.load("$DIR/test.npy")
print(f"    train {tr.shape}  test {te.shape}  "
      f"token range [{int(tr.min())}, {int(tr.max())}]  "
      f"train unique {len(set(tr.flatten()))}  test unique {len(set(te.flatten()))}")
PY
done

echo "Preprocessing complete."
