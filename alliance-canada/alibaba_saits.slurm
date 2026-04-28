#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# alibaba_saits.slurm
#
# Prepare h5 dataset, train SAITS, run inference, and evaluate on Alibaba
# microservice trace data.  Produces output in the same format as the SSSD
# inference so that evaluate_metrics.py can process results uniformly.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --job-name=alibaba-saits
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/3}"
SEQ_LEN="${SEQ_LEN:-200}"
MISSING_K="${MISSING_K:-10}"
EPOCHS="${EPOCHS:-200}"

RESULTS_BASE="${RESULTS_BASE:-$SCRATCH/alibaba-trace-results}"
RESULTS_DIR="$RESULTS_BASE/SAITS/alibaba_seq${SEQ_LEN}_k${MISSING_K}"
H5_PATH="$SLURM_TMPDIR/alibaba_saits.h5"

DATA_DIR="$REPO_DIR/TraceReconstruction-main/Datasets/alibaba/sequence_length_${SEQ_LEN}"

echo "========================================"
echo "  SAITS on Alibaba"
echo "  Seq len   : $SEQ_LEN"
echo "  Blackout k: $MISSING_K"
echo "  Epochs    : $EPOCHS"
echo "  Results   : $RESULTS_DIR"
echo "========================================"

mkdir -p "$RESULTS_DIR"
mkdir -p "$REPO_DIR/slurm-logs" 2>/dev/null || true

# ── Environment ──────────────────────────────────────────────────────────────
module purge
module load StdEnv/2023 python/3.11 cuda gcc scipy-stack

virtualenv --no-download "$SLURM_TMPDIR/venv"
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index torch h5py scikit-learn 2>/dev/null || \
  pip install     torch h5py scikit-learn

# ── Prepare h5 dataset ───────────────────────────────────────────────────────
echo "--- Preparing SAITS h5 dataset ---"
python "$REPO_DIR/alibaba-trace/prepare_saits_dataset.py" \
  --npy-train "$DATA_DIR/train.npy" \
  --npy-test  "$DATA_DIR/test.npy"  \
  --blackout  "$MISSING_K"          \
  --output    "$H5_PATH"

# ── Train + Infer ─────────────────────────────────────────────────────────────
echo "--- Training + Inference ---"
python "$REPO_DIR/alibaba-trace/saits_train_infer.py" \
  --h5-path     "$H5_PATH"     \
  --results-dir "$RESULTS_DIR" \
  --seq-len     "$SEQ_LEN"     \
  --blackout    "$MISSING_K"   \
  --epochs      "$EPOCHS"

# ── Evaluate ──────────────────────────────────────────────────────────────────
echo "--- Evaluation ---"
python "$REPO_DIR/alibaba-trace/evaluate_metrics.py" \
  --results-dir "$RESULTS_DIR" --save-json

echo "Job complete."
