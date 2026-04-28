#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# cold_start.slurm
#
# Training-Data-Size / Cold-Start Experiment (NEW — added experiment RQ-CS)
# =========================================================================
# This is an ADDED experiment in response to Reviewer 1's request for a
# training-data-size / cold-start analysis.  It is separate from RQ1-RQ6.
#
# SUBSAMPLING METHOD: uniform random sampling WITHOUT replacement (seed=42).
# This is NOT stratified sampling.
#
# SHARED-VOCABULARY DISCLOSURE:
# The source dataset phpbench_sharedcpu uses the 84-token shared
# CPU-benchmark vocabulary from varying-workload/preprocess_pts_shared_vocab.py.
# The paper's default (§III-B) uses per-dataset frequency-ranked vocabularies.
# This deviation is intentional (enables cross-size comparability and
# transfer-learning) but must be disclosed in any paper text or rebuttal.
# =========================================================================
#
# Train SSSD^S4 from scratch on a subsampled dataset of TRAIN_SIZE sequences
# and evaluate accuracy, perfect rate, and ROUGE-L.
#
# Pre-requisite: run cold-start/preprocess_coldstart.py once to create the
# subsampled dataset directories before submitting these jobs.
#
# Controlled via environment variables:
#   DATASET      Base dataset name (default: phpbench_sharedcpu)
#   TRAIN_SIZE   Number of training sequences (default: 1000)
#   SEQ_LEN      Sequence length (default: 200)
#   MISSING_K    Blackout size   (default: 10)
#   TRAIN_ITERS  Training iterations (default: 10000)
#
# Results are written to:
#   $SCRATCH/cold-start-results/SSSDS4/{DATASET}_n{TRAIN_SIZE}_seq{L}_k{K}/
#     T200_beta00.0001_betaT0.02/metrics.json
#
# Usage:
#   sbatch --export=ALL,DATASET=phpbench_sharedcpu,TRAIN_SIZE=1000 \
#          alliance-canada/cold_start.slurm
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-03:00:00
#SBATCH --job-name=coldstart
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/3}"
DATASET="${DATASET:-phpbench_sharedcpu}"
TRAIN_SIZE="${TRAIN_SIZE:-1000}"
SEQ_LEN="${SEQ_LEN:-200}"
MISSING_K="${MISSING_K:-10}"
TRAIN_ITERS="${TRAIN_ITERS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-64}"

# Subsampled dataset name follows the convention in preprocess_coldstart.py
SUBSAMPLED_DS="${DATASET}_n${TRAIN_SIZE}"

RESULTS_BASE="${RESULTS_BASE:-$SCRATCH/cold-start-results}"
COLD_RESULTS_DIR="$RESULTS_BASE/SSSDS4"
CONFIG_PATH="$SLURM_TMPDIR/coldstart_train_config.json"

echo "========================================================"
echo "  Cold-Start / Training-Size Experiment (RQ-CS)"
echo "  Base dataset  : $DATASET"
echo "  Train size    : $TRAIN_SIZE sequences"
echo "  Subsampled DS : $SUBSAMPLED_DS"
echo "  Seq len       : $SEQ_LEN   Blackout: $MISSING_K"
echo "  Train iters   : $TRAIN_ITERS"
echo "  Results base  : $RESULTS_BASE"
echo "========================================================"

mkdir -p "$RESULTS_BASE" "$COLD_RESULTS_DIR"
mkdir -p "$REPO_DIR/slurm-logs" 2>/dev/null || true

# Verify subsampled dataset exists
TRAIN_NPY="$REPO_DIR/TraceReconstruction-main/Datasets/$SUBSAMPLED_DS/sequence_length_$SEQ_LEN/train.npy"
if [[ ! -f "$TRAIN_NPY" ]]; then
    echo "ERROR: Subsampled training data not found: $TRAIN_NPY"
    echo "Run cold-start/preprocess_coldstart.py first."
    exit 1
fi
echo "Subsampled dataset verified: $TRAIN_NPY"

# ── Environment ──────────────────────────────────────────────────────────────
module purge
module load StdEnv/2023 python/3.11 cuda gcc

virtualenv --no-download "$SLURM_TMPDIR/venv"
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index -r "$REPO_DIR/alliance-canada/requirements-sssd-trace.txt"

export PYTHONPATH="$REPO_DIR/SSSD/src:${PYTHONPATH:-}"
export MAX_JOBS="${SLURM_CPUS_PER_TASK}"

# Build Cauchy kernel extension for S4
cd "$REPO_DIR/SSSD/src/entensions/cauchy"
python setup.py build_ext --inplace

# ── Generate config (train on SUBSAMPLED_DS, test on full DATASET) ────────────
# We test on the full original dataset's test split to keep evaluation
# consistent across all training sizes (the test set is never subsampled).
python "$REPO_DIR/alliance-canada/make_trace_config.py" \
  --dataset      "$SUBSAMPLED_DS" \
  --test-dataset "$DATASET" \
  --seq-len      "$SEQ_LEN" \
  --missing-k    "$MISSING_K" \
  --train-iters  "$TRAIN_ITERS" \
  --batch-size   "$BATCH_SIZE" \
  --results-dir  "$COLD_RESULTS_DIR" \
  --model        2 \
  --masking      "cm" \
  --output       "$CONFIG_PATH"

echo "Config written to $CONFIG_PATH"
cat "$CONFIG_PATH"

# ── Diagnostics ──────────────────────────────────────────────────────────────
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
nvidia-smi

# ── Train from scratch ────────────────────────────────────────────────────────
cd "$REPO_DIR"
echo "--- Training from scratch on $SUBSAMPLED_DS ($TRAIN_SIZE sequences) ---"
python "$REPO_DIR/SSSD/src/train.py" -c "$CONFIG_PATH"

# ── Inference on full test set ────────────────────────────────────────────────
echo "--- Inference (test on full $DATASET test set) ---"
python "$REPO_DIR/SSSD/src/inference.py" -c "$CONFIG_PATH" -ckpt_iter max

# ── Evaluate ──────────────────────────────────────────────────────────────────
echo "--- Evaluation ---"
# Config uses test-dataset != dataset because SUBSAMPLED_DS != DATASET
# → make_trace_config.py sets run_label = "{SUBSAMPLED_DS}_to_{DATASET}_seq{L}_k{K}"
RUN_LABEL="${SUBSAMPLED_DS}_to_${DATASET}_seq${SEQ_LEN}_k${MISSING_K}"
INFER_OUT="$COLD_RESULTS_DIR/$RUN_LABEL/T200_beta00.0001_betaT0.02"

if [[ ! -d "$INFER_OUT" ]]; then
    echo "ERROR: Expected output directory does not exist: $INFER_OUT"
    exit 1
fi
if [[ ! -f "$INFER_OUT/imputation0.npy" ]]; then
    echo "ERROR: imputation0.npy not found in $INFER_OUT"
    exit 1
fi
echo "Verified output directory: $INFER_OUT"

python "$REPO_DIR/alibaba-trace/evaluate_metrics.py" \
  --results-dir "$INFER_OUT" --save-json
echo "Metrics saved in $INFER_OUT/metrics.json"

SUMMARY_DIR="$RESULTS_BASE/summaries"
mkdir -p "$SUMMARY_DIR"
cp "$INFER_OUT/metrics.json" "$SUMMARY_DIR/coldstart_${DATASET}_n${TRAIN_SIZE}_seq${SEQ_LEN}_k${MISSING_K}.json"
echo "Summary copy: $SUMMARY_DIR/coldstart_${DATASET}_n${TRAIN_SIZE}_seq${SEQ_LEN}_k${MISSING_K}.json"

METRICS=$(cat "$INFER_OUT/metrics.json")
echo "Results: $METRICS"
echo "Job complete: dataset=$DATASET  n_train=$TRAIN_SIZE"
