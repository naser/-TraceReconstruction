#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# varying_workload.slurm
#
# Varying-Workload Experiment (RQ-W)
# Train SSSD^S4 on one workload variant and test on another (or the same) to
# measure cross-workload generalisation and degradation.
#
# Controlled via environment variables:
#   TRAIN_DS     Training dataset name (default: elk_clean)
#   TEST_DS      Test dataset name     (default: elk_noisy)
#   CONDITION    Human-readable label  (default: clean_to_noisy)
#   SEQ_LEN      default 200
#   MISSING_K    default 10
#   TRAIN_ITERS  default 10000
#
# Usage examples:
#   # Cross-workload: train on clean, test on noisy
#   sbatch --export=ALL,TRAIN_DS=elk_clean,TEST_DS=elk_noisy,CONDITION=clean_to_noisy \
#          alliance-canada/varying_workload.slurm
#
#   # In-workload baseline: train and test on clean
#   sbatch --export=ALL,TRAIN_DS=elk_clean,TEST_DS=elk_clean,CONDITION=clean_baseline \
#          alliance-canada/varying_workload.slurm
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-03:00:00
#SBATCH --job-name=varwl
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/3}"
TRAIN_DS="${TRAIN_DS:-elk_clean}"
TEST_DS="${TEST_DS:-elk_noisy}"
CONDITION="${CONDITION:-${TRAIN_DS}_to_${TEST_DS}}"
SEQ_LEN="${SEQ_LEN:-200}"
MISSING_K="${MISSING_K:-10}"
TRAIN_ITERS="${TRAIN_ITERS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-64}"

RESULTS_BASE="${RESULTS_BASE:-$SCRATCH/varying-workload-results}"
TRAIN_RESULTS_DIR="$RESULTS_BASE/SSSDS4"
CONFIG_PATH="$SLURM_TMPDIR/varwl_train_config.json"

echo "========================================================"
echo "  Varying-Workload Experiment (RQ-W)"
echo "  Train dataset : $TRAIN_DS"
echo "  Test  dataset : $TEST_DS"
echo "  Condition     : $CONDITION"
echo "  Seq len       : $SEQ_LEN   Blackout: $MISSING_K"
echo "  Train iters   : $TRAIN_ITERS"
echo "  Results base  : $RESULTS_BASE"
echo "========================================================"

mkdir -p "$RESULTS_BASE" "$TRAIN_RESULTS_DIR"
mkdir -p "$REPO_DIR/slurm-logs" 2>/dev/null || true

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

# ── Generate config (train on TRAIN_DS, test on TEST_DS) ─────────────────────
python "$REPO_DIR/alliance-canada/make_trace_config.py" \
  --dataset      "$TRAIN_DS" \
  --test-dataset "$TEST_DS" \
  --seq-len      "$SEQ_LEN" \
  --missing-k    "$MISSING_K" \
  --train-iters  "$TRAIN_ITERS" \
  --batch-size   "$BATCH_SIZE" \
  --results-dir  "$TRAIN_RESULTS_DIR" \
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

# ── Train on TRAIN_DS ─────────────────────────────────────────────────────────
cd "$REPO_DIR"
echo "--- Training on $TRAIN_DS ---"
python "$REPO_DIR/SSSD/src/train.py" -c "$CONFIG_PATH"

# ── Inference on TEST_DS ──────────────────────────────────────────────────────
echo "--- Inference (test on $TEST_DS) ---"
python "$REPO_DIR/SSSD/src/inference.py" -c "$CONFIG_PATH" -ckpt_iter max

# ── Evaluate ──────────────────────────────────────────────────────────────────
echo "--- Evaluation ---"

# Derive the exact output directory the config wrote to (mirrors make_trace_config.py logic)
if [[ "$TEST_DS" != "$TRAIN_DS" ]]; then
    RUN_LABEL="${TRAIN_DS}_to_${TEST_DS}_seq${SEQ_LEN}_k${MISSING_K}"
else
    RUN_LABEL="${TRAIN_DS}_seq${SEQ_LEN}_k${MISSING_K}"
fi
INFER_OUT="$TRAIN_RESULTS_DIR/$RUN_LABEL/T200_beta00.0001_betaT0.02"

if [[ ! -d "$INFER_OUT" ]]; then
    echo "ERROR: Expected output directory does not exist: $INFER_OUT"
    exit 1
fi

# Sanity check: verify that original0.npy was produced from the TEST dataset
# by confirming the file exists (it was saved by the fixed inference.py to output_directory)
if [[ ! -f "$INFER_OUT/imputation0.npy" ]]; then
    echo "ERROR: imputation0.npy not found in $INFER_OUT — inference may have failed"
    exit 1
fi
echo "Verified output directory: $INFER_OUT"

python "$REPO_DIR/alibaba-trace/evaluate_metrics.py" \
  --results-dir "$INFER_OUT" --save-json
echo "Metrics saved in $INFER_OUT/metrics.json"

# Copy metrics to a named summary file for easy aggregation
SUMMARY_DIR="$RESULTS_BASE/summaries"
mkdir -p "$SUMMARY_DIR"
cp "$INFER_OUT/metrics.json" "$SUMMARY_DIR/${CONDITION}_seq${SEQ_LEN}_k${MISSING_K}.json"
echo "Summary copy: $SUMMARY_DIR/${CONDITION}_seq${SEQ_LEN}_k${MISSING_K}.json"

echo "Job complete: train=$TRAIN_DS  test=$TEST_DS  condition=$CONDITION"
