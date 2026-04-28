#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# alibaba_cross_app.slurm
#
# Cross-application transfer experiment: train on one dataset, test on another.
# Either:
#   TRAIN=alibaba  TEST=apache|PLAID|elk
#   TRAIN=apache   TEST=alibaba
#   TRAIN=PLAID    TEST=alibaba
#   TRAIN=elk      TEST=alibaba
#
# Set TRAIN and TEST as environment variables before submission.
# The script reuses an already-trained SSSD^S4 checkpoint if available;
# otherwise trains from scratch on TRAIN dataset.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --job-name=alibaba-xapp
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/3}"
TRAIN="${TRAIN:-alibaba}"
TEST="${TEST:-apache}"
SEQ_LEN=200
MISSING_K=10
TRAIN_ITERS="${TRAIN_ITERS:-10000}"
BATCH_SIZE=64

RESULTS_BASE="${RESULTS_BASE:-$SCRATCH/alibaba-trace-results}"
TRAIN_RESULTS_DIR="$RESULTS_BASE/SSSDS4"
CONFIG_PATH="$SLURM_TMPDIR/xapp_train_config.json"
EVAL_DIR="$RESULTS_BASE/cross_app/${TRAIN}_to_${TEST}"

echo "========================================"
echo "  Cross-app transfer"
echo "  Train on : $TRAIN"
echo "  Test on  : $TEST"
echo "  Seq len  : $SEQ_LEN  Blackout: $MISSING_K"
echo "========================================"

mkdir -p "$EVAL_DIR"
mkdir -p "$REPO_DIR/slurm-logs" 2>/dev/null || true

# ── Environment ──────────────────────────────────────────────────────────────
module purge
module load StdEnv/2023 python/3.11 cuda gcc

virtualenv --no-download "$SLURM_TMPDIR/venv"
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index -r "$REPO_DIR/alliance-canada/requirements-sssd-trace.txt"

export PYTHONPATH="$REPO_DIR/SSSD/src:${PYTHONPATH:-}"
export MAX_JOBS="${SLURM_CPUS_PER_TASK}"

cd "$REPO_DIR/SSSD/src/entensions/cauchy"
python setup.py build_ext --inplace

# ── Generate config for TRAINING dataset ─────────────────────────────────────
python "$REPO_DIR/alliance-canada/make_trace_config.py" \
  --dataset     "$TRAIN" \
  --seq-len     "$SEQ_LEN" \
  --missing-k   "$MISSING_K" \
  --train-iters "$TRAIN_ITERS" \
  --batch-size  "$BATCH_SIZE" \
  --results-dir "$TRAIN_RESULTS_DIR" \
  --model       2 \
  --masking     "cm" \
  --output      "$CONFIG_PATH"

# ── Train on TRAIN dataset (if not already done) ─────────────────────────────
cd "$REPO_DIR"
echo "--- Training on $TRAIN ---"
python "$REPO_DIR/SSSD/src/train.py" -c "$CONFIG_PATH"

# ── Inference on TEST dataset ─────────────────────────────────────────────────
# Create a modified config that uses the TRAIN checkpoint but TEST data
TEST_DATA_DIR="$REPO_DIR/TraceReconstruction-main/Datasets/${TEST}/sequence_length_${SEQ_LEN}"
XAPP_CONFIG="$SLURM_TMPDIR/xapp_infer_config.json"

python - <<PYEOF
import json
with open("$CONFIG_PATH") as f:
    cfg = json.load(f)
# Point test data to TEST dataset; keep checkpoint from TRAIN
cfg["trainset_config"]["test_data_path"] = "$TEST_DATA_DIR/test.npy"
cfg["gen_config"]["output_directory"] = "$EVAL_DIR"
# Keep ckpt_path pointing to trained model
cfg["gen_config"]["ckpt_path"] = cfg["train_config"]["output_directory"]
with open("$XAPP_CONFIG", "w") as f:
    json.dump(cfg, f, indent=2)
print("Cross-app config written")
PYEOF

echo "--- Inference on $TEST ---"
python "$REPO_DIR/SSSD/src/inference.py" -c "$XAPP_CONFIG" -ckpt_iter max

# ── Evaluate ──────────────────────────────────────────────────────────────────
echo "--- Evaluation ---"
INFER_OUT=$(find "$EVAL_DIR" -name "imputation0.npy" -printf '%h\n' | head -1)
if [[ -n "$INFER_OUT" ]]; then
    python "$REPO_DIR/alibaba-trace/evaluate_metrics.py" \
      --results-dir "$INFER_OUT" --save-json
    echo "Cross-app accuracy saved in $INFER_OUT/metrics.json"
fi

echo "Job complete: train=$TRAIN test=$TEST"
