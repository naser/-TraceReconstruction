#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# alibaba_diffusion.slurm
#
# Train a diffusion model (DiffWave / SSSD^SA / SSSD^S4) on the Alibaba
# microservice trace dataset and run inference.
#
# Environment variables (set by sbatch --export or job-array wrapper):
#   MODEL      0=DiffWave  1=SSSD^SA  2=SSSD^S4  (default: 2)
#   SEQ_LEN    default 200
#   MISSING_K  default 10
#   TRAIN_ITERS default 10000
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --job-name=alibaba-diffusion
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/3}"
MODEL="${MODEL:-2}"
SEQ_LEN="${SEQ_LEN:-200}"
MISSING_K="${MISSING_K:-10}"
TRAIN_ITERS="${TRAIN_ITERS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-64}"

MODEL_NAMES=("DiffWave" "SSSDSA" "SSSDS4")
MODEL_NAME="${MODEL_NAMES[$MODEL]}"

RESULTS_BASE="${RESULTS_BASE:-$SCRATCH/alibaba-trace-results}"
RESULTS_DIR="${RESULTS_BASE}/${MODEL_NAME}"
CONFIG_PATH="$SLURM_TMPDIR/alibaba_config.json"

echo "========================================"
echo "  Model     : $MODEL_NAME (use_model=$MODEL)"
echo "  Seq len   : $SEQ_LEN"
echo "  Blackout k: $MISSING_K"
echo "  Iterations: $TRAIN_ITERS"
echo "  Results   : $RESULTS_DIR"
echo "========================================"

mkdir -p "$RESULTS_BASE" "$RESULTS_DIR"
mkdir -p "$(dirname "$REPO_DIR")/3/slurm-logs" 2>/dev/null || true
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

# ── Generate config ──────────────────────────────────────────────────────────
python "$REPO_DIR/alliance-canada/make_trace_config.py" \
  --dataset     "alibaba" \
  --seq-len     "$SEQ_LEN" \
  --missing-k   "$MISSING_K" \
  --train-iters "$TRAIN_ITERS" \
  --batch-size  "$BATCH_SIZE" \
  --results-dir "$RESULTS_DIR" \
  --model       "$MODEL" \
  --masking     "cm" \
  --output      "$CONFIG_PATH"

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

# ── Train ─────────────────────────────────────────────────────────────────────
cd "$REPO_DIR"
echo "--- Training ---"
python "$REPO_DIR/SSSD/src/train.py" -c "$CONFIG_PATH"

# ── Inference ─────────────────────────────────────────────────────────────────
echo "--- Inference ---"
python "$REPO_DIR/SSSD/src/inference.py" -c "$CONFIG_PATH" -ckpt_iter max

# ── Evaluate ──────────────────────────────────────────────────────────────────
echo "--- Evaluation ---"
# Find the actual results sub-directory created by inference.py
INFER_OUT=$(find "$RESULTS_DIR" -name "imputation0.npy" -printf '%h\n' | head -1)
if [[ -n "$INFER_OUT" ]]; then
    python "$REPO_DIR/alibaba-trace/evaluate_metrics.py" \
      --results-dir "$INFER_OUT" --save-json
    echo "Metrics saved in $INFER_OUT/metrics.json"
else
    echo "WARNING: could not find imputation output to evaluate."
fi

echo "Job complete."
