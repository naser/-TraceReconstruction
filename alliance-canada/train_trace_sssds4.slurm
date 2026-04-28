#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --job-name=trace-sssds4
#SBATCH --output=%x-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$SCRATCH/trace-reconstruction}"
DATASET="${DATASET:-stream}"
SEQ_LEN="${SEQ_LEN:-200}"
MISSING_K="${MISSING_K:-10}"
TRAIN_ITERS="${TRAIN_ITERS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RESULTS_DIR="${RESULTS_DIR:-$SCRATCH/trace-reconstruction-results}"
CONFIG_PATH="${CONFIG_PATH:-$SLURM_TMPDIR/trace_config.json}"

module purge
module load StdEnv/2023 python/3.11 cuda gcc

virtualenv --no-download "$SLURM_TMPDIR/venv"
source "$SLURM_TMPDIR/venv/bin/activate"
python -m pip install --no-index -r "$REPO_DIR/alliance-canada/requirements-sssd-trace.txt"

export PYTHONPATH="$REPO_DIR/SSSD/src:${PYTHONPATH:-}"
export MAX_JOBS="${SLURM_CPUS_PER_TASK}"

cd "$REPO_DIR/SSSD/src/entensions/cauchy"
python setup.py build_ext --inplace

python "$REPO_DIR/alliance-canada/make_trace_config.py" \
  --dataset "$DATASET" \
  --seq-len "$SEQ_LEN" \
  --missing-k "$MISSING_K" \
  --train-iters "$TRAIN_ITERS" \
  --batch-size "$BATCH_SIZE" \
  --results-dir "$RESULTS_DIR" \
  --output "$CONFIG_PATH"

cd "$REPO_DIR"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
PY
nvidia-smi

python "$REPO_DIR/SSSD/src/train.py" -c "$CONFIG_PATH"
python "$REPO_DIR/SSSD/src/inference.py" -c "$CONFIG_PATH" -ckpt_iter max
