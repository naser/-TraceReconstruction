#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# transfer_learning.slurm
#
# Realistic Transfer-Learning Experiment (NEW — added experiment RQ-TL)
# =======================================================================
# This is an ADDED experiment to address reviewer cold-start concerns.
# It is NOT a reframing of the existing RQ5 or RQ6:
#   RQ5 (current paper): zero-shot cross-application transfer.
#   RQ6 (current paper): one model trained on multiple apps simultaneously.
# RQ-TL adds: pretrain on a related source, fine-tune on a small target.
# This tests whether warm-starting reduces the amount of target data needed.
#
# Both the source (pybench_sharedcpu) and target (phpbench_sharedcpu) use
# the SAME 84-token shared CPU-benchmark vocabulary built by
# varying-workload/preprocess_pts_shared_vocab.py.  This is a deliberate
# methodological choice that DIFFERS from the paper's default encoding
# (each dataset gets its own frequency-ranked vocabulary, §III-B).  The
# shared vocab is required so that pretrained weights transfer meaningfully.
# This deviation must be disclosed if this experiment is cited in the paper.
#
# Warm-start mechanism:
#   1. Locate the best checkpoint from a previous PRETRAINED_CKPT_DIR run
#      (typically the 10000.pkl from a full pybench_sharedcpu training).
#   2. Copy it to the fine-tune output directory as 0.pkl  (model weights
#      only; optimizer state is reset so fine-tuning uses a fresh AdamW).
#   3. Run train.py — it finds 0.pkl, loads the model, and trains for
#      TRAIN_ITERS more iterations.  New checkpoints are named 1000.pkl,
#      2000.pkl, …, {TRAIN_ITERS}.pkl.
#   4. inference.py -ckpt_iter max  loads the final checkpoint.
#
# This is an intentional comparison with the cold-start scratch runs so that
# the only difference between "fine-tune" and "scratch" is the initialisation.
#
# Controlled via environment variables:
#   TARGET_DS          Target dataset base name (default: phpbench_sharedcpu)
#   TRAIN_SIZE         # of sequences in the small target split (default: 1000)
#   PRETRAINED_CKPT_DIR
#                      Directory containing the pretrained checkpoint (.pkl files)
#                      Should be the T200_... output dir of a full source-dataset run.
#   SEQ_LEN            default 200
#   MISSING_K          default 10
#   TRAIN_ITERS        Fine-tuning iterations (default: 10000)
#
# Results are written to:
#   $SCRATCH/transfer-learning-results/SSSDS4/
#     {TARGET_DS}_n{TRAIN_SIZE}_seq{L}_k{K}/T200_beta00.0001_betaT0.02/
#
# Usage:
#   sbatch --export=ALL,TARGET_DS=phpbench_sharedcpu,TRAIN_SIZE=500,\
#   PRETRAINED_CKPT_DIR=$SCRATCH/varying-workload-results/SSSDS4/\
#   pybench_sharedcpu_seq200_k10/T200_beta00.0001_betaT0.02 \
#   alliance-canada/transfer_learning.slurm
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-03:00:00
#SBATCH --job-name=tl-finetune
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/3}"
TARGET_DS="${TARGET_DS:-phpbench_sharedcpu}"
TRAIN_SIZE="${TRAIN_SIZE:-1000}"
SEQ_LEN="${SEQ_LEN:-200}"
MISSING_K="${MISSING_K:-10}"
TRAIN_ITERS="${TRAIN_ITERS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PRETRAINED_CKPT_DIR="${PRETRAINED_CKPT_DIR:-}"

SUBSAMPLED_DS="${TARGET_DS}_n${TRAIN_SIZE}"

RESULTS_BASE="${RESULTS_BASE:-$SCRATCH/transfer-learning-results}"
TL_RESULTS_DIR="$RESULTS_BASE/SSSDS4"
CONFIG_PATH="$SLURM_TMPDIR/tl_finetune_config.json"

echo "========================================================"
echo "  Transfer-Learning Experiment (RQ-TL)"
echo "  Pre-trained ckpt dir : $PRETRAINED_CKPT_DIR"
echo "  Target dataset       : $TARGET_DS"
echo "  Target train size    : $TRAIN_SIZE sequences"
echo "  Subsampled DS        : $SUBSAMPLED_DS"
echo "  Seq len              : $SEQ_LEN   Blackout: $MISSING_K"
echo "  Fine-tune iters      : $TRAIN_ITERS"
echo "  Results base         : $RESULTS_BASE"
echo "========================================================"

# ── Validate inputs ──────────────────────────────────────────────────────────
if [[ -z "$PRETRAINED_CKPT_DIR" ]]; then
    echo "ERROR: PRETRAINED_CKPT_DIR is not set."
    echo "  Example: PRETRAINED_CKPT_DIR=\$SCRATCH/varying-workload-results/SSSDS4/"
    echo "           pybench_sharedcpu_seq200_k10/T200_beta00.0001_betaT0.02"
    exit 1
fi
if [[ ! -d "$PRETRAINED_CKPT_DIR" ]]; then
    echo "ERROR: Pretrained checkpoint directory does not exist: $PRETRAINED_CKPT_DIR"
    exit 1
fi

# Verify subsampled target dataset exists
TRAIN_NPY="$REPO_DIR/TraceReconstruction-main/Datasets/$SUBSAMPLED_DS/sequence_length_$SEQ_LEN/train.npy"
if [[ ! -f "$TRAIN_NPY" ]]; then
    echo "ERROR: Subsampled training data not found: $TRAIN_NPY"
    echo "Run cold-start/preprocess_coldstart.py first."
    exit 1
fi
echo "Target dataset verified: $TRAIN_NPY"

mkdir -p "$RESULTS_BASE" "$TL_RESULTS_DIR"
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

# ── Warm-start: copy pretrained weights as 0.pkl ─────────────────────────────
# The fine-tuning output dir is mirrored from make_trace_config.py logic but
# using the transfer-learning results base.
# Config uses test-dataset=TARGET_DS (full test split, not subsampled).
FINETUNE_OUTPUT="$TL_RESULTS_DIR/${SUBSAMPLED_DS}_to_${TARGET_DS}_seq${SEQ_LEN}_k${MISSING_K}/T200_beta00.0001_betaT0.02"
mkdir -p "$FINETUNE_OUTPUT"

echo "--- Preparing warm-start checkpoint ---"
export PRETRAINED_CKPT_DIR FINETUNE_OUTPUT
python - <<'PY'
import torch, glob, os
from pathlib import Path

src_dir    = Path(os.environ["PRETRAINED_CKPT_DIR"])
output_dir = Path(os.environ["FINETUNE_OUTPUT"])

# Find maximum checkpoint in the pretrained directory
pkl_files = sorted(src_dir.glob("*.pkl"),
                   key=lambda p: (lambda s: int(s) if s.isdigit() else -1)(p.stem))
if not pkl_files:
    raise FileNotFoundError(f"No .pkl checkpoint files found in {src_dir}")

# Pick the highest-numbered checkpoint (best trained model)
valid = [p for p in pkl_files if p.stem.isdigit()]
if not valid:
    raise FileNotFoundError(f"No numerically-named .pkl files in {src_dir}")
max_ckpt = max(valid, key=lambda p: int(p.stem))
print(f"Loading pretrained checkpoint: {max_ckpt}")

ckpt = torch.load(str(max_ckpt), map_location="cpu")
if "model_state_dict" not in ckpt:
    raise KeyError(f"No 'model_state_dict' key in checkpoint {max_ckpt}")

# Save as 0.pkl with model weights only (fresh optimizer = proper fine-tuning)
warm_start = output_dir / "0.pkl"
torch.save({"model_state_dict": ckpt["model_state_dict"]}, str(warm_start))
print(f"Warm-start checkpoint written → {warm_start}")
print(f"Model state dict keys: {len(ckpt['model_state_dict'])}")
PY
echo "Warm-start checkpoint ready in $FINETUNE_OUTPUT"

# ── Generate config (train on SUBSAMPLED_DS, test on full TARGET_DS) ──────────
python "$REPO_DIR/alliance-canada/make_trace_config.py" \
  --dataset      "$SUBSAMPLED_DS" \
  --test-dataset "$TARGET_DS" \
  --seq-len      "$SEQ_LEN" \
  --missing-k    "$MISSING_K" \
  --train-iters  "$TRAIN_ITERS" \
  --batch-size   "$BATCH_SIZE" \
  --results-dir  "$TL_RESULTS_DIR" \
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

# ── Fine-tune from warm-start ─────────────────────────────────────────────────
cd "$REPO_DIR"
echo "--- Fine-tuning from pretrained weights on $SUBSAMPLED_DS ($TRAIN_SIZE seqs) ---"
# train.py will find 0.pkl via find_max_epoch() and continue training from iter 1
python "$REPO_DIR/SSSD/src/train.py" -c "$CONFIG_PATH"

# ── Inference on full test set ────────────────────────────────────────────────
echo "--- Inference (test on full $TARGET_DS test set) ---"
python "$REPO_DIR/SSSD/src/inference.py" -c "$CONFIG_PATH" -ckpt_iter max

# ── Evaluate ──────────────────────────────────────────────────────────────────
echo "--- Evaluation ---"
RUN_LABEL="${SUBSAMPLED_DS}_to_${TARGET_DS}_seq${SEQ_LEN}_k${MISSING_K}"
INFER_OUT="$TL_RESULTS_DIR/$RUN_LABEL/T200_beta00.0001_betaT0.02"

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
cp "$INFER_OUT/metrics.json" \
   "$SUMMARY_DIR/tl_finetune_${TARGET_DS}_n${TRAIN_SIZE}_seq${SEQ_LEN}_k${MISSING_K}.json"
echo "Summary: $SUMMARY_DIR/tl_finetune_${TARGET_DS}_n${TRAIN_SIZE}_seq${SEQ_LEN}_k${MISSING_K}.json"

METRICS=$(cat "$INFER_OUT/metrics.json")
echo "Results: $METRICS"
echo "Job complete: fine-tune on $TARGET_DS  n_train=$TRAIN_SIZE  source=$PRETRAINED_CKPT_DIR"
