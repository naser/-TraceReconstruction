#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_rerun.sh
#
# Rerun DiffWave, SSSD^SA, SSSD^S4, and SAITS on the Alibaba trace with the
# corrected random trace-level split (seed 42).  Configuration: seq=200,
# blackout=10 (the primary RQ1 comparison point).
#
# Usage (from workspace root):
#   bash alliance-canada/submit_rerun.sh
#
# Step 1: submits alibaba_preprocess.slurm (CPU job, ~20 min)
# Step 2: submits 4 training/inference jobs with --dependency=afterok:<PREP_JID>
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_BASE="${SCRATCH:-$HOME/scratch}/alibaba-trace-results-v2"
SLURM_DIR="$REPO_DIR/alliance-canada"
ACCOUNT="def-naser2"
PARTITION="gpubase_bygpu_b3"

mkdir -p "$REPO_DIR/slurm-logs"
export REPO_DIR RESULTS_BASE

echo "Repo root    : $REPO_DIR"
echo "Results base : $RESULTS_BASE"
echo ""

# ── Step 1: Preprocessing (CPU, no GPU needed) ───────────────────────────────
echo "Submitting preprocessing job …"
PREP_JID=$(sbatch \
  --account="$ACCOUNT" \
  --export="ALL,REPO_DIR=$REPO_DIR" \
  "$SLURM_DIR/alibaba_preprocess.slurm" \
  | awk '{print $NF}')
echo "  Preprocessing → Job $PREP_JID"

# ── Step 2: Training jobs (GPU, depend on preprocessing) ─────────────────────
echo ""
echo "Submitting 4 model training jobs (--dependency=afterok:$PREP_JID) …"

submit_gpu () {
    local script="$1"; local extra_vars="$2"; local jobname="$3"
    JID=$(sbatch \
      --partition="$PARTITION" \
      --account="$ACCOUNT" \
      --job-name="$jobname" \
      --dependency="afterok:$PREP_JID" \
      --export="ALL,REPO_DIR=$REPO_DIR,RESULTS_BASE=$RESULTS_BASE,$extra_vars" \
      "$script" | awk '{print $NF}')
    echo "  $jobname → Job $JID"
}

# DiffWave (model=0), SSSD^SA (model=1), SSSD^S4 (model=2)
submit_gpu "$SLURM_DIR/alibaba_diffusion.slurm" \
  "MODEL=0,SEQ_LEN=200,MISSING_K=10,TRAIN_ITERS=10000" \
  "ali-v2-DiffWave-k10"

submit_gpu "$SLURM_DIR/alibaba_diffusion.slurm" \
  "MODEL=1,SEQ_LEN=200,MISSING_K=10,TRAIN_ITERS=10000" \
  "ali-v2-SSSDSA-k10"

submit_gpu "$SLURM_DIR/alibaba_diffusion.slurm" \
  "MODEL=2,SEQ_LEN=200,MISSING_K=10,TRAIN_ITERS=10000" \
  "ali-v2-SSSDS4-k10"

# SAITS
submit_gpu "$SLURM_DIR/alibaba_saits.slurm" \
  "SEQ_LEN=200,MISSING_K=10,EPOCHS=200" \
  "ali-v2-SAITS-k10"

echo ""
echo "All jobs submitted. Monitor with:  squeue -u \$USER"
echo "Results will be in: $RESULTS_BASE"
echo ""
echo "After jobs finish, collect metrics with:"
echo "  module load StdEnv/2023 scipy-stack/2023b"
echo "  for d in $RESULTS_BASE/**/; do"
echo "    [[ -f \"\$d/imputation0.npy\" ]] && \\"
echo "    python $REPO_DIR/alibaba-trace/evaluate_metrics.py --results-dir \"\$d\" --save-json"
echo "  done"
