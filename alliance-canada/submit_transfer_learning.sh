#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_transfer_learning.sh
#
# Submit all RQ-TL (transfer-learning) fine-tuning experiments.
#
# === FRAMING NOTE ============================================================
# This is an ADDED experiment to address the reviewer cold-start concern.
# It is NOT a replacement or reframing of existing RQ5 / RQ6:
#   RQ5 (current paper): zero-shot cross-application transfer.
#   RQ6 (current paper): one model trained on multiple apps simultaneously.
# RQ-TL adds a third scenario not currently in the paper:
#   pretrain on a related source dataset, then fine-tune on a tiny target set.
# =============================================================================
#
# === SHARED-VOCABULARY DISCLOSURE ============================================
# Source (pybench_sharedcpu) and target (phpbench_sharedcpu) both use a
# SHARED 84-token CPU-benchmark vocabulary produced by
# varying-workload/preprocess_pts_shared_vocab.py.  The paper's default
# preprocessing (§III-B) builds a separate frequency-ranked vocabulary per
# dataset.  Using a shared vocabulary is necessary here so that the model's
# embedding weights transfer sensibly between corpora, but it is a deviation
# from the paper's preprocessing protocol that must be disclosed.
# =============================================================================
#
# Experiment design:
#
#   Source (pretrain): pybench_sharedcpu  — full 10 000-sequence training run
#                      from the RQ-W experiment (pts-pybench-baseline).
#                      Shares the 84-token CPU-benchmark vocab with target.
#
#   Target (fine-tune): phpbench_sharedcpu at sizes 100, 500, 1 000, 2 500, 5 000.
#
#   Evaluation: same 500-sequence test set used in cold-start / RQ-W to enable
#               direct comparison of fine-tuned vs scratch performance.
#
# Job dependency logic:
#   • If the pybench pretrained checkpoint already exists under
#     $SCRATCH/varying-workload-results (from a previous RQ-W run), the
#     fine-tuning jobs are submitted directly.
#   • Otherwise a new pretraining job is submitted first and all fine-tuning
#     jobs are held with --dependency=afterok:{PRETRAIN_JID}.
#
# Pre-requisite (run once, CPU-only):
#   python3 cold-start/preprocess_coldstart.py
#
# Run from workspace root:
#   bash alliance-canada/submit_transfer_learning.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TL_RESULTS_BASE="${SCRATCH:-$HOME/scratch}/transfer-learning-results"
VARWL_RESULTS_BASE="${SCRATCH:-$HOME/scratch}/varying-workload-results"
SLURM_FINETUNE="$REPO_DIR/alliance-canada/transfer_learning.slurm"
SLURM_VARWL="$REPO_DIR/alliance-canada/varying_workload.slurm"
ACCOUNT="${SLURM_ACCOUNT:-def-naser2_gpu}"

mkdir -p "$REPO_DIR/slurm-logs"
export REPO_DIR

echo "Repo root         : $REPO_DIR"
echo "TL results base   : $TL_RESULTS_BASE"
echo "VarWL results base: $VARWL_RESULTS_BASE"
echo "Account           : $ACCOUNT"
echo ""

# ── Verify subsampled datasets exist ─────────────────────────────────────────
DATASETS="$REPO_DIR/TraceReconstruction-main/Datasets"
echo "Checking subsampled datasets..."
ALL_OK=1
for N in 100 500 1000 2500 5000; do
    DS_PATH="$DATASETS/phpbench_sharedcpu_n${N}/sequence_length_200/train.npy"
    if [[ ! -f "$DS_PATH" ]]; then
        echo "  [MISSING] $DS_PATH"
        ALL_OK=0
    else
        echo "  [OK]      phpbench_sharedcpu_n${N}"
    fi
done
if [[ "$ALL_OK" -eq 0 ]]; then
    echo ""
    echo "ERROR: Subsampled datasets missing. Run:"
    echo "  python3 cold-start/preprocess_coldstart.py"
    exit 1
fi
echo ""

# ── Check for existing pretrained pybench checkpoint ─────────────────────────
PRETRAINED_CKPT_DIR="$VARWL_RESULTS_BASE/SSSDS4/pybench_sharedcpu_seq200_k10/T200_beta00.0001_betaT0.02"
PRETRAINED_JID=""

if ls "$PRETRAINED_CKPT_DIR"/*.pkl >/dev/null 2>&1; then
    MAX_CKPT=$(ls "$PRETRAINED_CKPT_DIR"/*.pkl | \
               awk -F'/' '{print $NF}' | \
               sed 's/\.pkl$//' | \
               grep '^[0-9]*$' | \
               sort -n | tail -1)
    echo "Found pretrained pybench_sharedcpu checkpoint at:"
    echo "  $PRETRAINED_CKPT_DIR/${MAX_CKPT}.pkl"
    echo "Fine-tuning jobs will use this checkpoint directly."
    PRETRAINED_JID=""
else
    echo "No pretrained checkpoint found for pybench_sharedcpu."
    echo "Submitting pretraining job first..."
    PRETRAIN_JID_RAW=$(sbatch \
        --account="$ACCOUNT" \
        --job-name="tl-pretrain-pybench" \
        ${SLURM_PARTITION:+--partition="$SLURM_PARTITION"} \
        --export="ALL,REPO_DIR=$REPO_DIR,RESULTS_BASE=$VARWL_RESULTS_BASE,TRAIN_DS=pybench_sharedcpu,TEST_DS=pybench_sharedcpu,CONDITION=pts-pybench-baseline,SEQ_LEN=200,MISSING_K=10,TRAIN_ITERS=10000" \
        "$SLURM_VARWL" | awk '{print $NF}')
    PRETRAINED_JID="$PRETRAIN_JID_RAW"
    echo "  Submitted pretrain job → Job $PRETRAINED_JID"
    echo "  Fine-tuning jobs will wait for pretraining to complete."
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Helper: submit one fine-tuning job
#   submit_tl <train_size>
# ─────────────────────────────────────────────────────────────────────────────
submit_tl() {
    local n="$1"
    local -a sbatch_args=(
      "--account=$ACCOUNT"
      "--job-name=tl-n${n}"
      "--export=ALL,REPO_DIR=$REPO_DIR,RESULTS_BASE=$TL_RESULTS_BASE,TARGET_DS=phpbench_sharedcpu,TRAIN_SIZE=${n},SEQ_LEN=200,MISSING_K=10,TRAIN_ITERS=10000,PRETRAINED_CKPT_DIR=$PRETRAINED_CKPT_DIR"
      "$SLURM_FINETUNE"
    )

    if [[ -n "${SLURM_PARTITION:-}" ]]; then
        sbatch_args=("--partition=$SLURM_PARTITION" "${sbatch_args[@]}")
    fi

    if [[ -n "$PRETRAINED_JID" ]]; then
        sbatch_args=("--dependency=afterok:${PRETRAINED_JID}" "${sbatch_args[@]}")
    fi

    JID=$(sbatch "${sbatch_args[@]}" | awk '{print $NF}')
    local dep_str=""
    [[ -n "$PRETRAINED_JID" ]] && dep_str=" (dep: $PRETRAINED_JID)"
    echo "  Submitted tl-n${n}  → Job $JID${dep_str}"
}

echo "════════════════════════════════════════════════════════════"
echo "  RQ-TL — Transfer-Learning / Fine-Tuning Experiment"
echo "  Source  : pybench_sharedcpu  (full 10k, pretrained)"
echo "  Target  : phpbench_sharedcpu (84-token shared CPU vocab)"
echo "════════════════════════════════════════════════════════════"
submit_tl 100
submit_tl 500
submit_tl 1000
submit_tl 2500
submit_tl 5000

echo ""
echo "All 5 transfer-learning fine-tuning jobs submitted."
echo "(n = 100, 500, 1000, 2500, 5000)"
echo ""
echo "After all jobs complete, generate the transfer-learning report:"
echo "  python3 cold-start/generate_transfer_report.py \\"
echo "      --cold-start-base \$SCRATCH/cold-start-results \\"
echo "      --tl-base \$SCRATCH/transfer-learning-results \\"
echo "      --varwl-base \$SCRATCH/varying-workload-results \\"
echo "      --output cold-start/TRANSFER_RESULTS.txt"
