#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_cold_start.sh
#
# Submit all RQ-CS (cold-start / training-data-size) experiments.
#
# Experiment design:
#   Dataset : phpbench_sharedcpu (84-token shared CPU-benchmark vocabulary;
#             also used in RQ-W varying-workload experiment so the full-10k
#             pts-phpbench-baseline result serves as the reference point).
#
#   Training sizes:
#     1.  n=100   — extreme low-data regime
#     2.  n=500   — very low data
#     3.  n=1000  — low data
#     4.  n=2500  — moderate data
#     5.  n=5000  — near-full data
#     (+  n=10000 reference loaded from existing RQ-W pts-phpbench-baseline run)
#
# Pre-requisite (run once, CPU-only):
#   python3 cold-start/preprocess_coldstart.py
#
# Run from workspace root:
#   bash alliance-canada/submit_cold_start.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_BASE="${SCRATCH:-$HOME/scratch}/cold-start-results"
SLURM_SCRIPT="$REPO_DIR/alliance-canada/cold_start.slurm"
ACCOUNT="${SLURM_ACCOUNT:-def-naser2_gpu}"

mkdir -p "$REPO_DIR/slurm-logs"
export REPO_DIR RESULTS_BASE

echo "Repo root    : $REPO_DIR"
echo "Results base : $RESULTS_BASE"
echo "Account      : $ACCOUNT"
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
        ROWS=$(python3 -c "import numpy as np; a=np.load('$DS_PATH'); print(a.shape[0])" 2>/dev/null || echo "?")
        echo "  [OK]      phpbench_sharedcpu_n${N}  (${ROWS} sequences)"
    fi
done
if [[ "$ALL_OK" -eq 0 ]]; then
    echo ""
    echo "ERROR: Some subsampled datasets are missing."
    echo "Run: python3 cold-start/preprocess_coldstart.py"
    exit 1
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Helper: submit one cold-start job
#   submit_cs <train_size>
# ─────────────────────────────────────────────────────────────────────────────
submit_cs() {
    local n="$1"
    local -a sbatch_args=(
      "--account=$ACCOUNT"
      "--job-name=cs-n${n}"
      "--export=ALL,REPO_DIR=$REPO_DIR,RESULTS_BASE=$RESULTS_BASE,DATASET=phpbench_sharedcpu,TRAIN_SIZE=${n},SEQ_LEN=200,MISSING_K=10,TRAIN_ITERS=10000"
      "$SLURM_SCRIPT"
    )

    if [[ -n "${SLURM_PARTITION:-}" ]]; then
        sbatch_args=("--partition=$SLURM_PARTITION" "${sbatch_args[@]}")
    fi

    JID=$(sbatch "${sbatch_args[@]}" | awk '{print $NF}')
    echo "  Submitted cs-n${n}  → Job $JID"
}

echo "════════════════════════════════════════════════════════════"
echo "  RQ-CS — Training-Data-Size / Cold-Start Experiment"
echo "  Dataset: phpbench_sharedcpu  (84-token shared CPU vocab)"
echo "════════════════════════════════════════════════════════════"
submit_cs 100
submit_cs 500
submit_cs 1000
submit_cs 2500
submit_cs 5000

echo ""
echo "All 5 cold-start jobs submitted (n=100, 500, 1000, 2500, 5000)."
echo ""
echo "Reference (n=10000): already available in RQ-W results:"
echo "  \$SCRATCH/varying-workload-results/SSSDS4/phpbench_sharedcpu_seq200_k10/"
echo "  (pts-phpbench-baseline condition)"
echo ""
echo "After all jobs complete, generate the report:"
echo "  python3 cold-start/generate_coldstart_report.py \\"
echo "      --cold-start-base \$SCRATCH/cold-start-results \\"
echo "      --varwl-base \$SCRATCH/varying-workload-results \\"
echo "      --output cold-start/COLDSTART_RESULTS.txt"
