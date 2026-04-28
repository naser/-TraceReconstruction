#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_varying_workload.sh
#
# Submit all RQ-W (varying-workload) experiments to the Narval SLURM cluster.
#
# Experiment design:
#
#   ELK workload experiment (4 conditions):
#     1. elk-clean-baseline  : train=elk_clean   test=elk_clean
#     2. elk-noisy-baseline  : train=elk_noisy   test=elk_noisy
#     3. elk-clean-to-noisy  : train=elk_clean   test=elk_noisy  <-- cross
#     4. elk-noisy-to-clean  : train=elk_noisy   test=elk_clean  <-- cross
#
#   PTS within-family experiment (4 conditions):
#   Memory-bandwidth family (stream vs ramspeed):
#     5. stream-baseline     : train=stream      test=stream
#     6. ramspeed-baseline   : train=ramspeed    test=ramspeed
#     7. stream-to-ramspeed  : train=stream      test=ramspeed   <-- cross
#     8. ramspeed-to-stream  : train=ramspeed    test=stream     <-- cross
#
#   CPU-compute family (pybench vs phpbench):
#     9. pybench-baseline    : train=pybench     test=pybench
#    10. phpbench-baseline   : train=phpbench    test=phpbench
#    11. pybench-to-phpbench : train=pybench     test=phpbench   <-- cross
#    12. phpbench-to-pybench : train=phpbench    test=pybench    <-- cross
#
# NOTE: Conditions 5, 6, 9, 10 replicate the existing RQ1/RQ2 training setup
#       but are re-submitted to obtain fresh checkpoints for consistent
#       comparison; existing Results/ files serve as double-check references.
#
# Run from workspace root:
#   bash alliance-canada/submit_varying_workload.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_BASE="${SCRATCH:-$HOME/scratch}/varying-workload-results"
SLURM_SCRIPT="$REPO_DIR/alliance-canada/varying_workload.slurm"
ACCOUNT="${SLURM_ACCOUNT:-def-naser2_gpu}"

mkdir -p "$REPO_DIR/slurm-logs"
export REPO_DIR RESULTS_BASE

echo "Repo root    : $REPO_DIR"
echo "Results base : $RESULTS_BASE"
echo "Account      : $ACCOUNT"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Helper: submit one varying-workload job
#   submit_varwl <train_ds> <test_ds> <condition_label> [<extra_vars>]
# ─────────────────────────────────────────────────────────────────────────────
submit_varwl() {
    local train_ds="$1"
    local test_ds="$2"
    local condition="$3"
    local -a sbatch_args=(
      "--account=$ACCOUNT"
      "--job-name=varwl-${condition}"
      "--export=ALL,REPO_DIR=$REPO_DIR,RESULTS_BASE=$RESULTS_BASE,TRAIN_DS=$train_ds,TEST_DS=$test_ds,CONDITION=$condition,SEQ_LEN=200,MISSING_K=10,TRAIN_ITERS=10000"
      "$SLURM_SCRIPT"
    )

    if [[ -n "${SLURM_PARTITION:-}" ]]; then
      sbatch_args=("--partition=$SLURM_PARTITION" "${sbatch_args[@]}")
    fi

    JID=$(sbatch "${sbatch_args[@]}" | awk '{print $NF}')

    echo "  Submitted varwl-${condition}  → Job $JID"
}

echo "════════════════════════════════════════════════════════════"
echo "  BLOCK 1 — ELK Workload Experiment (4 conditions)"
echo "════════════════════════════════════════════════════════════"

# In-workload baselines
submit_varwl elk_clean elk_clean  "elk-clean-baseline"
submit_varwl elk_noisy elk_noisy  "elk-noisy-baseline"

# Cross-workload  
submit_varwl elk_clean elk_noisy  "elk-clean-to-noisy"
submit_varwl elk_noisy elk_clean  "elk-noisy-to-clean"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  BLOCK 2 — PTS Memory-Bandwidth Family (stream vs ramspeed)"
echo "════════════════════════════════════════════════════════════"

# In-workload baselines (stream and ramspeed) — shared-vocab datasets
submit_varwl stream_sharedmem   stream_sharedmem   "pts-stream-baseline"
submit_varwl ramspeed_sharedmem ramspeed_sharedmem "pts-ramspeed-baseline"

# Cross-workload within memory-bandwidth family
submit_varwl stream_sharedmem   ramspeed_sharedmem "pts-stream-to-ramspeed"
submit_varwl ramspeed_sharedmem stream_sharedmem   "pts-ramspeed-to-stream"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  BLOCK 3 — PTS CPU-Benchmark Family (pybench vs phpbench)"
echo "════════════════════════════════════════════════════════════"

# In-workload baselines (pybench and phpbench) — shared-vocab datasets
submit_varwl pybench_sharedcpu  pybench_sharedcpu  "pts-pybench-baseline"
submit_varwl phpbench_sharedcpu phpbench_sharedcpu "pts-phpbench-baseline"

# Cross-workload within CPU-benchmark family
submit_varwl pybench_sharedcpu  phpbench_sharedcpu "pts-pybench-to-phpbench"
submit_varwl phpbench_sharedcpu pybench_sharedcpu  "pts-phpbench-to-pybench"

echo ""
echo "All 12 varying-workload jobs submitted."
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       $REPO_DIR/slurm-logs/"
echo "Results in:    $RESULTS_BASE/"
