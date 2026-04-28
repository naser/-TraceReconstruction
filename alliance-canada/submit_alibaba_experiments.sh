#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_alibaba_experiments.sh
#
# Submit all Alibaba trace experiments to the Narval SLURM cluster.
# Run from the workspace root:  bash alliance-canada/submit_alibaba_experiments.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_BASE="${SCRATCH:-$HOME/scratch}/alibaba-trace-results"
SLURM_DIR="$REPO_DIR/alliance-canada"

mkdir -p "$REPO_DIR/slurm-logs"
export REPO_DIR RESULTS_BASE

echo "Repo root    : $REPO_DIR"
echo "Results base : $RESULTS_BASE"
echo ""

# Helper to submit with named environment
submit () {
    local script="$1"; shift
    local extra_vars="$1"; shift
    local jobname="$1"

    JID=$(sbatch \
      --partition=gpubase_bygpu_b3 \
      --account="def-naser2" \
      --job-name="$jobname" \
      --export="ALL,REPO_DIR=$REPO_DIR,RESULTS_BASE=$RESULTS_BASE,$extra_vars" \
      "$script" | awk '{print $NF}')
    echo "  Submitted $jobname  → Job $JID"
}

# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 1 – RQ1 EQUIVALENT: all 4 models, blackout 1 and blackout 10, seq=200
# Matching paper Table 1 (blackout 1) and Table 2 / Table 3 (blackout 10).
# Models: 0=DiffWave, 1=SSSD^SA, 2=SSSD^S4  via shared train.py
#         CSDI^S4 via its standalone script (alibaba_csdis4.slurm)
# ═══════════════════════════════════════════════════════════════════════════
echo "Block 1: RQ1 – all 4 models × blackout {1,10}, seq=200"

for MODEL in 0 1 2; do
    for K in 1 10; do
        NAMES=("DiffWave" "SSSDSA" "SSSDS4")
        NAME="${NAMES[$MODEL]}"
        submit "$SLURM_DIR/alibaba_diffusion.slurm" \
               "MODEL=$MODEL,SEQ_LEN=200,MISSING_K=$K,TRAIN_ITERS=10000" \
               "ali-rq1-${NAME}-k${K}"
    done
done

# CSDI^S4 – uses standalone imputer (separate SLURM script)
for K in 1 10; do
    submit "$SLURM_DIR/alibaba_csdis4.slurm" \
           "SEQ_LEN=200,MISSING_K=$K,EPOCHS=200" \
           "ali-rq1-CSDIS4-k${K}"
done

# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 2 – RQ2 EQUIVALENT: SSSD^S4 × ALL blackout sizes × ALL seq lengths
# Matching paper Table 4 (blackout sizes) and Figure 3 (sequence lengths).
# Full matrix: 5 blackout sizes × 4 sequence lengths = 20 jobs.
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "Block 2: RQ2 – SSSD^S4 × blackout {5,10,20,30,40} × seq {50,100,150,200}"

for SEQ in 50 100 150 200; do
    for K in 5 10 20 30 40; do
        # k=10, seq=200 was already submitted in Block 1 (same trained model).
        # SLURM deduplication: job names are unique so re-submitting is safe;
        # the training step will reload the existing checkpoint.
        submit "$SLURM_DIR/alibaba_diffusion.slurm" \
               "MODEL=2,SEQ_LEN=$SEQ,MISSING_K=$K,TRAIN_ITERS=10000" \
               "ali-rq2-S4-s${SEQ}-k${K}"
    done
done

# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 3 – RQ3 EQUIVALENT: SAITS transformer baseline, blackout 10, seq=200
# Matching paper Table 5 (SSSD^S4 vs SAITS).
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "Block 3: RQ3 – SAITS, seq=200, blackout=10"

submit "$SLURM_DIR/alibaba_saits.slurm" \
       "SEQ_LEN=200,MISSING_K=10,EPOCHS=200" \
       "ali-rq3-SAITS"

# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 4 – RQ5 EQUIVALENT: cross-application transfer
# Matching paper Table 6.
# Dataset directories are all lowercase; plaid not PLAID.
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "Block 4: RQ5 – cross-app transfer involving Alibaba"

# Train Alibaba → test on apache, plaid, elk
for TARGET in apache plaid elk; do
    submit "$SLURM_DIR/alibaba_cross_app.slurm" \
           "TRAIN=alibaba,TEST=$TARGET,TRAIN_ITERS=10000" \
           "ali-rq5-ali-to-${TARGET}"
done

# Train apache / plaid / elk → test Alibaba
for SOURCE in apache plaid elk; do
    submit "$SLURM_DIR/alibaba_cross_app.slurm" \
           "TRAIN=$SOURCE,TEST=alibaba,TRAIN_ITERS=10000" \
           "ali-rq5-${SOURCE}-to-ali"
done

echo ""
echo "All jobs submitted. Monitor with:  squeue -u $USER"
echo "Results will be in: $RESULTS_BASE"
