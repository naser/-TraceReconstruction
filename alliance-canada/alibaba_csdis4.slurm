#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# alibaba_csdis4.slurm
#
# Train and evaluate CSDI^S4 on the Alibaba microservice trace dataset using
# the standalone CSDIS4Imputer API (imputers/CSDIS4.py).
# Reads train.npy / test.npy directly (shape N×1×T, as saved by preprocessing).
# Outputs imputation*.npy / original*.npy / mask*.npy so that
# evaluate_metrics.py can process results the same way as all other models.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --job-name=alibaba-csdis4
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/3}"
SEQ_LEN="${SEQ_LEN:-200}"
MISSING_K="${MISSING_K:-10}"
EPOCHS="${EPOCHS:-200}"
MISSING_K_INT=$(( MISSING_K ))          # ensure integer

RESULTS_BASE="${RESULTS_BASE:-$SCRATCH/alibaba-trace-results}"
RESULTS_DIR="${RESULTS_BASE}/CSDIS4/alibaba_seq${SEQ_LEN}_k${MISSING_K}"
DATA_DIR="$REPO_DIR/TraceReconstruction-main/Datasets/alibaba/sequence_length_${SEQ_LEN}"

echo "========================================"
echo "  CSDI^S4 on Alibaba"
echo "  Seq len   : $SEQ_LEN"
echo "  Blackout k: $MISSING_K_INT  (center missing, paper default)"
echo "  Epochs    : $EPOCHS"
echo "  Results   : $RESULTS_DIR"
echo "========================================"

mkdir -p "$RESULTS_DIR"
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

# ── Driver inline ─────────────────────────────────────────────────────────────
cd "$REPO_DIR"
python - <<PYEOF
import os, sys, numpy as np, torch

sys.path.insert(0, "$REPO_DIR/SSSD/src")
from imputers.CSDIS4 import CSDIS4Imputer

SEQ_LEN   = int("$SEQ_LEN")
K         = int("$MISSING_K_INT")
EPOCHS    = int("$EPOCHS")
DATA_DIR  = "$DATA_DIR"
RESULTS_DIR = "$RESULTS_DIR"

# ----- Load data (shape: N×1×T) → transpose to N×T×1 for CSDI API -----
train_raw = np.load(os.path.join(DATA_DIR, "train.npy"))  # (N,1,T)
test_raw  = np.load(os.path.join(DATA_DIR, "test.npy"))   # (N,1,T)

train_np = train_raw.transpose(0, 2, 1).astype(np.float32)  # (N,T,1)
test_np  = test_raw .transpose(0, 2, 1).astype(np.float32)  # (N,T,1)

# Concatenate – CSDIS4Imputer does its own train/valid/test split internally
combined = np.concatenate([train_np, test_np], axis=0)
# CSDIS4 expects a torch tensor (calls .size(1) on it)
combined_t = torch.from_numpy(combined)
print(f"Combined series shape: {combined_t.shape}  (passed to CSDI train)")

# ----- Train -----
imputer = CSDIS4Imputer()
imputer.train(
    series               = combined_t,
    masking              = "nrm",         # nrm = segment-based blackout, works on clean data
    missing_ratio_or_k   = K,
    train_split          = 0.90,
    valid_split          = 0.05,          # last 5 % = implicit test inside CSDI
    epochs               = EPOCHS,
    samples_generate     = 10,
    path_save            = RESULTS_DIR + "/",
    batch_size           = 16,
    lr                   = 1e-3,
    layers               = 4,
    channels             = 64,
    nheads               = 8,
    difussion_embedding_dim = 128,
    beta_start           = 0.0001,
    beta_end             = 0.5,
    num_steps            = 50,
    schedule             = "quad",
    is_unconditional     = 0,
    target_strategy      = "random",
)

# ----- Re-run inference on the fixed 500-sequence test set with centered mask -----
imputer.load_weights(
    path_load_model = os.path.join(RESULTS_DIR, "model_csdi_training"),
    path_config     = os.path.join(RESULTS_DIR, "config_csdi_training.json"),
)

# Build centered blackout mask (paper default)
n_test, T, C = test_np.shape
centre = T // 2
start  = centre - K // 2
end    = start + K

# Produce imputation*.npy in SSSD format (N×C×T, mask 1=observed 0=missing)
BATCH = 125
for bi, b_start in enumerate(range(0, n_test, BATCH)):
    batch = test_np[b_start: b_start + BATCH]
    n_b   = batch.shape[0]

    # zero-fill blackout for conditioning
    x_hat = batch.copy()
    x_hat[:, start:end, :] = 0.0

    imp_samples = imputer.impute(x_hat)   # returns (n_samples, N, T, C)
    # average over diffusion samples
    imp_mean = imp_samples.mean(axis=0)   # (N, T, C)

    # indicating_mask: 1 where hidden (blackout region)
    indicating = np.zeros_like(batch)
    indicating[:, start:end, :] = 1.0
    missing_mask = 1.0 - indicating       # 1=observed, 0=hidden

    # convert to SSSD layout (N, C, T) = (N, 1, T)
    np.save(os.path.join(RESULTS_DIR, f"imputation{bi}.npy"),
            imp_mean.transpose(0, 2, 1).astype(np.float32))
    np.save(os.path.join(RESULTS_DIR, f"original{bi}.npy"),
            batch.transpose(0, 2, 1).astype(np.float32))
    np.save(os.path.join(RESULTS_DIR, f"mask{bi}.npy"),
            missing_mask.transpose(0, 2, 1).astype(np.float32))
    print(f"  Saved batch {bi}: shape {imp_mean.shape}")

print("CSDI^S4 inference complete.")
PYEOF

# ── Evaluate ──────────────────────────────────────────────────────────────────
echo "--- Evaluation ---"
python "$REPO_DIR/alibaba-trace/evaluate_metrics.py" \
  --results-dir "$RESULTS_DIR" --save-json

echo "Job complete."
