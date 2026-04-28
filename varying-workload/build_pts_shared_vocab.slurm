#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# build_pts_shared_vocab.slurm
#
# Downloads PTS LTTng traces from Zenodo 437170 and builds shared-vocabulary
# datasets for:
#   - stream + ramspeed  (memory-bandwidth family)
#   - pybench + phpbench (CPU-benchmark family)
#
# This is a CPU-only job; no GPU required.
# Approximate disk: ~20 GB in $SCRATCH/zenodo-pts-cache  (cleared at end)
# Approximate time: 2-4 hours (dominated by downloads)
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --job-name=pts-shared-vocab
#SBATCH --account=def-naser2_cpu
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/3}"
mkdir -p "$REPO_DIR/slurm-logs" 2>/dev/null || true

echo "================================================================"
echo "  Build PTS Shared Vocabulary"
echo "  Node    : $(hostname)"
echo "  Started : $(date)"
echo "  Scratch : $SCRATCH"
echo "================================================================"

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load StdEnv/2023 python/3.11

virtualenv --no-download "$SLURM_TMPDIR/venv"
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index numpy

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export SCRATCH="${SCRATCH:-$HOME/scratch}"

# ── Verify pre-staged downloads ───────────────────────────────────────────────
echo ""
echo "--- Verifying pre-staged Zenodo downloads ---"
CACHE="$SCRATCH/zenodo-pts-cache"
for ds in stream ramspeed pybench phpbench; do
    f="$CACHE/${ds}.zip"
    if [[ ! -f "$f" ]]; then
        echo "ERROR: $f not found. Run login-node downloads first:"
        echo "  bash $REPO_DIR/varying-workload/download_pts_zenodo.sh"
        exit 1
    fi
    echo "  OK: $f ($(du -sh "$f" | cut -f1))"
done

echo ""
echo "--- Building shared vocab for mem-bandwidth family (stream + ramspeed) ---"
python3 "$REPO_DIR/varying-workload/preprocess_pts_shared_vocab.py" --family mem

echo ""
echo "--- Building shared vocab for CPU-benchmark family (pybench + phpbench) ---"
python3 "$REPO_DIR/varying-workload/preprocess_pts_shared_vocab.py" --family cpu

echo ""
echo "--- Verifying output ---"
python3 - <<'PY'
import json, numpy as np
from pathlib import Path

repo = Path(".")
base = repo / "TraceReconstruction-main" / "Datasets"

# Load vocab size from JSON (authoritative) rather than max token in training data.
# The per-dataset max ID can differ because different syscalls appear in each
# dataset's chunk-split training portion — that is expected and correct.
mem_vocab = json.load(open(repo / "varying-workload" / "pts_mem_vocab.json"))
cpu_vocab = json.load(open(repo / "varying-workload" / "pts_cpu_vocab.json"))
mem_vocab_size = mem_vocab["vocab_size"]
cpu_vocab_size = cpu_vocab["vocab_size"]

for ds, expected_vocab in [
    ("stream_sharedmem",   mem_vocab_size),
    ("ramspeed_sharedmem", mem_vocab_size),
    ("pybench_sharedcpu",  cpu_vocab_size),
    ("phpbench_sharedcpu", cpu_vocab_size),
]:
    arr = np.load(base / ds / "sequence_length_200" / "train.npy")
    ids = sorted(set(arr.flatten().tolist()))
    # All IDs must be valid (1 .. vocab_size; oob = vocab_size+1 is allowed but rare)
    assert arr.min() >= 1, f"FAIL {ds}: min id {arr.min()} < 1"
    assert arr.max() <= expected_vocab + 1, f"FAIL {ds}: max id {arr.max()} > oob {expected_vocab+1}"
    print(f"  PASS {ds}: shape={arr.shape}  unique_ids={len(ids)}  "
          f"range=[{ids[0]},{ids[-1]}]  shared_vocab={expected_vocab}")

# Verify the two families share consistent vocab JSON
assert mem_vocab["datasets"] == ["stream", "ramspeed"], "FAIL: unexpected mem datasets"
assert cpu_vocab["datasets"] == ["pybench", "phpbench"], "FAIL: unexpected cpu datasets"
print(f"  PASS: mem vocab covers {mem_vocab_size} shared syscalls")
print(f"  PASS: cpu vocab covers {cpu_vocab_size} shared syscalls")
PY

echo ""
echo "================================================================"
echo "  Done. $(date)"
echo "  Vocab files: varying-workload/pts_mem_vocab.json"
echo "               varying-workload/pts_cpu_vocab.json"
echo "  New dataset dirs (originals untouched):"
echo "    TraceReconstruction-main/Datasets/stream_sharedmem/"
echo "    TraceReconstruction-main/Datasets/ramspeed_sharedmem/"
echo "    TraceReconstruction-main/Datasets/pybench_sharedcpu/"
echo "    TraceReconstruction-main/Datasets/phpbench_sharedcpu/"
echo "  Resubmit cross-workload jobs with:"
echo "    bash alliance-canada/submit_varying_workload.sh"
echo "================================================================"
