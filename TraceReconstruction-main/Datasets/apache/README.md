# Apache Web Server Dataset

## Source
Fournier et al., "On Improving Deep Learning Trace Analysis with System Call Arguments"
https://zenodo.org/records/4091287

## Raw Data
Download `data.zip` (~1.1 GB) from Zenodo. The zip contains raw LTTng kernel traces:
- `data/requests/train/kernel/` — training traces
- `data/requests/test/kernel/`  — test traces

## Preprocessing

### Option 1 — Automated (recommended)
After completing WSL setup (restart your PC if you ran `wsl --install`):
```bash
# Open WSL terminal and run:
bash /mnt/d/Naser/3/setup_wsl_and_extract.sh
```
This script handles everything: installing babeltrace, downloading the 1.1 GB
Zenodo archive, extracting syscall sequences, and placing train_syscalls.txt
and test_syscalls.txt in this folder.

Then run the final step on Windows:
```
python d:\Naser\3\TraceReconstruction-main\Datasets\apache\preprocess_apache.py
```

### Option 2 — Manual (if WSL is already available)
1. Install babeltrace: `sudo apt-get install babeltrace python3-babeltrace`
2. Download data.zip from https://zenodo.org/records/4091287 (~1.1 GB)
3. Extract: `unzip data.zip`
4. Run the extraction portion of `setup_wsl_and_extract.sh` (Step 5)
5. Run `preprocess_apache.py` on Windows

## Notes
- The paper excludes the anomaly-injection portions of the trace
  (only uses normal HTTP request handling traffic)
- The final dataset should have: 10,000 training sequences and 500 test sequences
  in numpy format (shape: N x seq_len x 1, dtype int32) for lengths 50, 100, 150, 200
