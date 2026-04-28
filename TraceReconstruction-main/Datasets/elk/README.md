# ELK Stack Dataset

## Source
Noferesti and Ezzati-Jivan, "Enhancing empirical software performance engineering
research with kernel-level events: A comprehensive system tracing approach"
https://github.com/mnoferestibrocku/dataset-repo

## Raw Data Status
The raw trace data has been downloaded to:
  d:\Naser\3\elk-dataset-repo\Trace-RawData\
as split files (elktrace_aa through elktrace_an, ~275 MB total).

Repository scripts for data collection and analysis are in elk-dataset-repo/.

## Preprocessing

### Option 1 — Automated (recommended)
After completing WSL setup (restart your PC if you ran `wsl --install`):
```bash
# Open WSL terminal and run:
bash /mnt/d/Naser/3/setup_wsl_and_extract.sh
```
This handles: reassembling the 14 split trace parts, extracting the tar.gz,
parsing with babeltrace, and creating elk_syscalls.txt in this folder.

Then run the final step on Windows:
```
python d:\Naser\3\TraceReconstruction-main\Datasets\elk\preprocess_elk.py
```

### Option 2 — Manual (if WSL is already available)
1. In WSL: `cat /mnt/d/Naser/3/elk-dataset-repo/Trace-RawData/elktrace_* > elktrace.tar.gz`
2. `tar -xzf elktrace.tar.gz`
3. Install babeltrace: `sudo apt-get install babeltrace python3-babeltrace`
4. Run the ELK extraction portion of `setup_wsl_and_extract.sh` (Step 3)
5. Run `preprocess_elk.py` on Windows

## Notes
- The paper uses ALL portions of the ELK trace, including segments with
  injected noise (CPU, I/O, Network, Memory noise types), unlike other datasets
- This makes ELK the most challenging dataset in the study
- Final dataset: 10,000 training + 500 test sequences, seq lengths 50/100/150/200
