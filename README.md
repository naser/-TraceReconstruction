# Execution Trace Reconstruction Using Diffusion-Based Generative Models

Official repository for the paper:

> **Leveraging Diffusion Models for Execution Trace Reconstruction**  

We introduce a novel application of diffusion-based generative models for reconstructing masked portions of execution traces (system call sequences and microservice call graphs), conduct a thorough evaluation across 12+ datasets, and compare against transformer and RNN baselines.

---

## Table of Contents

1. [Overview](#overview)
2. [Datasets](#datasets)
3. [Repository Structure](#repository-structure)
4. [Models](#models)
5. [Experiments](#experiments)
   - [RQ1–RQ6: Main ICSE Experiments](#rq1rq6-main-icse-experiments)
   - [RQ-Ali: Alibaba Microservice Trace (New)](#rq-ali-alibaba-microservice-trace-new)
   - [RQ-W: Varying Workload (New)](#rq-w-varying-workload-new)
   - [RQ-CS: Cold Start / Sample Efficiency (New)](#rq-cs-cold-start--sample-efficiency-new)
   - [RQ-TL: Transfer Learning (New)](#rq-tl-transfer-learning-new)
6. [Results Summary](#results-summary)
7. [Cluster Setup (Alliance Canada / Narval)](#cluster-setup-alliance-canada--narval)
8. [Baselines](#baselines)
9. [Citation](#citation)

---

## Overview

Execution traces capture the runtime behavior of software systems as sequences of events (system calls, API calls). Missing or corrupted segments are common in practice due to resource limits, monitoring gaps, and distributed system complexity. This paper asks:

> *Can diffusion-based generative models reconstruct masked segments of execution traces with higher accuracy than existing approaches?*

We evaluate four diffusion models — **SSSD^S4**, **SSSD^SA**, **CSDI^S4**, and **DiffWave** — across 12 OS-level system call datasets and the Alibaba 2021 distributed microservice trace, using sequence lengths of 50–200 and blackout sizes of 1–40 tokens. Comparisons are made against **SAITS** (transformer imputation) and **LSTM** baselines.

---

## Datasets

All raw trace data is publicly available. The preprocessing scripts in this repository generate the `.npy` dataset files used for training and evaluation.

| Dataset | Source | Download | Preprocessing |
|---------|--------|----------|---------------|
| **Phoronix Test Suites (PTS)** — compress-gzip, ffmpeg, scimark2, stream, ramspeed, phpbench, pybench, iozone, unpack-linux | [Martin & Marangozova-Martin, INRIA](https://inria.hal.science/hal-02047273/document) | [Zenodo](https://zenodo.org/records/437207) | `TraceReconstruction-main/Datasets/preprocess_plaid.py` |
| **Apache** | [Fournier et al., 2021](https://arxiv.org/pdf/2103.06915) | [Zenodo](https://zenodo.org/records/4091287) | See `ctf_reader.py` |
| **PLAID** | [Ring et al., ACM TISSEC 2021](https://dl.acm.org/doi/full/10.1145/3461462) | [GitLab](https://gitlab.com/jhring/uvm_ids) | `plaid-dataset/scripts/` |
| **ELK** | [Noferesti & Ezzati-Jivan, JSS 2024](https://www.sciencedirect.com/science/article/pii/S0164121224001626) | [GitHub](https://github.com/mnoferestibrocku/dataset-repo) | `elk-dataset-repo/` |
| **Alibaba 2021 Microservice Trace** | [Luo et al., ACM SoCC 2021](https://dl.acm.org/doi/10.1145/3472883.3487003) | [Alibaba OSS](http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2021MicroservicesTraces/MSCallGraph/MSCallGraph_0.tar.gz) | `alibaba-trace/prepare_saits_dataset.py` |

> **Note:** Raw data files and processed `.npy` arrays are not committed to this repository due to size. Download the raw data using the links above and run the preprocessing scripts to regenerate the datasets.

---

## Repository Structure

```
.
├── TraceReconstruction-main/       # Core paper artifacts (ICSE 2025)
│   ├── Datasets/                   # Preprocessing scripts + dataset structure
│   │   ├── preprocess_plaid.py
│   │   └── <dataset>/
│   │       └── sequence_length_<N>/   # train.npy / test.npy (generated)
│   └── Results/                    # Raw model outputs (generated)
│       ├── CSDI_S4/
│       ├── DiffWave/
│       ├── SSSD_S4/
│       └── SSSD_SA/
│
├── SSSD/                           # Diffusion model training/inference code
│   └── src/                        # SSSD^S4, SSSD^SA, CSDI^S4, DiffWave
│
├── SAITS/                          # SAITS transformer baseline
│   ├── run_models.py
│   ├── Global_Config.py
│   └── configs/
│
├── alibaba-trace/                  # Alibaba microservice trace experiment (EMSE/IEEE-T)
│   ├── prepare_saits_dataset.py    # Preprocessing: CSV → train/test .npy
│   ├── saits_train_infer.py        # SAITS training & inference on Alibaba
│   ├── evaluate_metrics.py         # Accuracy / PerfRate / ROUGE-L evaluation
│   ├── EXPERIMENT_REPORT.md        # Full experiment design & status
│   └── RESULTS_REPORT.txt          # Final numerical results
│
├── cold-start/                     # RQ-CS: Sample efficiency & transfer learning
│   ├── preprocess_coldstart.py     # Build phpbench_sharedcpu_n{100,500,...} datasets
│   ├── generate_coldstart_report.py
│   ├── generate_transfer_report.py
│   ├── COLDSTART_RESULTS.txt       # Sample-efficiency table
│   └── TRANSFER_RESULTS.txt        # Transfer learning (pretrain→fine-tune) table
│
├── varying-workload/               # RQ-W: Cross-workload generalization
│   ├── preprocess_elk_workloads.py         # ELK clean/noisy datasets
│   ├── preprocess_elk_workloads_v2.py
│   ├── preprocess_pts_shared_vocab.py      # Shared PTS vocabulary builder
│   ├── build_pts_shared_vocab.slurm
│   ├── analyze_varying_workload.py
│   ├── generate_varwl_report.py
│   ├── pts_cpu_vocab.json          # 84-token shared CPU-benchmark vocabulary
│   ├── pts_mem_vocab.json          # 98-token shared memory-bandwidth vocabulary
│   ├── VARWL_REPORT.md             # Full report with methodology
│   └── VARWL_RESULTS.txt           # Numerical results
│
├── plaid-dataset/                  # PLAID dataset preprocessing pipeline
│   ├── src/
│   ├── scripts/
│   └── data/
│
├── elk-dataset-repo/               # ELK dataset documentation & info
│
├── alliance-canada/                # SLURM job scripts for Narval cluster
│   ├── alibaba_saits.slurm
│   ├── alibaba_diffusion.slurm
│   ├── alibaba_csdis4.slurm
│   ├── alibaba_preprocess.slurm
│   ├── cold_start.slurm
│   ├── transfer_learning.slurm
│   ├── varying_workload.slurm
│   ├── train_trace_sssds4.slurm
│   ├── make_trace_config.py        # Auto-generate SSSD config files
│   ├── submit_alibaba_experiments.sh
│   ├── submit_cold_start.sh
│   ├── submit_transfer_learning.sh
│   ├── submit_varying_workload.sh
│   └── requirements-sssd-trace.txt
│
├── slurm-logs/                     # SLURM stderr/stdout logs from all runs
│
├── lstm_baseline.py                # LSTM baseline implementation
├── check_events.py                 # Utility: event count / dataset sanity checks
├── ctf_reader.py                   # CTF trace reader (for Apache / ELK raw data)
└── setup_wsl_and_extract.sh        # WSL environment setup script
```

---

## Models

### Diffusion Models (primary)

Implemented in `SSSD/` — forked from [AI4HealthUOL/SSSD](https://github.com/AI4HealthUOL/SSSD).

| Model | Architecture | Reference |
|-------|-------------|-----------|
| **SSSD^S4** | S4-based structured state-space diffusion | Alcaraz & Strodthoff, 2023 |
| **SSSD^SA** | Self-attention diffusion | Alcaraz & Strodthoff, 2023 |
| **CSDI^S4** | Conditional score-based diffusion with S4 | Tashiro et al., 2021 |
| **DiffWave** | Dilated CNN diffusion (adapted for sequences) | Kong et al., 2021 |

Default hyperparameters used throughout:
- Diffusion steps `T = 200`, `β₀ = 0.0001`, `βT = 0.02`
- Training iterations: 10,000
- Learning rate: 2×10⁻⁴
- Masking strategy: **centered blackout** (contiguous block at sequence center)
- Hardware: NVIDIA A100 32 GB (Alliance Canada Narval cluster)

### Transformer Baseline

**SAITS** (Self-Attention-based Imputation for Time Series) — implemented in `SAITS/`, from [WenjieDu/SAITS](https://github.com/WenjieDu/SAITS).

### RNN Baseline

**LSTM** — implemented in `lstm_baseline.py`, trained on the same sequences with identical masking strategy.

---

## Experiments

### RQ1–RQ6: Main ICSE Experiments

Located in `TraceReconstruction-main/`.

- **RQ1:** Which diffusion model achieves best reconstruction accuracy across all datasets and configurations?
- **RQ2:** Effect of sequence length (50, 100, 150, 200) on reconstruction quality.
- **RQ3:** Diffusion models vs. SAITS transformer baseline.
- **RQ4:** Effect of blackout size (1, 5, 10, 20, 30, 40) on reconstruction quality.
- **RQ5:** Zero-shot cross-application transfer: train on one dataset, test on another.
- **RQ6:** Multi-application training: one model trained on all datasets simultaneously.

Datasets used: compress-gzip, ffmpeg, scimark2, stream, ramspeed, phpbench, pybench, iozone, unpack-linux (PTS), apache, plaid, elk.

### RQ-Ali: Alibaba Microservice Trace (New)

Located in `alibaba-trace/`. Added for IEEE Transactions revision in response to reviewer feedback on scope.

**Motivation:** The original 12 datasets are all single-host OS-level system call sequences. Reviewer 1 requested evaluation on a distributed microservice trace.

**Dataset:** Alibaba 2021 MSCallGraph — 6,088,846 call events from ~20,000 microservices over 12 hours; vocabulary of **7,377** distinct `<rpctype>:<dm>` call types (vs. 87–108 syscalls per existing dataset).

**Key results** (seq=200, blackout=10, random trace-level 80/20 split, seed=42):

| Model | Accuracy | Perfect Rate | ROUGE-L |
|-------|----------|-------------|---------|
| SSSD^SA | 10.08% | 3.40% | 12.78 |
| DiffWave | 9.32% | 3.40% | 11.20 |
| SSSD^S4 | 7.18% | 1.00% | 9.96 |
| SAITS | 6.74% | 0.40% | 11.98 |
| Random baseline | 0.014% | — | — |

All diffusion models outperform SAITS and are 500–720× above the random baseline, consistent with main paper findings. Lower absolute accuracy reflects the 51–85× larger vocabulary.

See `alibaba-trace/RESULTS_REPORT.txt` and `alibaba-trace/EXPERIMENT_REPORT.md` for full analysis.

### RQ-W: Varying Workload (New)

Located in `varying-workload/`. Added for IEEE Transactions revision.

**Research question:** Does SSSD^S4 generalize to a *different workload* from the one it was trained on?

Three workload families evaluated with **shared vocabularies**:

| Family | Train → Test | Acc Δ vs. baseline |
|--------|-------------|-------------------|
| ELK | clean ↔ noisy stress-ng | ≈ 0 (negligible degradation) |
| PTS Memory | Stream ↔ RAMspeed | −0.54% |
| PTS CPU | PyBench ↔ PHPBench | −0.68% |

**Finding:** Cross-workload transfer incurs only minimal accuracy loss (< 1%) when a shared vocabulary is used, demonstrating that SSSD^S4 is robust to workload variation within a domain.

See `varying-workload/VARWL_REPORT.md` for full methodology and `varying-workload/VARWL_RESULTS.txt` for numerical results.

### RQ-CS: Cold Start / Sample Efficiency (New)

Located in `cold-start/`. Added for IEEE Transactions revision.

**Research question:** How many training sequences does SSSD^S4 need to achieve good reconstruction quality?

Model: SSSD^S4, dataset: phpbench_sharedcpu (shared 84-token vocabulary), seq=200, k=10.

| N_train | Accuracy | PerfRate | ROUGE-L |
|---------|----------|---------|---------|
| 100 | 79.58% | 49.00% | 83.88% |
| 500 | 91.46% | 78.80% | 92.60% |
| 1,000 | 89.02% | 67.00% | 89.82% |
| 2,500 | 91.26% | 77.20% | 92.06% |
| 5,000 | 90.60% | 78.80% | 91.34% |
| 10,000 (reference) | 89.14% | 72.60% | 90.12% |

**Finding:** SSSD^S4 reaches competitive accuracy with as few as 500 training sequences. Performance plateaus quickly, with diminishing returns beyond 500–1,000 samples.

See `cold-start/COLDSTART_RESULTS.txt`.

### RQ-TL: Transfer Learning (New)

Located in `cold-start/`. Added for IEEE Transactions revision.

**Research question:** Does pre-training on a related source trace reduce the target training data needed?

Setup: pretrain on `pybench_sharedcpu` (10,000 seqs), fine-tune on `phpbench_sharedcpu` (N sequences), shared 84-token vocabulary.

| N_train | Scratch Acc | Fine-tuned Acc | Δ |
|---------|------------|---------------|---|
| 100 | 79.58% | 90.72% | **+11.14%** |
| 500 | 91.46% | 93.70% | +2.24% |
| 1,000 | 89.02% | 94.66% | +5.64% |
| 2,500 | 91.26% | 91.10% | −0.16% |
| 5,000 | 90.60% | 94.60% | +4.00% |

**Finding:** Transfer learning provides the largest benefit at very low data regimes (N=100: +11.14% accuracy), demonstrating that a pretrained model can bootstrap effective reconstruction with minimal target data.

See `cold-start/TRANSFER_RESULTS.txt`.

---

## Results Summary

| Experiment | Best Model | Best Accuracy |
|-----------|-----------|--------------|
| RQ1 (ICSE, all datasets) | SSSD^S4 | 53–99% (dataset-dependent) |
| RQ-Ali (Alibaba microservice) | SSSD^SA | 10.08% (7,377-token vocab) |
| RQ-W (cross-workload) | SSSD^S4 | < 1% degradation vs. same-workload |
| RQ-CS (cold start, N=500) | SSSD^S4 | 91.46% |
| RQ-TL (transfer, N=100) | SSSD^S4 + TL | 90.72% (+11.14% vs. scratch) |

---

## Cluster Setup (Alliance Canada / Narval)

All experiments were run on the **Narval** cluster (Alliance Canada) using NVIDIA A100 GPUs (`def-naser2_gpu` account).

SLURM job scripts are in `alliance-canada/`. Key scripts:

| Script | Purpose |
|--------|---------|
| `train_trace_sssds4.slurm` | Main SSSD^S4/SA/CSDI training |
| `alibaba_diffusion.slurm` | Alibaba diffusion model training |
| `alibaba_saits.slurm` | Alibaba SAITS baseline |
| `alibaba_csdis4.slurm` | Alibaba CSDI^S4 |
| `alibaba_preprocess.slurm` | Alibaba data preprocessing |
| `cold_start.slurm` | Cold-start sample-efficiency runs |
| `transfer_learning.slurm` | Transfer-learning fine-tuning runs |
| `varying_workload.slurm` | Cross-workload generalization runs |

Submission convenience scripts (`submit_*.sh`) launch arrays of jobs for all experimental conditions automatically.

### Environment Setup

```bash
# Install dependencies
pip install -r alliance-canada/requirements-sssd-trace.txt

# Or use the conda environment (SAITS)
conda env create -f SAITS/conda_env_dependencies.yml
```

### Running a Single Experiment

```bash
# 1. Generate config files for all datasets
python alliance-canada/make_trace_config.py

# 2. Preprocess a dataset (example: PLAID)
python TraceReconstruction-main/Datasets/preprocess_plaid.py

# 3. Train SSSD^S4 (example)
python SSSD/src/main.py --config <path_to_config>

# 4. Evaluate results
python alibaba-trace/evaluate_metrics.py  # or equivalent per-experiment script
```

---

## Baselines

### LSTM Baseline

```bash
python lstm_baseline.py
```

The `lstm_baseline.py` script trains a sequence-to-sequence LSTM on the same centered-blackout reconstruction task and evaluates accuracy, perfect rate, and ROUGE-L.

### SAITS Baseline

```bash
python SAITS/run_models.py --config SAITS/configs/<dataset>.json
```

SAITS is a self-attention-based imputation model; see `SAITS/README.md` for configuration details.

---

## Citation

If you use this code or results in your research, please cite:

```bibtex
@inproceedings{tracereconstruction2025,
  title     = {Execution Trace Reconstruction Using Diffusion-Based Generative Models},
  booktitle = {Proceedings of the 47th International Conference on Software Engineering (ICSE)},
  year      = {2025}
}
```

For the Alibaba microservice trace dataset:

```bibtex
@inproceedings{luo2021alibaba,
  title     = {Characterizing Microservice Dependency and Performance: Alibaba Trace Analysis},
  author    = {Luo, Shutian and others},
  booktitle = {Proceedings of the ACM Symposium on Cloud Computing (SoCC)},
  pages     = {412--426},
  year      = {2021}
}
```

---

## License

See `SSSD/LICENCE` and `SAITS/LICENSE` for the licenses of the respective model implementations.
