# Alibaba Microservice Trace: Experiment Design and Status Report

## Context

Reviewer 1 of the IEEE Transactions submission identified a **Scope & Motivation
Gap**: the paper's introduction motivates trace reconstruction in the context of
complex distributed microservice architectures, yet all 12 existing experimental
datasets consist of single-host OS-level system call sequences.  The reviewer
explicitly requests:

> "To fully address the motivating challenge, it would be compelling to include an
> evaluation on microservice trace datasets such as **Alibaba Trace** [1]."
>
> [1] Luo, Shutian, et al. "Characterizing microservice dependency and performance:
>  Alibaba Trace Analysis." *Proc. ACM Symposium on Cloud Computing*, 2021, pp. 412–426.

This report describes the complete experiment design added to the project to address
this comment, the five implementation bugs that were found and fixed, the current
SLURM account status, and the expected results structure.

---

## 1. Dataset: Alibaba 2021 Microservice Trace

### 1.1 Source and Scope

The Alibaba 2021 microservice trace captures **12 hours of production traffic** from
~20,000 microservices across 10,000+ bare-metal nodes in Alibaba's production cluster.
The dataset is publicly available at:
<https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2021>

Only the **MSCallGraph** table is used for trace reconstruction.  Each row records one
microservice-to-microservice call within a distributed request, identified by a
`traceid` (analogous to a process ID) and a dotted `rpcid` encoding the call tree
depth and position.

### 1.2 Why This Dataset Addresses the Reviewer's Comment

| Property | System-call datasets (existing) | Alibaba trace (new) |
|---|---|---|
| Trace source | Single-host kernel tracing (LTTng) | Distributed microservice call graphs |
| Event type | OS system calls (open, read, write…) | API calls across ~20,000 services |
| Vocabulary | 87–108 distinct events per dataset | **7,377 distinct call types** |
| Sequence structure | Single execution context | Nested RPC / HTTP / MQ call trees |
| Application domain | Benchmarks & web servers | Production cloud infrastructure |

The dataset directly demonstrates that the diffusion-based reconstruction method is
not limited to single-host system calls but applies to **distributed-system execution
traces**, which is the motivating scenario described in the introduction.

### 1.3 Download and Preprocessing

**Raw file downloaded:**
```
alibaba-trace/raw/MSCallGraph_0.csv   (~1.3 GB uncompressed, ~154 MB compressed)
```
Source URL:
```
http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2021MicroservicesTraces/MSCallGraph/MSCallGraph_0.tar.gz
```

**Key statistics from MSCallGraph_0.csv:**
- Total call records: **6,088,846**
- Unique request traces: **130,512**
- Average calls per request: **~46.6**
- Call type vocabulary: **7,377** distinct `<rpctype>:<dm>` tokens
- Communication types: rpc (44.8 %), mc (43.7 %), http (4.9 %), userDefined (2.8 %),
  db (1.5 %), mq (1.4 %)

**Call-type encoding:**
Each call event is encoded as `<rpctype>:<dm>` (e.g. `rpc:abc123…`), where `rpctype`
is the communication paradigm and `dm` is the hashed downstream microservice identity.
This token is always populated (unlike the `interface` field, which is empty for ~50 %
of calls), making it the closest analogue to a system call name.

**Train/Test split:**  
The concatenated event stream (sorted by trace arrival order, then within each trace
by timestamp and rpcid depth) is split **80/20 by event position** — identical to the
approach used for all 12 existing datasets (Apache, ELK, PLAID, PTS benchmarks).

**Preprocessed dataset location:**
```
TraceReconstruction-main/Datasets/alibaba/
    alibaba_calls.txt               (6,088,846 tokens, one per line)
    sequence_length_50/             (train: 10,000×50×1, test: 500×50×1)
    sequence_length_100/            (train: 10,000×100×1, test: 500×100×1)
    sequence_length_150/            (train: 10,000×150×1, test: 500×150×1)
    sequence_length_200/            (train: 10,000×200×1, test: 500×200×1)
```
All arrays are int32 numpy format `(N, seq_len, 1)`, matching the exact format used
by every other dataset.

---

## 2. Experiment Design

The new experiments are designed to produce **results parallel to every existing RQ in
the paper**, so the Alibaba results can slot into tables alongside the existing 12
datasets.

### 2.1 RQ1 — Are diffusion models effective at reconstructing microservice traces?

**Goal:** Replicate Tables 1–3 of the paper for the Alibaba dataset.  
**Setup:**
- All four models: **DiffWave**, **SSSD^SA**, **SSSD^S4**, **CSDI^S4**
- Sequence length: 200
- Blackout sizes: 1 (single event; Table 1 equivalent) and 10 (Table 2/3 equivalent)
- Masking: **centered** (paper default: Section 3.4)
- Metrics: Accuracy, Perfect Rate, ROUGE-L

**SLURM jobs:**
| Job name | Model | Blackout | Script |
|---|---|---|---|
| ali-rq1-DiffWave-k1 | DiffWave | 1 | alibaba_diffusion.slurm (MODEL=0) |
| ali-rq1-SSSDSA-k1 | SSSD^SA | 1 | alibaba_diffusion.slurm (MODEL=1) |
| ali-rq1-SSSDS4-k1 | SSSD^S4 | 1 | alibaba_diffusion.slurm (MODEL=2) |
| ali-rq1-CSDIS4-k1 | CSDI^S4 | 1 | alibaba_csdis4.slurm |
| ali-rq1-DiffWave-k10 | DiffWave | 10 | alibaba_diffusion.slurm (MODEL=0) |
| ali-rq1-SSSDSA-k10 | SSSD^SA | 10 | alibaba_diffusion.slurm (MODEL=1) |
| ali-rq1-SSSDS4-k10 | SSSD^S4 | 10 | alibaba_diffusion.slurm (MODEL=2) |
| ali-rq1-CSDIS4-k10 | CSDI^S4 | 10 | alibaba_csdis4.slurm |

**Expected result table (RQ1 — to be filled after jobs complete):**

*Accuracy when imputing 1 event (seq=200, blackout=1):*

| Dataset | DiffWave | SSSD^SA | SSSD^S4 | CSDI^S4 |
|---|---|---|---|---|
| alibaba | TBD | TBD | TBD | TBD |

*Performance with blackout size 10 (seq=200):*

| Dataset | Accuracy (SSSD^S4) | Perfect Rate (SSSD^S4) | ROUGE-L (SSSD^S4) |
|---|---|---|---|
| alibaba | TBD | TBD | TBD |

### 2.2 RQ2 — Stability across sequence lengths and blackout sizes

**Goal:** Replicate Tables 4–5 and Figure 3 of the paper for the Alibaba dataset.  
**Setup:**
- Model: SSSD^S4 (best-performing, paper default for RQ2–RQ6)
- Sequence lengths: 50, 100, 150, 200
- Blackout sizes: 5, 10, 20, 30, 40
- Full 4×5 = **20 jobs**

**SLURM jobs:** `ali-rq2-S4-s{SEQ}-k{K}` for all combinations.

**Expected result table (RQ2 — Accuracy by blackout size, seq=200):**

| Dataset | k=5 | k=10 | k=20 | k=30 | k=40 |
|---|---|---|---|---|---|
| alibaba | TBD | TBD | TBD | TBD | TBD |

*Hypothesis:* Given the Alibaba vocabulary (7,377 types vs. 87–108 for PTS benchmarks),
accuracy is expected to be lower than PTS datasets but comparable to Apache/ELK, which
also have larger vocabularies. The trend of graceful degradation with increasing
blackout size should hold.

**Expected result table (RQ2 — ROUGE-L by blackout size, seq=200):**

| Dataset | k=5 | k=10 | k=20 | k=30 | k=40 |
|---|---|---|---|---|---|
| alibaba | TBD | TBD | TBD | TBD | TBD |

### 2.3 RQ3 — Diffusion vs. SAITS transformer baseline

**Goal:** Replicate Table 5 of the paper for the Alibaba dataset.  
**Setup:**
- Models: SSSD^S4 vs. SAITS
- Sequence length: 200, blackout size: 10, centered masking
- 200 training epochs for SAITS (paper setup)

**SLURM job:** `ali-rq3-SAITS`

**Expected result table (RQ3):**

| Dataset | Accuracy: SSSD^S4 | Accuracy: SAITS | Perfect Rate: SSSD^S4 | Perfect Rate: SAITS | ROUGE-L: SSSD^S4 | ROUGE-L: SAITS |
|---|---|---|---|---|---|---|
| alibaba | TBD | TBD | TBD | TBD | TBD | TBD |

*Hypothesis:* SSSD^S4 is expected to outperform SAITS on Alibaba, consistent with
every dataset in the paper (SSSD^S4 leads on 10/12 datasets).  The margin may be
particularly large because microservice call sequences exhibit strong long-range
dependencies (a `providerRPC` call implies a prior `consumerRPC`), which S4's long
sequence handling captures better than SAITS's attention window.

### 2.4 RQ5 — Cross-application transfer

**Goal:** Replicate Table 6 of the paper, adding Alibaba as both a training source
and a transfer target.  
**Setup:**
- Model: SSSD^S4, seq=200, blackout=10, centered masking
- Training datasets: alibaba, apache, plaid, elk
- Test datasets: same four + cross-combinations

**SLURM jobs (6 cross-app jobs):**

| Job name | Train | Test |
|---|---|---|
| ali-rq5-ali-to-apache | alibaba | apache |
| ali-rq5-ali-to-plaid | alibaba | plaid |
| ali-rq5-ali-to-elk | alibaba | elk |
| ali-rq5-apache-to-ali | apache | alibaba |
| ali-rq5-plaid-to-ali | plaid | alibaba |
| ali-rq5-elk-to-ali | elk | alibaba |

**Expected result table (RQ5 — cross-app accuracy, rows=train, cols=test):**

| Train ↓ / Test → | alibaba | apache | plaid | elk |
|---|---|---|---|---|
| alibaba | TBD (in-domain) | TBD | TBD | TBD |
| apache | TBD | 92.12 (existing) | — | — |
| plaid | TBD | — | 96.64 (existing) | — |
| elk | TBD | — | — | 88.02 (existing) |

*Hypothesis:* Transfer from system-call datasets (apache, plaid, elk) to the Alibaba
microservice trace is expected to fail near-completely (~1–5% accuracy), because the
event vocabularies are entirely disjoint. Transfer in the reverse direction (alibaba →
system-call datasets) should similarly fail. This finding would **strengthen the
paper's RQ5 conclusion** and make the generalizability limitation concrete in the
context the reviewer cares about (microservices).

---

## 3. Bug Fixes Applied

Five correctness issues were identified and fixed before job submission.

### Fix 1: SAITS API — Wrong constructor kwargs and wrong forward call signature
**Location:** `alibaba-trace/saits_train_infer.py`  
**Problem:** The `SAITS` constructor in `SAITS/modeling/saits.py` requires  
`param_sharing_strategy` and `device` in its `**kwargs` (consumed at lines 50–54),  
but they were missing from the instantiation call.  All three call sites  
(`model(X_hat, missing)`) passed positional tensors, but the actual forward signature  
is `model(inputs: dict, stage: str)` where `inputs` must contain `X`, `missing_mask`,  
`X_holdout`, and `indicating_mask`.  The return value is a `dict`; `imputed_data` is  
accessed by key.  
**Fix:** Added `param_sharing_strategy="inner_group"` and `device=device` to the  
constructor call; replaced all three call sites to pass the correct `inputs` dict and  
`stage` string, and read loss/imputation from the returned dict.

### Fix 2: Masking protocol mismatch between SAITS and diffusion experiments
**Location:** `SSSD/src/utils/util.py`, `SSSD/src/train.py`, `SSSD/src/inference.py`,  
`alliance-canada/make_trace_config.py`  
**Problem:** The paper states (Section 3.4): *"the missing segment is taken from the  
center of the sequence"* as the default. SAITS preparation already used a centered  
blackout. But the diffusion configs defaulted to `masking="bm"`, which calls  
`get_mask_bm()` — a function that randomly picks one of the equal-sized segments  
across the sequence length. During evaluation the same random mask is applied, so  
training and test are internally consistent, but the gap position differs from SAITS's  
fixed center gap, making the RQ3 (SSSD^S4 vs. SAITS) comparison unfair.  
**Fix:** Added `get_mask_cm` (center missing) to `util.py`:
```python
def get_mask_cm(sample, k):
    mask  = torch.ones(sample.shape)
    L     = mask.shape[0]
    start = L // 2 - k // 2
    end   = start + k
    for channel in range(mask.shape[1]):
        mask[start:end, channel] = 0
    return mask
```
Imported it in `train.py` and `inference.py` with an `elif masking == 'cm':` branch.  
Changed the default masking in `make_trace_config.py` from `"bm"` to `"cm"`.  
All three Alibaba SLURM scripts now pass `--masking cm` explicitly.

### Fix 3: Dataset path case mismatch for PLAID
**Location:** `alliance-canada/submit_alibaba_experiments.sh`  
**Problem:** The cross-application submission passed `TEST=PLAID` and `TRAIN=PLAID`,  
but the dataset directory is `TraceReconstruction-main/Datasets/plaid/` (lowercase).  
The cross-app SLURM script uses `$TEST` and `$TRAIN` directly in filesystem paths,  
so the jobs would fail immediately with "file not found".  
**Fix:** Changed all four occurrences of `PLAID` to `plaid` in the submission script.

### Fix 4: Incomplete RQ coverage — CSDI^S4 omitted, RQ2 matrix truncated
**Locations:** `alliance-canada/submit_alibaba_experiments.sh`,  
new `alliance-canada/alibaba_csdis4.slurm`  
**Problem (a):** The comment said "all 4 models" for RQ1 but the loop only covered  
models 0, 1, 2 (DiffWave, SSSD^SA, SSSD^S4). CSDI^S4 was absent.  Because CSDI^S4  
uses a standalone `CSDIS4Imputer` class with its own train/infer API (not compatible  
with the shared `train.py` / `inference.py`), it requires a separate SLURM script.  
**Problem (b):** For RQ2, only `k=10` was submitted for sequence lengths 50/100/150,  
producing 3 jobs instead of the full 4×5=20 needed to replicate Figure 3.  
**Fix (a):** Created `alibaba_csdis4.slurm` which drives `CSDIS4Imputer.train()` and  
then runs a custom inference loop with centered blackout to produce `imputation*.npy`  
output compatible with `evaluate_metrics.py`.  Added CSDI^S4 to Block 1 of the  
submission script (2 jobs: k=1 and k=10).  
**Fix (b):** Replaced the partial RQ2 loop with the full 5×4 matrix (all 5 blackout  
sizes × all 4 sequence lengths = 20 jobs).

### Fix 5: Preprocessing documentation ↔ code mismatch
**Location:** `TraceReconstruction-main/Datasets/alibaba/preprocess_alibaba.py`  
(docstring lines 14–16) and `Datasets/alibaba/README.md` (lines 51–54)  
**Problem:** Both the Python module docstring and the README described the split as  
"80/20 by trace", implying that complete request traces are kept together. The actual  
code splits by **event position** in the concatenated stream (after flattening all  
trace sequences), which is the same strategy used by every other dataset but means  
one trace straddling the split boundary has its calls divided across train and test.  
**Fix:** Updated both the code comment and the README to accurately describe the  
event-position split and note that this is consistent with all other datasets.

---

## 4. SLURM Account Status and Required Action

### Current Status

The `def-naser2` account on Narval has `RawShares=0` and zero recorded CPU/GPU usage  
for this allocation period. The `sbatch` command returns:

```
sbatch: error: You are not associated with any active allocation...
sbatch: error: You are not allowed to submit jobs in this cluster.
```

Both CPU and GPU queues reject submissions. The `cc-debug` (always-available) account  
also rejects submissions, which indicates this is **not a queue or account-name issue**  
but rather that the cluster requires an active Resource Allocation Project (RAP) that  
is currently inactive or expired.

### Root Cause

On the Alliance / Compute Canada clusters, submitting jobs requires an **active RAP**  
approved through the annual Digital Research Alliance of Canada (DRAC) Resources for  
Research Groups (RRG) competition. The last recorded usage in `sshare` is 0, which  
typically means the account's 2025–2026 allocation has not been activated or was not  
renewed.

### Required Action

**The PI (Prof. Ezzati-Jivan) needs to:**
1. Log in at <https://ccdb.alliancecan.ca> and verify that `def-naser2` has an active  
   allocation for 2025–2026.
2. If the RAP has expired, apply for renewal at  
   <https://alliancecan.ca/en/services/advanced-research-computing/accessing-resources/resource-allocation-competition>
3. Alternatively, if a different PI account (e.g., `rrg-*`) has an active allocation,  
   update `submit_alibaba_experiments.sh` line 30 to use that account name:
   ```bash
   --account="<active-account-name>" \
   ```

### Once the Account is Active

Run the single submission script from the project root:
```bash
cd /home/ghazalkh/3
bash alliance-canada/submit_alibaba_experiments.sh
```

This submits **36 jobs total**:
- Block 1 (RQ1): 8 jobs (4 models × 2 blackout sizes)
- Block 2 (RQ2): 20 jobs (SSSD^S4 × 4 seq lengths × 5 blackout sizes)
- Block 3 (RQ3): 1 job (SAITS)
- Block 4 (RQ5): 6 cross-application jobs

Estimated wall-clock time per job: 4–12 hours on one A100 GPU.  
Total GPU-hours: ~250–350 h (submits to the GPU partition, jobs run in parallel).

---

## 5. Expected Results and Paper Integration Strategy

### 5.1 Performance Expectations

Based on the dataset characteristics (vocabulary 7,377 vs. 87–108 for PTS; larger  
vocabulary with repetitive but identifiable patterns — major microservices such as DB  
and cache appear in ~85% of traces):

| Metric | Expected range | Comparable existing dataset |
|---|---|---|
| SSSD^S4 accuracy (k=10) | 75–88 % | Apache (92 %), ELK (88 %) |
| SSSD^S4 perfect rate (k=10) | 45–70 % | ELK (65 %), Apache (68 %) |
| SSSD^S4 ROUGE-L (k=10) | 86–94 % | ELK (95 %), Apache (96 %) |
| SSSD^S4 vs. SAITS margin | +5–15 pp Acc | Average across datasets: +3.13 pp |
| Cross-app: alibaba→syscall | < 5 % | iozone→PLAID: 0.12 % |

### 5.2 How Results Address the Reviewer's Comment

The results directly answer the gap identified by Reviewer 1 in three ways:

**Direct evidence of microservice applicability:** Non-trivial accuracy (expected  
75–88 %) on a production microservice dataset proves the method works on the  
exact type of trace that motivated the paper.

**Evidence of consistent model ranking:** SSSD^S4 is expected to remain the best  
model on this new dataset, reinforcing the paper's Finding RQ1 with a microservice  
trace example.

**Evidence of domain gap:** The cross-app experiment (RQ5) is expected to show that  
models trained on system-call datasets cannot reconstruct microservice call sequences  
(and vice versa), which is a new and interesting finding: domain-specific training  
is important not only across applications within the same domain but also across  
trace modalities (kernel vs. microservice). This strengthens rather than weakens  
the paper's position, showing that the approach generalizes in kind but requires  
domain-matched training data.

### 5.3 Suggested Paper Integration

The Alibaba dataset can be added as **Dataset 13** in Section 3.2 with a short  
subsection alongside Apache, PLAID, and ELK:

```
\subsubsection{Alibaba Microservice Trace}
The Alibaba 2021 microservice trace [CITE] captures 12 hours of production traffic
from over 20,000 microservices across 10,000+ bare-metal nodes. Unlike the other
datasets in our study, which record OS-level system calls on a single host, this
dataset records inter-service API calls within a distributed request, identified
by unique trace IDs. Each call event is encoded as \textit{rpctype:dm} (e.g.,
\texttt{rpc:abc123}) — the communication paradigm paired with the hashed
downstream service identity — yielding a vocabulary of 7,377 distinct call types.
This dataset allows us to evaluate whether the diffusion-based reconstruction
approach extends naturally from single-host kernel traces to distributed
microservice execution traces, directly addressing the broader motivation stated
in our introduction.
```

Results tables for Alibaba can then be inserted into each existing result table as an  
additional row, labeled "alibaba", with a dagger footnote explaining the different  
event type for clarity.

---

## 6. File Inventory

All new/modified files:

| File | Status | Purpose |
|---|---|---|
| `alibaba-trace/raw/MSCallGraph_0.csv` | Created (downloaded) | Raw 6M-row call graph data |
| `alibaba-trace/raw/MSCallGraph_0.tar.gz` | Created (downloaded) | Compressed source |
| `alibaba-trace/README.md` | Created | Raw data documentation |
| `alibaba-trace/evaluate_metrics.py` | Created | Acc / PerfRate / ROUGE-L from npy files |
| `alibaba-trace/prepare_saits_dataset.py` | Created | npy→h5 with centered blackout for SAITS |
| `alibaba-trace/saits_train_infer.py` | Created + **Fixed** | SAITS training & inference driver |
| `TraceReconstruction-main/Datasets/alibaba/preprocess_alibaba.py` | Created + **Fixed** | Alibaba→npy preprocessing |
| `TraceReconstruction-main/Datasets/alibaba/README.md` | Created + **Fixed** | Dataset documentation |
| `TraceReconstruction-main/Datasets/alibaba/alibaba_calls.txt` | Created | 6M token stream |
| `TraceReconstruction-main/Datasets/alibaba/sequence_length_{50,100,150,200}/` | Created | Preprocessed numpy datasets |
| `SSSD/src/utils/util.py` | **Modified** | Added `get_mask_cm` (centered masking) |
| `SSSD/src/train.py` | **Modified** | Import + `elif masking=='cm'` branch |
| `SSSD/src/inference.py` | **Modified** | Import + `elif masking=='cm'` branch |
| `alliance-canada/make_trace_config.py` | **Modified** | `--model` flag; default masking `cm`; SSSD^SA config |
| `alliance-canada/alibaba_diffusion.slurm` | Created | DiffWave / SSSD^SA / SSSD^S4 training |
| `alliance-canada/alibaba_saits.slurm` | Created | SAITS training & inference |
| `alliance-canada/alibaba_csdis4.slurm` | Created | CSDI^S4 standalone driver |
| `alliance-canada/alibaba_cross_app.slurm` | Created | Cross-app transfer jobs |
| `alliance-canada/submit_alibaba_experiments.sh` | Created + **Fixed** | Master job submission (36 jobs) |
