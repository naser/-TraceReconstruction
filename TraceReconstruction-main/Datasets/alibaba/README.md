# Alibaba Microservice Trace Dataset

## Source
Luo et al., "Characterizing Microservice Dependency and Performance: Alibaba Trace Analysis"  
ACM Symposium on Cloud Computing (SoCC), 2021.  
https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2021

This dataset is recommended by Reviewer 1 as evidence that the trace reconstruction
approach applies to microservice architectures (not only OS-level system calls).

## What This Dataset Is
The Alibaba 2021 microservice trace captures 12 hours of production traffic from
over 20,000 microservices across 10,000+ bare-metal nodes. Each "call graph"
records a complete web request as a tree of microservice-to-microservice calls,
identified by a unique `traceid`.

The four data tables are:
| Table | Description | Size |
|---|---|---|
| MSCallGraph | Call graphs (traceid, caller, callee, interface, rpctype, rt) | 25 GB |
| MSResource | Container CPU/memory utilization per microservice | 16 GB |
| MSRTQps | Call rate and response time metrics | 19 GB |
| Node | Bare-metal node CPU/memory utilization | 1.1 GB |

**For trace reconstruction, only `MSCallGraph` is used.**

## Raw Data
The raw data lives in `alibaba-trace/raw/` in the workspace root:

```
alibaba-trace/raw/
    MSCallGraph_0.csv        # ~1.3 GB uncompressed (~154 MB compressed)
    MSCallGraph_0.tar.gz     # keep for reference / re-extraction
```

`MSCallGraph_0.tar.gz` was downloaded from:
```
http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2021MicroservicesTraces/MSCallGraph/MSCallGraph_0.tar.gz
```

Additional files (MSCallGraph_1 through MSCallGraph_144) are available at the
same URL pattern if more data is needed; a single file already provides ~6M call
records across ~145,000 traces — well above the 10,500 sequences needed.

## Call-Type Encoding (Analogous to System Calls)
Each call event is represented as `<rpctype>:<dm>` e.g. `rpc:abc123…` or
`mc:def456…`, where:
- `rpctype` ∈ {rpc, http, mc, db, mq, userDefined} — communication paradigm
- `dm` — hashed identity of the downstream microservice (always present)

This gives a vocabulary similar in nature to OS system call names.

## Train / Test Split
The ~130K traces are sorted by arrival order, flattened into a single event
stream, then split 80 / 20 **by event position** (first 80 % → train pool,
last 20 % → test pool). This is identical to the split used for Apache, ELK,
PLAID, and all PTS benchmarks in this project. A trace whose calls span the
split boundary is split across pools — the same accepted trade-off as every
other dataset. Sliding windows are then applied to each pool independently.

## Preprocessing
Run once from the project root:

```bash
python TraceReconstruction-main/Datasets/alibaba/preprocess_alibaba.py
```

This creates:
```
TraceReconstruction-main/Datasets/alibaba/
    alibaba_calls.txt               # raw token stream (one call per line)
    sequence_length_50/
        train.npy  test.npy  training  testing
    sequence_length_100/
        train.npy  test.npy  training  testing
    sequence_length_150/
        train.npy  test.npy  training  testing
    sequence_length_200/
        train.npy  test.npy  training  testing
```

Output arrays have shape `(N, seq_len, 1)` with dtype `int32`, matching the
format used by Apache, ELK, PLAID, and the PTS benchmarks.

## Notes
- `interface` field is non-empty for ~50% of calls (RPC/HTTP); absent for MC/DB.
  Using `rpctype:dm` ensures a fully-populated token for every call.
- Sequence counts: 10,000 training + 500 test per sequence length.
- This dataset exercises the diffusion model on *distributed-system* traces
  (microservice API calls) rather than single-host kernel system calls.
