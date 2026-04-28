# Alibaba Trace — Raw Data

This folder holds the raw MSCallGraph data from the
Alibaba 2021 Microservice Trace dataset.

## Contents

| File | Size | Description |
|---|---|---|
| `MSCallGraph_0.csv` | ~1.3 GB | Extracted call graph records |
| `MSCallGraph_0.tar.gz` | ~154 MB | Original compressed archive |

## Download

Files were fetched from Alibaba OSS:

```
http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2021MicroservicesTraces/MSCallGraph/MSCallGraph_0.tar.gz
```

To download additional files (if more training data is needed):

```bash
BASE="http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2021MicroservicesTraces/MSCallGraph"
for i in $(seq 1 144); do
    wget -c "${BASE}/MSCallGraph_${i}.tar.gz"
    tar -xzf "MSCallGraph_${i}.tar.gz"
done
```

## CSV Schema

Columns: `(index), traceid, timestamp, rpcid, um, rpctype, dm, interface, rt`

| Column | Description |
|---|---|
| `traceid` | Unique ID for a complete request call graph |
| `timestamp` | Milliseconds from experiment start (0–43200000) |
| `rpcid` | Dotted hierarchy ID (e.g. `0.1.2.3`) showing call depth/position |
| `um` | Upstream (calling) microservice — hashed |
| `rpctype` | Call paradigm: `rpc`, `http`, `mc`, `db`, `mq`, `userDefined` |
| `dm` | Downstream (called) microservice — hashed |
| `interface` | API endpoint called (empty for mc/db calls) |
| `rt` | Response time in ms |

## Dataset Reference

> Luo, Shutian, et al.  
> "Characterizing Microservice Dependency and Performance: Alibaba Trace Analysis."  
> *Proceedings of the ACM Symposium on Cloud Computing*, 2021. pp. 412–426.
