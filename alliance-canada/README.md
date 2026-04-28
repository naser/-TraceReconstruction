# Alliance Canada GPU Run Guide

This folder gives you a working starting point for running the trace reconstruction training on an Alliance Canada GPU node.

## What it assumes

- Your repo is copied to the cluster, ideally under `scratch`
- You want to train `SSSD^{S4}` on one of the prepared trace datasets in `TraceReconstruction-main/Datasets`
- You are using a single GPU job

## Recommended layout on the cluster

```bash
mkdir -p $SCRATCH/trace-reconstruction
rsync -av --exclude __pycache__ /path/to/this/project/ $SCRATCH/trace-reconstruction/
```

## Submit a job

From the repo root on the cluster:

```bash
sbatch --account=def-yourpi \
  --export=ALL,REPO_DIR=$SCRATCH/trace-reconstruction,DATASET=stream,SEQ_LEN=200,MISSING_K=10 \
  alliance-canada/train_trace_sssds4.slurm
```

You can change:

- `DATASET`: `stream`, `ffmpeg`, `compress-gzip`, `plaid`, `elk`, `apache`, etc.
- `SEQ_LEN`: `50`, `100`, `150`, or `200`
- `MISSING_K`: blackout or missing-span size such as `1`, `5`, `10`, `20`, `30`, `40`
- `TRAIN_ITERS`: defaults to `10000`
- `BATCH_SIZE`: defaults to `64`

## Outputs

Checkpoints and generated imputations are written under:

```bash
$SCRATCH/trace-reconstruction-results/<dataset>_seq<seq_len>_k<missing_k>/
```

## Notes

- The job builds the optional CUDA cauchy extension inside the allocated GPU job so it does not depend on login-node CUDA detection.
- The trace training code in this repo has been adjusted to batch the `train.npy` files correctly; the original hardcoded split did not fit the `10000`-sample trace datasets.
- If you hit GPU memory pressure, resubmit with a smaller `BATCH_SIZE`, for example `32`.
