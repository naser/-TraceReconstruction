"""
prepare_saits_dataset.py  ─  Convert trace npy dataset (blackout format) to
the h5 format expected by SAITS.

SAITS training uses MIT (Masked Imputation Task) so the training set just needs
X without pre-applied masks.  For the test set we need X, X_hat (with the
centre blackout zeroed out), missing_mask and indicating_mask.

Usage:
    python prepare_saits_dataset.py \
        --npy-train /path/to/train.npy \
        --npy-test  /path/to/test.npy  \
        --blackout  10                 \
        --output    /path/to/dataset.h5
"""

import argparse, math
import numpy as np
import h5py


def make_blackout_masks(n_seqs, seq_len, feature_num, blackout_k):
    """Create centre blackout masks.
    Returns indicating_mask: 1 where artificially hidden (the blackout region).
             missing_mask:   0 where hidden, 1 where observed.
    """
    centre = seq_len // 2
    start  = centre - blackout_k // 2
    end    = start + blackout_k   # [start, end)

    indicating = np.zeros((n_seqs, seq_len, feature_num), dtype=np.float32)
    indicating[:, start:end, :] = 1.0

    missing = 1.0 - indicating
    return missing, indicating


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy-train", required=True)
    ap.add_argument("--npy-test",  required=True)
    ap.add_argument("--blackout",  type=int, default=10)
    ap.add_argument("--output",    required=True)
    args = ap.parse_args()

    train = np.load(args.npy_train).astype(np.float32)  # (N_train, T, 1)
    test  = np.load(args.npy_test ).astype(np.float32)  # (N_test,  T, 1)

    # SSSD stores as (N, 1, T) → transpose to SAITS convention (N, T, 1)
    if train.shape[1] == 1 and train.shape[2] > 1:
        train = train.transpose(0, 2, 1)
    if test.shape[1] == 1 and test.shape[2] > 1:
        test  = test.transpose(0, 2, 1)

    n_tr, seq_len, feature_num = train.shape
    n_te = test.shape[0]

    # ── Training set  ───────────────────────────────────────────────────────
    # For MIT training, SAITS only needs X (no pre-applied masks)
    empirical_mean = np.nanmean(train.reshape(-1, feature_num), axis=0)

    # ── Test set  ───────────────────────────────────────────────────────────
    missing_mask, indicating_mask = make_blackout_masks(
        n_te, seq_len, feature_num, args.blackout)

    X_hat        = test.copy()
    X_hat[indicating_mask == 1] = 0.0   # zero-fill the blackout region

    # ── Write h5  ───────────────────────────────────────────────────────────
    with h5py.File(args.output, "w") as hf:
        # train
        tr = hf.create_group("train")
        tr.create_dataset("X",                    data=train)
        tr.create_dataset("empirical_mean_for_GRUD", data=empirical_mean)

        # val  (use last 10 % of training as validation)
        val_start = int(n_tr * 0.9)
        vl = hf.create_group("val")
        vl_missing, vl_indicating = make_blackout_masks(
            n_tr - val_start, seq_len, feature_num, args.blackout)
        vl_X_hat = train[val_start:].copy()
        vl_X_hat[vl_indicating == 1] = 0.0
        vl.create_dataset("X",                data=train[val_start:])
        vl.create_dataset("X_hat",            data=vl_X_hat)
        vl.create_dataset("missing_mask",     data=1.0 - vl_indicating)
        vl.create_dataset("indicating_mask",  data=vl_indicating)

        # test
        te = hf.create_group("test")
        te.create_dataset("X",               data=test)
        te.create_dataset("X_hat",           data=X_hat)
        te.create_dataset("missing_mask",    data=missing_mask)
        te.create_dataset("indicating_mask", data=indicating_mask)

    print(f"Wrote {args.output}")
    print(f"  train: {n_tr} × {seq_len} × {feature_num}  "
          f"(val split from last {n_tr - val_start})")
    print(f"  test:  {n_te} × {seq_len} × {feature_num}  "
          f"blackout=[{seq_len//2 - args.blackout//2}, "
          f"{seq_len//2 - args.blackout//2 + args.blackout})")


if __name__ == "__main__":
    main()
