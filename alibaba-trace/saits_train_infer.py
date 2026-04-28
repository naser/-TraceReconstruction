"""
saits_train_infer.py  ─  Self-contained SAITS training & inference for trace
reconstruction.  Produces imputation*.npy / original*.npy / mask*.npy output
files in the same format as the SSSD inference.py so that evaluate_metrics.py
can process them directly.

Usage (called automatically from the SLURM script):
    python saits_train_infer.py \
        --h5-path  /path/to/dataset.h5 \
        --results-dir /path/to/saits_results \
        --seq-len 200 \
        --blackout 10 \
        --epochs 200
"""

import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn
import h5py

# ── SAITS path  ─────────────────────────────────────────────────────────────
REPO_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAITS_SRC  = os.path.join(REPO_DIR, "SAITS")
sys.path.insert(0, SAITS_SRC)

from modeling.saits import SAITS as SAITSModel   # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
class H5BlackoutDataset(torch.utils.data.Dataset):
    """Dataset that loads X from the h5 train group and applies MIT masking."""

    def __init__(self, h5_path, group, blackout_k, seq_len):
        with h5py.File(h5_path, "r") as hf:
            self.X = torch.tensor(hf[group]["X"][:], dtype=torch.float32)
            if group in ("val", "test"):
                self.X_hat = torch.tensor(hf[group]["X_hat"][:],           dtype=torch.float32)
                self.missing = torch.tensor(hf[group]["missing_mask"][:],  dtype=torch.float32)
                self.indicating = torch.tensor(hf[group]["indicating_mask"][:], dtype=torch.float32)
            else:
                # Training: generate blackout masks on the fly in __getitem__
                self.X_hat = None
                self.blackout_k = blackout_k
                self.seq_len    = seq_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]   # (T, 1)
        seq_len = x.shape[0]

        if self.X_hat is None:
            # Create random-position blackout for MIT training
            centre = seq_len // 2
            start  = centre - self.blackout_k // 2
            end    = start + self.blackout_k
            mask = torch.ones(seq_len, 1)
            mask[start:end] = 0.0
            x_hat = x * mask
            indicating = 1.0 - mask
            return {"X": x, "X_hat": x_hat,
                    "missing_mask": mask, "indicating_mask": indicating}
        else:
            return {"X":               x,
                    "X_hat":           self.X_hat[idx],
                    "missing_mask":    self.missing[idx],
                    "indicating_mask": self.indicating[idx]}


def masked_mae(imp, X, mask):
    """MAE only over the indicated (artificially missing) positions."""
    diff = (imp - X).abs() * mask
    total = mask.sum()
    return diff.sum() / (total + 1e-8)


def train_saits(h5_path, results_dir, seq_len, blackout_k, epochs, device):
    train_ds = H5BlackoutDataset(h5_path, "train", blackout_k, seq_len)
    val_ds   = H5BlackoutDataset(h5_path, "val",   blackout_k, seq_len)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128,
                                               shuffle=True, drop_last=False)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=128,
                                               shuffle=False, drop_last=False)

    feature_num = 1

    # param_sharing_strategy and device are required kwargs consumed by
    # saits.py __init__ via **kwargs (lines 50-54 of the local SAITS source).
    model = SAITSModel(
        n_groups=1,
        n_group_inner_layers=1,
        d_time=seq_len,
        d_feature=feature_num,
        d_model=256,
        d_inner=256,
        n_head=4,
        d_k=64,
        d_v=64,
        dropout=0.0,
        diagonal_attention_mask=True,
        input_with_mask=True,
        MIT=True,
        ORT=True,
        param_sharing_strategy="inner_group",
        device=device,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            X           = batch["X"].to(device)
            X_hat       = batch["X_hat"].to(device)
            missing     = batch["missing_mask"].to(device)
            indicating  = batch["indicating_mask"].to(device)

            opt.zero_grad()
            # SAITS forward signature: model(inputs: dict, stage: str)
            # inputs must contain X, missing_mask, X_holdout, indicating_mask.
            inputs = {
                "X":               X_hat,
                "missing_mask":    missing,
                "X_holdout":       X,
                "indicating_mask": indicating,
            }
            result = model(inputs, stage="train")
            # SAITS returns a dict; use the combined reconstruction + imputation loss
            loss = result["reconstruction_loss"] + result["imputation_loss"]
            loss.backward()
            opt.step()

        # Validation
        model.eval()
        val_maes = []
        with torch.no_grad():
            for batch in val_loader:
                X           = batch["X"].to(device)
                X_hat       = batch["X_hat"].to(device)
                missing     = batch["missing_mask"].to(device)
                indicating  = batch["indicating_mask"].to(device)

                inputs = {
                    "X":               X_hat,
                    "missing_mask":    missing,
                    "X_holdout":       X,
                    "indicating_mask": indicating,
                }
                result = model(inputs, stage="val")
                val_maes.append(result["imputation_MAE"].item())

        val_mae = float(np.mean(val_maes))
        if epoch % 20 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:4d}/{epochs}  val_MAE={val_mae:.4f}", flush=True)

        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"  Best val MAE: {best_val:.4f}")
    model.load_state_dict(best_state)
    ckpt = os.path.join(results_dir, "best_model.pt")
    torch.save(best_state, ckpt)
    print(f"  Saved model → {ckpt}")
    return model


def infer_saits(model, h5_path, results_dir, blackout_k, device):
    test_ds = H5BlackoutDataset(h5_path, "test", blackout_k, seq_len=None)
    loader  = torch.utils.data.DataLoader(test_ds, batch_size=125,
                                          shuffle=False, drop_last=False)
    model.eval()
    all_imp, all_orig, all_mask = [], [], []

    with torch.no_grad():
        for batch in loader:
            X_hat      = batch["X_hat"].to(device)
            missing    = batch["missing_mask"].to(device)
            indicating = batch["indicating_mask"].to(device)

            # SAITS test stage: no imputation loss is computed, just imputed_data
            inputs = {
                "X":               X_hat,
                "missing_mask":    missing,
                "X_holdout":       X_hat,        # placeholder; not used in test
                "indicating_mask": indicating,
            }
            result  = model(inputs, stage="test")
            imputed = result["imputed_data"]
            imp  = imputed.cpu().numpy()              # (B, T, 1)
            orig = batch["X"].numpy()                 # (B, T, 1)
            msk  = batch["missing_mask"].numpy()      # (B, T, 1)  1=observed

            # convert to SSSD layout (B, 1, T)
            all_imp .append(imp .transpose(0, 2, 1))
            all_orig.append(orig.transpose(0, 2, 1))
            all_mask.append(msk .transpose(0, 2, 1))

    n = len(all_imp)
    for i, (imp, orig, msk) in enumerate(zip(all_imp, all_orig, all_mask)):
        np.save(os.path.join(results_dir, f"imputation{i}.npy"), imp)
        np.save(os.path.join(results_dir, f"original{i}.npy"),   orig)
        np.save(os.path.join(results_dir, f"mask{i}.npy"),        msk)

    print(f"  Saved {n} batch(es) of inference results in {results_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5-path",     required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--seq-len",     type=int, default=200)
    ap.add_argument("--blackout",    type=int, default=10)
    ap.add_argument("--epochs",      type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    print("Training SAITS …", flush=True)
    global seq_len
    seq_len = args.seq_len
    model = train_saits(args.h5_path, args.results_dir,
                        args.seq_len, args.blackout, args.epochs, device)

    print("Running inference …", flush=True)
    infer_saits(model, args.h5_path, args.results_dir, args.blackout, device)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
