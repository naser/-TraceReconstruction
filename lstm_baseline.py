# LSTM baseline for trace event reconstruction (next-event prediction).
# Implements the comparison baseline described in RQ3 of the IEEE Transactions paper
# "Leveraging Diffusion Models for Execution Trace Reconstruction".
#
# Usage:
#   python lstm_baseline.py --dataset compress-gzip --seq_len 200 --blackout 10
#
# The model takes the preceding (seq_len - blackout) events as context and
# predicts the next event autoregressively until the blackout is filled.

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",   required=True, help="Dataset name, e.g. compress-gzip")
parser.add_argument("--seq_len",   type=int, default=200)
parser.add_argument("--blackout",  type=int, default=10,
                    help="Number of events to reconstruct (blackout size)")
parser.add_argument("--data_root", default=os.path.join(os.path.dirname(__file__),
                                            "TraceReconstruction-main", "Datasets"),
                    help="Path to the Datasets folder")
parser.add_argument("--epochs",    type=int, default=30)
parser.add_argument("--lr",        type=float, default=1e-3)
parser.add_argument("--hidden",    type=int, default=256)
parser.add_argument("--layers",    type=int, default=2)
parser.add_argument("--batch",     type=int, default=64)
parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

DEVICE = torch.device(args.device)
print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
data_dir = os.path.join(args.data_root, args.dataset, f"sequence_length_{args.seq_len}")
train_npy = np.load(os.path.join(data_dir, "train.npy"))  # (10000, seq_len, 1)
test_npy  = np.load(os.path.join(data_dir, "test.npy"))   # (500,   seq_len, 1)

train_data = torch.tensor(train_npy[:, :, 0], dtype=torch.long)  # (10000, seq_len)
test_data  = torch.tensor(test_npy[:, :, 0],  dtype=torch.long)  # (500,   seq_len)

vocab_size = int(train_data.max().item()) + 1
print(f"Dataset: {args.dataset}  |  seq_len={args.seq_len}  |  vocab={vocab_size}")

# Context: all events before the blackout region (paper: preceding 199 events
# when blackout=1, giving the model seq_len-1 context tokens).
context_len = args.seq_len - args.blackout  # events available as context

# For training: predict each event given its prefix (teacher forcing)
# Input: [e_0, ..., e_{T-2}]  Target: [e_1, ..., e_{T-1}]
X_train = train_data[:, :-1]  # (N, seq_len-1)
Y_train = train_data[:, 1:]   # (N, seq_len-1)

loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=args.batch, shuffle=True)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LSTMPredictor(nn.Module):
    def __init__(self, vocab_size, hidden, layers, embed_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm  = nn.LSTM(embed_dim, hidden, num_layers=layers,
                             batch_first=True, dropout=0.1 if layers > 1 else 0)
        self.out   = nn.Linear(hidden, vocab_size)

    def forward(self, x, state=None):
        emb = self.embed(x)
        out, state = self.lstm(emb, state)
        return self.out(out), state

model = LSTMPredictor(vocab_size, args.hidden, args.layers).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
best_acc = 0.0
for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits, _ = model(xb)
        loss = criterion(logits.reshape(-1, vocab_size), yb.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    # Quick evaluation on test set with blackout reconstruction
    model.eval()
    with torch.no_grad():
        ctx = test_data[:, :context_len].to(DEVICE)          # (500, context_len)
        ground_truth = test_data[:, context_len:].to(DEVICE)  # (500, blackout)

        # Feed context through LSTM
        _, state = model.lstm(model.embed(ctx))

        # Autoregressively predict blackout events
        curr_token = ctx[:, -1:]                 # (500, 1)
        preds = []
        for _ in range(args.blackout):
            logit, state = model(curr_token, state)
            pred = logit[:, -1, :].argmax(dim=-1, keepdim=True)  # (500, 1)
            preds.append(pred)
            curr_token = pred
        preds = torch.cat(preds, dim=1)          # (500, blackout)

        correct = (preds == ground_truth).float().mean().item()
        if correct > best_acc:
            best_acc = correct

    print(f"Epoch {epoch:3d}/{args.epochs}  loss={total_loss/len(loader):.4f}  "
          f"acc={correct*100:.2f}%  best={best_acc*100:.2f}%")

print(f"\nFinal reconstruction accuracy (blackout={args.blackout}): {best_acc*100:.2f}%")
