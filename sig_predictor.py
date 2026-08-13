#!/usr/bin/env python3
"""Block‑signature RNN — predicts the row‑1 pattern for a given order.

The row‑1 "block signature" encodes the column block decomposition used
by the construction method.  Trained on ~800 known matrices, this model
predicts signature features (run‑lengths, block sizes, spectral
properties) from the order alone, enabling signature‑guided search for
unseen orders.
"""

import math, os, time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

CSV_DIR = Path.home() / "open_hadamard"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def order_features(n):
    """Feature vector derived from the order n = 4k."""
    k = n // 4
    feats = [
        float(n), float(n) / 2000.0,
        float(k), float(k) / 500.0,
        float(n & 1), float((n & (n - 1)) == 0),
        float(n % 8) / 8.0, float(n % 3) / 3.0,
        float(math.log(n + 1)),
    ]
    return np.array(feats, dtype=np.float32)


def signature_features(row, max_runs=32):
    """Extract compressed signature from row 1 of a Hadamard matrix."""
    n = len(row)
    changes = np.where(np.diff(row) != 0)[0] + 1
    run_lens = np.diff(np.concatenate([[0], changes, [n]]))
    signs = row[np.concatenate([[0], changes])]

    # signed run lengths
    srl = np.zeros(max_runs, dtype=np.float32)
    m = min(len(run_lens), max_runs)
    for i in range(m):
        srl[i] = float(signs[i]) * run_lens[i]

    # spectral flatness
    target = float(n)
    pw = np.abs(np.fft.fft(row.astype(np.float64))) ** 2
    flatness = float(np.sum((pw - target) ** 2))

    # autocorrelation at lags 1..16
    acorr = np.zeros(16, dtype=np.float32)
    for lag in range(1, min(17, n)):
        acorr[lag - 1] = float(np.dot(row[:n - lag], row[lag:]))

    # block statistics
    pos_runs = [rl for s, rl in zip(signs, run_lens) if s > 0]
    neg_runs = [rl for s, rl in zip(signs, run_lens) if s < 0]
    pos_avg = float(np.mean(pos_runs)) if pos_runs else 0.0
    neg_avg = float(np.mean(neg_runs)) if neg_runs else 0.0
    pos_max = float(max(pos_runs)) if pos_runs else 0.0
    neg_max = float(max(neg_runs)) if neg_runs else 0.0
    n_runs = float(len(run_lens))

    block_stats = np.array([pos_avg, neg_avg, pos_max, neg_max, n_runs,
                             flatness, flatness / (n**3)], dtype=np.float32)

    return np.concatenate([srl, acorr, block_stats])


def method_onehot(order):
    """One‑hot construction method from order properties."""
    from hoa64.hadamard import _is_prime_power
    v = np.zeros(8, dtype=np.float32)
    if order <= 2: v[0] = 1.0
    elif (order & (order - 1)) == 0: v[1] = 1.0
    else:
        q = order - 1
        if q >= 3 and q % 4 == 3 and _is_prime_power(q): v[2] = 1.0
        q2 = order // 2 - 1
        if order % 2 == 0 and q2 and q2 >= 5 and q2 % 4 == 1 and _is_prime_power(q2):
            v[3] = 1.0
        if 668 <= order <= 1964: v[4] = 1.0
        if order % 4 == 0 and _is_prime_power(order // 4) and (order // 4) % 4 == 1:
            v[5] = 1.0
    if v.sum() == 0: v[6] = 1.0  # Kronecker / other
    return v


SIG_FEAT_DIM = 32 + 16 + 7  # runs + acorr + block_stats = 55
INPUT_DIM = 9 + 8            # order features + method onehot = 17


class SigDataset(Dataset):
    def __init__(self, csv_files, max_samples=None):
        self.inputs = []
        self.targets = []
        for p in csv_files:
            try:
                H = np.loadtxt(str(p), delimiter=",", dtype=np.int8)
                order = H.shape[0]
                if order < 4: continue
                r1 = H[1]  # row 1 = block signature
                inp = np.concatenate([order_features(order), method_onehot(order)])
                tgt = signature_features(r1.astype(np.float64))
                self.inputs.append(inp)
                self.targets.append(tgt)
                if max_samples and len(self.inputs) >= max_samples:
                    break
            except Exception:
                continue
        self.inputs = torch.tensor(np.stack(self.inputs), dtype=torch.float32)
        self.targets = torch.tensor(np.stack(self.targets), dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return self.inputs[i], self.targets[i]


class SigPredictor(nn.Module):
    def __init__(self, input_dim=17, hidden=128, output_dim=SIG_FEAT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_sig_model():
    csv_files = sorted(CSV_DIR.glob("hadamard_*.csv"))
    dataset = SigDataset(csv_files, max_samples=600)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"Training on {len(dataset)} signatures")

    model = SigPredictor().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=0.002)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(100):
        total = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total += loss.item()
        if (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch+1:3d}: loss={total/len(loader):.5f}")

    model_path = CSV_DIR / "sig_predictor.pt"
    torch.save(model.state_dict(), str(model_path))
    print(f"Model saved to {model_path}")
    return model


def predict_signature(model, order):
    """Predict block signature features for a given order."""
    import numpy as np
    inp = np.concatenate([order_features(order), method_onehot(order)])
    x = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        pred = model(x).cpu().numpy()[0]
    return pred  # shape (55,)


if __name__ == "__main__":
    model = train_sig_model()
    # test prediction
    for n in [92, 268, 1212]:
        pred = predict_signature(model, n)
        runs_pred = pred[:6]
        print(f"  H({n:5d}): predicted runs={runs_pred}")
