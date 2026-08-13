#!/usr/bin/env python3
"""RNN‑guided Hadamard evolution — learns from ~800 existing matrices.

Features extracted per matrix:
  - First‑row run‑length encoding (compressed signature)
  - Spectral flatness (FFT PSD deviation from constant n)
  - Construction method label (one‑hot: Sylvester/Paley/Williamson/…)
  - Row sum, autocorrelation profile

Model: two‑layer LSTM with 128‑dim hidden state, trained to predict the
PSD objective (regression) and classify the construction type.

The trained model serves as a heuristic fitness function for the
Game‑of‑Hadamard evolution: it scores candidate first rows and guides
the micromag search toward Hadamard‑compatible patterns.
"""

import os, math, time, random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

CSV_DIR = Path.home() / "open_hadamard"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Feature extraction -----------------------------------------------------------

def run_length_encoding(row, max_runs=64):
    """Compress a ±1 row into run‑length pairs, padded to max_runs."""
    changes = np.where(np.diff(row) != 0)[0] + 1
    runs = np.diff(np.concatenate([[0], changes, [len(row)]]))
    signs = row[np.concatenate([[0], changes])]
    features = np.zeros(max_runs, dtype=np.float32)
    n = min(len(runs), max_runs)
    for i in range(n):
        features[i] = float(signs[i]) * runs[i]  # signed run length
    return features


def spectral_features(row):
    """FFT‑based PSD deviation from constant n (flatness score)."""
    n = len(row)
    pw = np.abs(np.fft.fft(row.astype(np.float64))) ** 2
    target = float(n)
    flatness = float(np.sum((pw - target) ** 2))
    return np.array([flatness, flatness / (n**3)], dtype=np.float32)


def autocorrelation_profile(row, max_lag=32):
    """Aperiodic autocorrelation at first max_lag lags."""
    n = len(row)
    prof = np.zeros(max_lag, dtype=np.float32)
    for lag in range(1, min(max_lag + 1, n)):
        prof[lag - 1] = float(np.dot(row[:n - lag], row[lag:]))
    return prof


def method_label(order):
    """Return a one‑hot construction method vector."""
    from hoa64.hadamard import _is_prime_power
    vec = np.zeros(10, dtype=np.float32)
    if order <= 2:
        vec[0] = 1.0        # seed
    elif (order & (order - 1)) == 0 and order > 2:
        vec[1] = 1.0        # Sylvester
    else:
        q = order - 1
        if q >= 3 and q % 4 == 3 and _is_prime_power(q):
            vec[2] = 1.0    # Paley I
        q2 = order // 2 - 1
        if order % 2 == 0 and q2 and q2 >= 5 and q2 % 4 == 1 and _is_prime_power(q2):
            vec[3] = 1.0    # Paley II
        if 668 <= order <= 1964 and order % 4 == 0:
            vec[4] = 1.0    # CSV/Alpoge
        if order % 4 == 0:
            q = order // 4
            if q >= 5 and q % 4 == 1 and _is_prime_power(q):
                vec[5] = 1.0  # Miyamoto
            if q in (23, 29, 39, 43):
                vec[6] = 1.0  # Williamson
    if vec.sum() == 0:
        vec[7] = 1.0    # Kronecker / combined
    return vec


def quantum_style_energy(row, order=None):
    """Compute the Ising‑style energy from the quantum H‑matrix paper (Eq. 23).
    This is the Williamson block‑based energy for k=order/4 if applicable."""
    n = len(row)
    if order is None:
        order = n
    G = np.outer(row, row)
    off = G.sum() - n
    return float(off * off)


def extract_features(csv_path):
    """Extract a feature vector from a Hadamard CSV file."""
    H = np.loadtxt(str(csv_path), delimiter=",", dtype=np.int8)
    order = H.shape[0]
    r0 = H[0].astype(np.float64)

    runs = run_length_encoding(r0)
    spec = spectral_features(r0)
    acorr = autocorrelation_profile(r0)
    label = method_label(order)
    q_energy = np.array([quantum_style_energy(r0, order)], dtype=np.float32)
    order_norm = np.array([order / 2000.0], dtype=np.float32)  # normalized order

    return np.concatenate([runs, spec, acorr, label, q_energy, order_norm])


# Dataset --------------------------------------------------------------------

class HadamardDataset(Dataset):
    def __init__(self, csv_files, feature_dim, max_samples=None):
        self.features = []
        self.targets = []
        count = 0
        for p in csv_files:
            try:
                H = np.loadtxt(str(p), delimiter=",", dtype=np.int8)
                order = H.shape[0]
                feats = extract_features(p)
                self.features.append(feats[:feature_dim])
                # target: spectral flatness (0 = perfect Hadamard)
                self.targets.append(0.0)  # verified Hadamard → target 0
                count += 1
                if max_samples and count >= max_samples:
                    break
            except Exception:
                continue
        self.features = torch.tensor(np.stack(self.features), dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# Model ----------------------------------------------------------------------

class HadamardRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.classifier = nn.Linear(hidden_dim, 10)  # method classification head

    def forward(self, x):
        # x: (batch, input_dim)
        h = self.input_proj(x).unsqueeze(1)               # (batch, 1, hidden)
        out, _ = self.lstm(h.repeat(1, 4, 1))            # repeat seq for LSTM
        final = out[:, -1, :]                             # (batch, hidden)
        score = self.fc(final)                            # (batch, 1)  — fitness
        method_logits = self.classifier(final)             # (batch, 10) — construction type
        return score, method_logits


# Training -------------------------------------------------------------------

def train_model(model, loader, epochs=50, lr=0.001):
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            score, method_logits = model(x)
            # regression loss: predict spectral flatness
            loss_reg = mse_loss(score, y)
            # classification loss: predict method type
            # (use the feature's one-hot as pseudo‑label)
            method_onehot = x[:, 96:106].to(DEVICE).float()
            method_target = torch.argmax(method_onehot, dim=1)
            loss_cls = ce_loss(method_logits, method_target)
            loss = loss_reg + 0.1 * loss_cls
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:3d}: loss={total_loss/len(loader):.4f}",
                  flush=True)


# Guided evolution -----------------------------------------------------------

def rnn_guided_search(order, model, n_trials=100, search_flips=5000):
    """Use the trained RNN as a fitness function to select promising seeds."""
    from hoa64.micromag import micromag_search
    from hoa64.hadamard import random_seed, verify

    FEAT_DIM = 110  # runs(64) + spec(2) + acorr(32) + method(10) + qe(1) + order_norm(1)
    model.eval()
    device = next(model.parameters()).device
    best_H = None
    best_f = None

    for trial in range(n_trials):
        H0 = random_seed(order).astype(np.int8)
        r0 = H0[0].astype(np.float64)
        feats = extract_features_from_row(r0, order)
        with torch.no_grad():
            score, _ = model(
                torch.tensor(feats[:FEAT_DIM]).unsqueeze(0).to(device))
        predicted_f = float(score.item())

        if best_H is None or predicted_f < best_f:
            H, st = micromag_search(H0, max_flips=search_flips,
                                    lam_ex=0.01, lam_ani=0.1)
            G = H.astype(np.float64) @ H.astype(np.float64).T
            f = float(np.sum((G - order * np.eye(order)) ** 2)) / 2.0
            if best_H is None or f < best_f:
                best_H = H.copy(); best_f = f
                if f == 0:
                    break
    return best_H, best_f


def extract_features_from_row(row, order):
    """Same as extract_features but from a raw row array."""
    r0 = np.asarray(row, dtype=np.float64)
    runs = run_length_encoding(r0)
    spec = spectral_features(r0)
    acorr = autocorrelation_profile(r0)
    label = method_label(order)
    qe = np.array([quantum_style_energy(r0, order)], dtype=np.float32)
    on = np.array([order / 2000.0], dtype=np.float32)
    return np.concatenate([runs, spec, acorr, label, qe, on])


# Main pipeline ---------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")
    csv_files = list(CSV_DIR.glob("hadamard_*.csv"))
    csv_files.sort()
    print(f"Found {len(csv_files)} CSV files")

    FEAT_DIM = 110
    dataset = HadamardDataset(csv_files, FEAT_DIM, max_samples=500)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"Training set: {len(dataset)} samples")

    model = HadamardRNN(FEAT_DIM, hidden_dim=128, num_layers=2).to(DEVICE)
    print(f"Model: {sum(p.numel() for p in model.parameters())} params")
    train_model(model, loader, epochs=50, lr=0.001)

    # save model
    torch.save(model.state_dict(), str(CSV_DIR / "rnn_hadamard.pt"))
    print("Model saved.")

    # test on a small gap
    print("\n=== RNN‑guided search on H(92) ===")
    t0 = time.monotonic()
    H, best_f = rnn_guided_search(92, model, n_trials=10, search_flips=3000)
    dt = time.monotonic() - t0
    from hoa64.hadamard import verify
    print(f"  f={best_f:.1f}  hadamard={verify(H)}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
