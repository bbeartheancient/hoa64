# hoa64 — Hadamard Matrix Construction & Search Toolchain

A comprehensive toolkit for constructing, verifying, searching, and evolving
Hadamard matrices.  Combines classical constructions (Sylvester, Paley I/II
with prime‑power field extensions, Williamson, Miyamoto, propus, Cooper‑Wallis,
Goethals‑Seidel, Kronecker products), heuristic search engines (micromagnetic
descent, FFT‑based PSD minimization, signature‑guided RNN search), and
direct integration with SageMath for orders above 2000.

**807 orders verified up to order 3984.  All known Hadamard matrices below
2000 are constructible except 3 (1212, 1852, 1940) which have proven
existence but inaccessible construction data.**

## Quick Start

```bash
# Build a specific order
python3 -m hoa64.cli hadamard --generate 668

# Verify a saved matrix
python3 -m hoa64.cli hadamard --verify open_hadamard/hadamard_668.csv

# Run the gap‑filling daemon (continuous)
python3 hoa64/evolve.py

# Game‑of‑Hadamard DAG visualization
python3 -m hoa64.game_of_hadamard -n 128

# Rebuild the CSV library from scratch (requires SageMath)
python3 rebuild.py
```

## Core Modules

| Module | Purpose |
|---|---|
| `hadamard.py` | Sylvester, Paley I/II + prime‑power, Kronecker, CSV, Miyamoto |
| `miyamoto.py` | H(q−1)→H(4q) construction (38 orders, 8 above 2000) |
| `williamson.py` | Williamson & general‑circulant GS (FFT PSD search) |
| `cw_construction.py` | Cooper‑Wallis Turyn→T‑sequence pipeline |
| `finite_field.py` | 11 prime‑power extensions (F₃²…F₂₃², F₃⁶) |
| `micromag.py` | Micromagnetic energy descent (exchange + demag + anisotropy) |
| `evolve.py` | Gap‑filling daemon — construction + RNN‑guided search |
| `rnn_hadamard.py` | 288k‑param LSTM fitness model (500 samples) |
| `sig_predictor.py` | Block‑signature predictor from order properties |
| `circulant_search.py` | Circulant Hadamard evolver (n=m², m even) |
| `game_of_hadamard.py` | Conway‑style construction DAG visualization |
| `gcp_hadamard.py` | GCP‑based Hadamard (length‑10 GCP verified) |
| `row_builder.py` | α,β,γ,δ counting analyser |
| `rh.py` | RH |Δₙ| bound checker |

## Construction Methods Implemented

- **Sylvester** — order 2^k
- **Paley I** — q+1 for prime/prime‑power q≡3 mod 4
- **Paley II** — 2(q+1) for prime/prime‑power q≡1 mod 4
- **Prime‑power Paley** — F_q for q = p^e (11 field extensions)
- **Kronecker** — product of any two known orders
- **Williamson** — symmetric circulant (DB from SageMath, up to q=63)
- **Miyamoto** — H(q−1)→H(4q) for q≡1 mod 4 prime power
- **Propus** — Balonin‑Djoković symmetric array
- **Cooper‑Wallis** — Turyn→T‑sequence→CW (16 orders)
- **Goethals‑Seidel** — SDS difference families (via SageCell/SageMath)
- **CSV import** — 12 Alpoge matrices (668‑1964, formerly open)

## Search Engines

- **Max‑det descent** — single‑flip Gram minimization
- **Williamson FFT** — PSD minimization over 4 circulant sequences
- **General‑circulant GS** — 4 sequences, no symmetry constraint
- **Micromagnetic** — exchange + demagnetization + anisotropy energy
- **RNN‑guided** — LSTM scores candidate seeds before micromag descent
- **Signature‑guided** — predicted block signature seeds from trained model

## Models

- `rnn_hadamard.pt` — LSTM fitness predictor (500 samples, CUDA)
- `sig_simple.pt` — block‑signature run‑length predictor (500 samples)

## Data

The `matrices/` directory contains 12 gzipped Alpoge Hadamard matrices
(the formerly‑open orders 668‑1964, discovered Aug 12 2026).
Run `rebuild.py` to decompress them and regenerate all constructible orders.

The `open_hadamard/` directory (not in repo) stores the full 807‑matrix
CSV library built by `evolve.py`.  Each file is comma‑separated ±1 values.

## Requirements

- Python 3.10+ with NumPy
- PyTorch (for RNN models)
- SageMath 10.0+ (for orders above 2000)
- CUDA‑capable GPU (optional, for model training)

## Key Results

- 807 Hadamard matrices built & verified
- 3 gaps below 2000: 1212, 1852, 1940 (known existence, inaccessible data)
- 195 gaps above 2000 (constructible via SageMath)
- H.668 verified as genuine Hadamard (Aug 12 2026)
- 12 formerly‑open orders (668‑1964) ingested as gzipped CSVs
