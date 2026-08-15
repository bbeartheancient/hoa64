# hoa64 — Hadamard Matrix Construction & Search Toolchain

A comprehensive toolkit for constructing, verifying, searching, and evolving
Hadamard matrices.  Combines classical constructions (Sylvester, Paley I/II
with prime‑power field extensions, Williamson, Miyamoto, propus, Cooper‑Wallis,
Goethals‑Seidel, Kronecker products), heuristic search engines (micromagnetic
descent, FFT‑based PSD minimization, signature‑guided RNN search), and
direct integration with SageMath for orders above 2000.

**807 orders verified up to order 3984.  All known Hadamard matrices below
2000 are constructible except 3 (1212, 1852, 1940) which have been proven
to exist but have inaccessible construction data.**

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

# Web GUI ("hadamard lab") on 127.0.0.1:8770
python3 -m hoa64.cli webapp
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
| `terrain.py` | Hadamard‑layered Perlin fBm terrain generator |
| `orbitals.py` | Hydrogenic real‑orbital \|ψ\|² sampler (SN3D convention) |
| `hadamard_space.py` | ℍ³ transmute: row‑simplex PCA → Poincaré ball |
| `em_physics.py` | Antenna physics core: lossy‑dielectric propagation, Friis, textbook antenna builders, Stokes polarization *(alpha)* |
| `antenna_design.py` | Optimal‑construction recommender (band + site conditions) *(alpha)* |
| `parts_db.py` | Curated off‑the‑shelf antenna parts DB + matcher *(alpha — 57 catalog rows)* |
| `fdtd.py` | 3‑D Yee FDTD Maxwell solver (air/water, polarization) *(alpha)* |
| `antenna_evo.py` | Thin‑wire MoM evaluator + Hadamard‑seeded topology SA *(alpha)* |
| `kicad_gen.py` | Procedural KiCad 7 footprints/boards for PCB antennas *(alpha)* |
| `site_survey.py` | SRTM virtual site survey: tight DEM window, path profile, 4/3‑earth Fresnel/Deygout, radio horizon, Esri satellite mosaic *(alpha)* |
| `noise_data.py` | NOISEX‑92 noise database access + log‑mel DSP *(alpha)* |
| `dit_noise.py` | DiT‑backbone noise classifier (adaLN‑Zero, lazy torch) *(alpha)* |
| `webapp/` | FastAPI + vanilla‑JS web GUI (see Webapp below) |
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

## Webapp

A FastAPI + vanilla‑JS (no build step) web GUI — the "hadamard lab".
Launch it with:

```bash
python3 -m hoa64.cli webapp            # serves 127.0.0.1:8770
python3 -m hoa64.cli webapp --host 0.0.0.0 --port 9000
```

Dependencies (into the working env, e.g. sage‑dev):

```bash
pip install fastapi 'uvicorn[standard]' httpx
```

Smoke‑check the whole thing in‑process (no server needed):

```bash
python3 -m hoa64.webapp.selftest
```

Tabs:

- **Matrix Lab** — construct/verify matrices, pixel‑art preview, ℍ³
  transmute (row‑simplex PCA → Poincaré ball with geodesics)
- **Search Studio** — launch max‑det/micromag/tile search jobs, live
  progress over WebSocket, mid‑run retune, export to library; one Run
  panel (matrix + E/BEST/T waveforms, Micromag‑style series toggles)
- **Micromag Sim** — annealing lab with live site‑energy/gradient/flux
  heatmaps, waveforms, and library‑goal evolution (E_goal/goal_agree)
- **HOA Studio** — speaker‑array designer, scene encode/rotate/analyze
- **Terrain** — `[GENERATE]` / `[SURVEY]` sidebar switch.
  GENERATE is the Hadamard‑layered fBm heightfield with per‑octave
  mute/solo. SURVEY is the virtual site survey (moved here from
  Antenna): SRTM Terrarium DEM in a tight path window, labeled TX/RX
  elevation profile (ground / LOS / Fresnel 0.6 r₁ / blockage), and a
  metric 3‑D link view (satellite‑textured mesh, masts at ground+h,
  LOS chord, Fresnel rings). Blank or coincident RX probes eight
  400 m bearings and keeps the clearest hop instead of inventing a
  blocked due‑east path. Radio horizon and suggested mast heights
  land in the link‑budget table.
- **Orbitals** — hydrogenic |ψ|² point cloud (3D / XZ splat / both)
- **Library** — construction‑DAG classification and achieved‑vs‑bound chart
- **Antenna** *(ALPHA — needs extensive field testing)* — physics‑based
  antenna lab: DESIGN ranks optimal
  constructions by band + site conditions (exact antenna theory — dipole,
  patch, helix, yagi… with equation traces), PARTS matches a curated
  off‑the‑shelf parts DB (57 rows, everythingRF deep links), FIELDS runs a
  3‑D FDTD Maxwell solver (air/water/interface, live |E| slice viewer +
  E_RMS strip chart), EVOLVE anneals Hadamard‑seeded wire topologies
  scored by a real thin‑wire Method‑of‑Moments solver, SMITH sweeps
  Z_in(f) on an interactive Γ‑plane chart, and PCB types export
  procedural KiCad files. Site survey lives on the Terrain tab.
- **Noise** *(ALPHA — window‑level split over 15 long recordings; needs a
  larger database before real‑world generalization claims)* — DiT‑backbone
  noise classifier: train on the NOISEX‑92
  database (live loss/accuracy chart), then classify WAV files or live
  mic captures with a log‑mel spectrogram + class‑probability bars

Seven retro‑monitor themes (MONO, P1, AMB, PLS, DMG, CGB, VGA — VGA with
four subthemes). `[SET]` opens a centered popup overlay (outside `#app`
so display filters do not trap it) with per‑display controls
(brightness, contrast, saturation, bivert, DMG ghosting, CGB palette
packs). Original GLSL layer: CRT/DMG post passes, electric/quantum/flux
shaders. HTML/JS/CSS are served `Cache-Control: no-cache` so a refresh
picks up lab edits.

**Security:** the server is unauthenticated and binds to localhost by
default; several endpoints take filesystem paths.  Treat it as
trusted‑local only — do not expose it on a public interface without
adding auth.

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
- fastapi + uvicorn + httpx (for the webapp)
- CUDA‑capable GPU (optional, for model training)

## Key Results

- 807 Hadamard matrices built & verified
- 3 gaps below 2000: 1212, 1852, 1940 (known existence, inaccessible data)
- 195 gaps above 2000 (constructible via SageMath)
- H.668 verified as genuine Hadamard (Aug 12 2026)
- 12 formerly‑open orders (668‑1964) ingested as gzipped CSVs
