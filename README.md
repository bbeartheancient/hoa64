# hoa64 — Hadamard Matrix Construction & Search Toolchain

A comprehensive toolkit for constructing, verifying, searching, and evolving
Hadamard matrices.  Combines classical constructions (Sylvester, Paley I/II
with prime‑power field extensions, Williamson, Miyamoto, propus, Cooper‑Wallis,
Goethals‑Seidel, Kronecker products), heuristic search engines (micromagnetic
descent, FFT‑based PSD minimization, signature‑guided RNN search), and
direct integration with SageMath for orders above 2000.

**810 orders verified up to order 7408.  Gaps below 2000 are 1212 and 1940
(existence proven, construction data still inaccessible).  Order 1852
ships as a Djoković 1992 solution (`matrices/hadamard_1852.csv.gz`).**

Hadamard Matrix Library/Solver

<img width="784" height="349" alt="image" src="https://github.com/user-attachments/assets/009bc226-fdf1-40d8-aacf-96ae66a1945d" />

Micromagnetic Matrix Annealing

<img width="783" height="392" alt="image" src="https://github.com/user-attachments/assets/b5608afc-a6d3-45f7-9707-87752524dca6" />

High-Order Ambisonic Field Calculator

<img width="783" height="293" alt="image" src="https://github.com/user-attachments/assets/4e43c0c5-85c4-4611-b8fa-b069b1b628fb" />

fBm Terrain Generator

<img width="796" height="296" alt="image" src="https://github.com/user-attachments/assets/74bf097a-4fc6-44eb-9532-cd0e5655a2e5" />

Atomic Orbital Simulator

<img width="796" height="296" alt="image" src="https://github.com/user-attachments/assets/6a35601c-65df-4742-baf5-1b08fa69d3a9" />

Virtual Site Survey

<img width="607" height="300" alt="image" src="https://github.com/user-attachments/assets/cfc803fd-05a5-4756-9199-54bb1413775a" />

RF Antenna + Filter Designer

<img width="614" height="372" alt="image" src="https://github.com/user-attachments/assets/1dd0c671-fc7b-4835-81dd-a8002c92db70" />

Metamaterials Designer

<img width="796" height="296" alt="image" src="https://github.com/user-attachments/assets/b360cd90-d604-4dfa-81d1-0d92102f9792" />

Intelligent Noise Analyzer

<img width="796" height="271" alt="image" src="https://github.com/user-attachments/assets/07622c5a-8bef-4c4c-86a7-fd1bb8267f12" />


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
| `em_physics.py` | Antenna physics core: lossy‑dielectric propagation, Friis, textbook builders, Stokes polarization, phased‑array factor/tapers *(alpha)* |
| `antenna_design.py` | Optimal‑construction recommender (band + site conditions) *(alpha)* |
| `parts_db.py` | Curated off‑the‑shelf antenna parts DB + matcher *(alpha — 103 catalog rows)* |
| `fdtd.py` | 3‑D Yee FDTD Maxwell solver (air/water, polarization) *(alpha)* |
| `antenna_evo.py` | Thin‑wire MoM evaluator + Hadamard‑seeded topology SA *(alpha)* |
| `kicad_gen.py` | Procedural KiCad 7 footprints/boards for PCB antennas, RF filters, and 2-layer materials *(alpha)* |
| `rf_filter.py` | PCB RF-filter synthesis (Butterworth/Chebyshev LPF/HPF/BPF/BSF) — distributed stepped/stub/hairpin plus lumped `lc`/`dc_lc`/`c_shunt`/`qw_tl`/`rc`/`crc`/`rl` + ABCD S-params + Hadamard SA *(alpha)* |
| `materials.py` | H.8 flux-tile homes: conductive cloth, mutual-cap touchpad, spin-ice/metamaterial cell *(alpha)* |
| `mcu.py` | Microcontroller lab: WS2812 firmware, ESP-NOW mesh, edge-engine export (CircuitPython / Rust `no_std` / bare-metal C) *(alpha)* |
| `site_survey.py` | SRTM virtual site survey: tight DEM window, path profile, 4/3‑earth Fresnel/Deygout, radio horizon, Esri satellite mosaic *(alpha)* |
| `noise_data.py` | NOISEX‑92 access + 4 synthesized RF baseband classes (BLE/WiFi/Zigbee/LoRa) + log‑mel DSP *(alpha)* |
| `dit_noise.py` | DiT‑backbone noise classifier (adaLN‑Zero, Muon/AdamW, lazy torch) *(alpha)* |
| `rf_capture.py` | Live local‑radio capture for the analyzer: wifi (/proc/net/dev) + BLE (HCI ioctl) activity counters → measured‑cadence baseband envelope *(alpha)* |
| `muon.py` | Muon/Dion3 optimizer: cursed-quintic Newton–Schulz orthogonalization of momentum (Gram-NS + row subsample) *(alpha)* |
| `gerzon.py` | Gerzon 1975 AB module (A-format → WXYZ); H₄ after L_F↔L_B swap; H₂/wall cell SA |
| `holographic.py` | Holographic entropy S = A/(4ℓₚ²) of volume V; search uses (S/S_*−1)² |
| `crown.py` | Spherical-crown diffraction (Liu 2022 OPSF + 2-D FFT) cell SA |
| `brillouin.py` | Brillouin-zone folding (Guan 2026): X→Γ period-doubling, fold coherence, weave CD |
| `sudoku.py` | Sudoku solvers remapped to Hadamard rows (backtrack, overlay, CSP, DLX, residuals) |
| `actual_size.py` | Press 1980 actual size from e, ħ, G, m_e, m_p — L, R⊕, T, pitch = L/n |
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
- **CSV import** — 12 Alpoge matrices (668‑1964, formerly open) plus Djoković H.1852

## Search Engines

- **Max‑det descent** — single‑flip Gram minimization
- **Williamson FFT** — PSD minimization over 4 circulant sequences
- **General‑circulant GS** — 4 sequences, no symmetry constraint
- **Micromagnetic** — exchange + demagnetization + anisotropy energy
- **Tile SA** — 2×2 H₂-cell simulated annealing
- **Gerzon AB** — 1975 A-format → WXYZ; H₂ prior on |Z|, 45°/225° cancel move
- **Holographic** — S = A/(4ℓₚ²) is the entropy of volume V (A = area of ∂V); the SA residual is the scale-free (S/S_* − 1)²
- **Crown** — spherical-crown diffraction (occlusion-utilizing PSF, 2-D FFT)
- **Brillouin** — Guan 2026 zone folding: period doubling folds X onto Γ; fold coherence + weave CD; Materials lab reports the same on the Walsh lattice
- **Sudoku** — Wikipedia Sudoku solvers with the goal changed from digits/boxes to pairwise-orthogonal ±1 rows (backtrack, pattern overlay, CSP, exact cover / dancing links, residuals, stochastic)
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

- **Matrix Lab** — construct/verify matrices (auto/sylvester/paley/miyamoto/
  cw/gcp/row_builder), Search panel with the full engine list, pixel‑art
  preview, ℍ³ transmute (row‑simplex PCA → Poincaré ball with geodesics)
- **Search Studio** — launch max‑det/micromag/tile/Gerzon/holographic/crown/
  brillouin/sudoku search jobs, live progress over WebSocket, mid‑run retune,
  export to library; one Run panel (matrix + E/BEST/T waveforms,
  Micromag‑style series toggles)
- **Micromag Sim** — annealing lab; algorithm select runs any search
  engine (micromag/tile/gerzon/holographic/crown/brillouin/sudoku/maxdet/
  williamson/gs/circulant) with live site‑energy/gradient/flux
  heatmaps, waveforms, and library‑goal evolution (E_goal/goal_agree).
  Flux of Sylvester (and any A⊗H₈) is a 4‑tile H.8 tessellation — the
  (0,0) block is exactly `flux_map(H₈)`, the other three differ only
  on Kronecker seams. Those four atoms persist at every dyadic scale
  (4 unique 16×16 / 32×32 / … tiles); the large‑order “variation” is
  Walsh *placement* (H.256 tile counts 341/171/171/341, nested
  top‑left = H.128). Paley/generic library matrices do not tile.
  Search prior: `lam_tile` rewards H.8 (or H.4) tessellation.
  Gerzon AB: `lam_z` rewards stride-2 H₂ cells in the 1975 WXYZ
  basis (`E_z = lam_z · mean((|Z_int|−2)²)`); a fraction of moves
  flip the 45°/225° (L_F, R_B) cancel pair. The [GERZON] layer is
  the overlapping |Z| wall field; [READ GERZON] inspects a start
  matrix without annealing.
- **Materials** — three homes for the same catalog. CLOTH is a two‑layer
  yarn (+ face / − reverse, walls are cuts). TOUCH is a mutual‑cap
  pad (each wall bond is a capacitor). META is a spin‑ice unit cell
  (the H.8 atom) plus the Walsh tile lattice. All three export KiCad
  as 2‑layer F.Cu/B.Cu pads (no B.Cu GND pour — that would short the
  reverse layer). After export, `[FOOTPRINT]` / `[BOARD]` toggles the
  preview between the layout and the generated `.kicad_pcb`.
  Brillouin-zone folding reports X→Γ coherence and weave CD on the
  Walsh lattice. `[ACTUAL SIZE]` sets pitch from Press 1980
  (L ≈ ε^{1/4} (2a₀)(e²/4πϵ₀Gm_p²)^{1/4}) so n cells span one creature.
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
- **Library** — `[MAP]` construction‑DAG classification and
  achieved‑vs‑bound chart; `[CHALLENGES]` is DARPA's 23 mathematical
  challenges (2007) with this lab's honest tooling alignment
  (`active` / `partial` / `latent` / `none` — statuses are *not*
  progress toward solutions). Engine chips deep‑link into the tab
  that implements them.
- **Antenna** *(ALPHA — needs extensive field testing)* — physics‑based
  antenna lab: DESIGN ranks optimal
  constructions by band + site conditions (exact antenna theory — dipole,
  patch, helix, yagi… with equation traces), PARTS matches a curated
  off‑the‑shelf parts DB (103 rows, everythingRF deep links; a wide query
  such as 2400–5800 lists every overlapping row when no part covers the
  full span — dual‑band SKUs are one row per lobe), FIELDS runs a
  3‑D FDTD Maxwell solver (air/water/interface, live |E| slice viewer +
  E_RMS strip chart), EVOLVE anneals Hadamard‑seeded wire topologies
  scored by a real thin‑wire Method‑of‑Moments solver, SMITH sweeps
  Z_in(f) on an interactive Γ‑plane chart, ARRAY is the phased‑array
  designer (tapers, steering, grating‑lobe metrics, polar plot of
  total_db over the array factor), and PCB types export
  procedural KiCad files (MIFA / scaled `RF_Antenna.pretty` library /
  evolved walk, with a FOOTPRINT/BOARD canvas preview in fixed KiCad
  layer colours — red F.Cu, blue B.Cu; silk/edge/keepout are outlines
  only so they do not wash out the copper — and the JLCPCB
  λ/4·50 Ω·RL≥10 dB checklist). Site survey lives on the Terrain tab.
- **Filter** *(ALPHA)* — PCB RF filters from the everythingRF Filter
  Digest. Same Butterworth/Chebyshev *g*-values drive every kind.
  Distributed topos: stepped‑Z LPF, gap+stub HPF, hairpin BPF,
  open‑stub BSF. Lumped topos from the same prototype: `lc` ladders,
  `dc_lc` / `c_shunt` coupled-resonator BPF, `qw_tl` Richards/Kuroda
  lines, and passive `rc`/`crc`/`rl` (approximate in 50 Ω). DESIGN
  synthesises geometry or a BOM and plots S21/S11 from a lossy ABCD
  cascade (RL ≥ 10 dB, 40 dBc @ 10 % from the edge). EVOLVE
  Hadamard‑perturbs distributed section lengths/widths. KICAD exports
  `.kicad_mod` / `.kicad_pcb`, plus a KiCad ≥8 `.kicad_block` zip
  (schematic + board) for lumped topos.
- **Noise** *(ALPHA — window‑level split over 15 long recordings + 4
  synthesized RF envelopes; needs a larger database before real‑world
  generalization claims)* — DiT‑backbone noise classifier: train on
  NOISEX‑92 plus synthesized BLE/WiFi/Zigbee/LoRa *baseband envelopes*
  (cadence/duty/spectral shape at 19.98 kHz — not the RF carrier) with
  a Muon (Dion3) or AdamW optimizer and a live loss/accuracy chart,
  then classify WAV files or live captures with a log‑mel
  spectrogram + class‑probability bars. Live capture sources: the mic,
  or the machine's own radios — wifi (via /proc/net/dev counters) and
  BLE (via the HCIGETDEVINFO ioctl) activity polled at ~100 Hz and
  rendered as a measured‑cadence baseband envelope (managed‑mode wifi
  sees this station's traffic only; Zigbee/LoRa report unavailable
  without a local radio, idle captures are never fabricated)
- **Microcontroller** *(ALPHA — firmware is template-generated; hardware
  testing is on the user)* — `[LED]` paints WS2812 frames (serpentine
  GRB) and downloads ESP32 / Teensy / CircuitPython sketches, or
  pushes a frame to a device IP. `[MESH]` (RSSI tomography, no CSI)
  downloads ESP-NOW node sketches, collects the gateway RSSI matrix,
  and exports `mesh_field.json`. `[EDGE]` downloads
  `hadamard_core` / `flux_map` / `terrain_fbm` kernels as CircuitPython,
  Rust `#![no_std]`, or bare-metal C.

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

The `matrices/` directory contains 13 gzipped Hadamard matrices: 12 Alpoge
orders (668‑1964, formerly open, ingested Aug 12 2026) plus Djoković's
H.1852. Run `rebuild.py` to decompress them and regenerate all
constructible orders.

The `open_hadamard/` directory (not in repo) stores the full 810‑matrix
CSV library built by `evolve.py` (no hard order ceiling).  Each file is
comma‑separated ±1 values.

## Requirements

- Python 3.10+ with NumPy
- PyTorch (for RNN models)
- SageMath 10.0+ (for orders above 2000)
- fastapi + uvicorn + httpx (for the webapp)
- CUDA‑capable GPU (optional, for model training)

## Key Results

- 810 Hadamard matrices built & verified (highest 7408)
- 2 gaps below 2000: 1212, 1940 (known existence, inaccessible data)
- H.1852 verified (Djoković 1992; ships in `matrices/`)
- H.668 verified as genuine Hadamard (Aug 12 2026)
- 12 formerly‑open Alpoge orders (668‑1964) ingested as gzipped CSVs
