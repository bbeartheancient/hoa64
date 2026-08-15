# AGENTS.md — hoa64

Guidance for AI coding agents working in this repository. Read this before
making changes.

## Project overview

`hoa64` is a single flat Python package (this directory **is** the package —
it uses relative imports, so it must be imported/run as `hoa64` from the
parent directory `/home/bbear`). It serves two purposes that coexist in the
same modules:

1. **HOA-7 spatial calculator** (the original project): 7th-order Ambisonics
   (Ambix ACN / SN3D) encode/decode/rotate/analysis of audio scenes, emitting
   JSON spatial reports intended for consumption by other AI models
   ("Qwythos"). Fixed spherical-harmonic geometry, 64 = (7+1)² channels.
2. **Hadamard matrix construction & search toolchain** (the current focus,
   see `README.md`): classical constructions (Sylvester, Paley I/II with
   prime-power field extensions, Williamson, Miyamoto, propus,
   Cooper-Wallis, Goethals-Seidel, Kronecker products) plus heuristic search
   engines (max-det descent, micromagnetic SA, tile-based SA, Williamson/GS
   FFT PSD minimization, RNN-guided search). 807 orders verified up to 3984;
   gaps below 2000: 1212, 1852, 1940 (existence proven, data inaccessible —
   note `matrices/hadamard_1852.csv.gz` now ships a Djokovic 1992 solution).

The `hoa64` name itself is a pun: Sylvester H-64 matches the 7th-order HOA
channel count, and FOA is a normalized H4.

## Layout

- `__init__.py` — package facade; re-exports the public HOA API and the
  `hadamard` module API. `__version__ = "0.5.0"`.
- `__main__.py` — delegates to `cli.main()`.
- `cli.py` — argparse CLI with subcommands: `analyze`, `demo-scene`,
  `vision`, `serve`, `detect`, `live`, `hadamard`, `condition`.
- HOA pipeline modules: `basis.py`, `encode.py`, `decode.py`, `rotate.py`,
  `wigner.py` (fast Wigner-D rotation), `analysis.py`, `stft.py`,
  `audio_io.py`, `synth.py`, `stream.py`, `report.py` (JSON schema),
  `vision.py`, `detector.py` (optional torchvision), `conditioning.py`
  (ComfyUI submission), `live_audio.py` (PulseAudio capture), `server.py`
  (stdlib-only HTTP API, `http.server`, default 127.0.0.1:8765).
- `webapp/` — FastAPI + vanilla-JS web GUI ("hadamard lab",
  `hoa64 webapp`, default 127.0.0.1:8770): `app.py` (`create_app`; an
  /api/{path} catch-all registered after the routers returns JSON 404 for
  unknown API paths — without it unmatched POSTs fall through to the
  StaticFiles mount and surface as 405),
  `routes_hadamard.py` (/api construction/verify/library endpoints +
  /api/viz/hadamard-space ℍ³ transmute),
  `routes_search.py` (Phase 2: /api/search job endpoints + `/ws/job/{id}`
  progress WebSocket with mid-run micromag retune via `job.params["live"]`),
  `routes_sim.py` (Phase 3: /api/sim/micromag annealing lab with live
  `site_energy`/`energy_gradient`/`flux_map`/`gerzon.Z_wall` heatmap
  frames; optional `goal_order` (= `order`, must be in the library)
  anneals toward that library matrix via `micromag_sa`'s goal
  attraction — frames carry `E_goal`/`goal_agree`, and `lam_goal` is
  mid-run retunable; `lam_z` is the Gerzon H₂ prior, live-retunable;
  a step-0 frame with the resolved start matrix's preview PNG + energy
  decomposition is reported before the anneal begins (sylvester/library
  starts only — a random start is generated inside `micromag_sa`);
  `GET /api/sim/gerzon` inspects a start matrix's Z-wall with no anneal),
  `routes_hoa.py` (Phase 4: /api/hoa speaker-array designer, scene
  encode/rotate/analyze with one-shot WAV tokens),
  `routes_gen.py` (Phase 5a: /api/gen terrain/orbital/noise-field
  generative endpoints; terrain also returns `layers_f32` — the signed
  amp-weighted per-octave fBm contribs on the heightmap grid, for
  client-side octave mute/solo; octaves capped ≤ 8; the orbital endpoint
  still emits `proj_png_b64` for compat but the tab no longer uses it —
  the XZ projection is a client-side density splat of the point cloud;
  the orbitals tab has a [3D][XZ][BOTH] selector — BOTH overlays the
  transparent-cleared cloud (renderer alpha, post pipeline bypassed) on
  the dimmed splat — and the QUANTUM_FRAG layer is DRIVEN BY the cloud: a
  64-bin radial |ψ|² profile uploads to the shader's `uDensity` texture
  after each Simulate (shell brightness ∝ density(r) + radial phase warp);
  the shader canvas lives inside `.orb-viewport` (position:relative +
  overflow:hidden) and feeds the 3D scene as `scene.background` via
  CanvasTexture — clipped, behind the cloud, post-processed with it;
  the terrain tab's 3D view (RIGHT) and layer view (LEFT) are equal-width
  1:1 panels via `.ter-views` (both canvases 384², renderer
  `setSize(w,h,false)` so CSS owns display size), and the layer/octave
  controls live in the sidebar),
  `routes_palettes.py` (Phase 7: GET /api/palettes — walks
  `Assets/gb/common/Palettes/**.pal` [binary: 4 RGB24 colors repeated per
  slot + `\x81APGB` trailer; parser reads the first 12 bytes], returns
  `{category, name, colors}` per file, 5-min cache), `routes_library.py`
  (Phase 5b: /api/dag
  construction-DAG classification + /api/detbounds achieved-vs-bound,
  both cached/threaded), `routes_challenges.py` (GET /api/challenges —
  DARPA's 23 mathematical challenges + this lab's honest tooling
  alignment; consumed by Library's CHALLENGES layer), `routes_antenna.py` (antenna lab under
  /api/antenna: sync /design recommender + /parts matcher (full-cover
  first, then overlap fallback so a 2400–5800 query still lists every
  in-range row; dual-band SKUs are one row per lobe) + /kicad
  export [patch/meander_ifa/mifa/loop/lib/evolved; GET /kicad/library
  lists installed RF_Antenna.pretty; response carries preview prims +
  JLCPCB design_params; bounded one-shot-per-file token cache] +
  /smith MoM Z_in(f)→Γ(f) sweep [dipole or explicit wire geometry] +
  /array sync phased-array designer [design_array; tapers/steering/
  grating-lobe metrics; non-finite SLL normalized to null] +
  /survey + /survey/map (SRTM DEM + Esri imagery; used by the Terrain
  tab, not an Antenna panel),
  job-based /fields FDTD lab and /evolve Hadamard-seeded topology SA —
  same callback→report/`_BudgetStop`/`params["live"]` wiring as
  routes_sim; /evolve streams the far-field `pattern_png_b64` mid-run on
  best_E improvements (≥2 s apart; the post-loop final frame always
  carries it) via the MoM `pattern` callable riding in the SA callback's
  `geom`), `routes_filter.py` (PCB RF-filter lab under /api/filter:
  sync /design Butterworth/Chebyshev LPF/HPF/BPF/BSF with a `topo`
  selector — distributed stepped-Z/gap+stub/hairpin/open-stub plus
  lumped `lc` (LPF/HPF/BPF/BSF), `dc_lc` and `c_shunt` coupled-resonator
  BPF, `qw_tl` Richards/Kuroda commensurate lines, and passive
  `rc`/`crc`/`rl` — + S21/S11 sweep + BOM (`components[]`) +
  /kicad footprint/board, and KiCad ≥8 design blocks
  (`.kicad_block` folder = block.kicad_sch + block.json + block.kicad_pcb
  + zip) for lumped topos; lumped topos allow f down to 1 Hz;
  job-based /evolve Hadamard SA on section
  lengths/widths, distributed topos only), `routes_materials.py`
  (cloth/touchpad/
  metamaterial from H.8 flux tiles + /kicad; materials boards skip the
  B.Cu GND pour so reverse-layer cells stay discrete, and the response
  carries `preview_board` for the FOOTPRINT/BOARD toggle),
  `routes_noise.py` (/api/noise: /classes lists 19 labels — 15 NOISEX-92
  + 4 synthesized RF baseband envelopes ble/wifi/zigbee/lora — plus
  `live_sources` availability from rf_capture;
  /train DiT job with muon|adamw, /analyze WAV-path or live capture —
  mic via live_audio, local wifi/ble radio telemetry via rf_capture
  `live_source`, capture stats echoed — classify → mel PNG + probs), `routes_mcu.py` (/api/mcu Microcontroller lab: /firmware LED
  matrix (ESP32/Teensy Arduino + CircuitPython) and ESP-NOW mesh node
  sketches via one-shot token downloads, /push GRB frame → device HTTP
  POST (bare-host validated, trusted-local), /mesh/collect gateway RSSI
  matrix fetch, /export edge engine bundles),
  `static/vendor/`
  (pinned three.js 0.170.0 + OrbitControls, import map in index.html),
  `static/js/kicad_layers.js` (fixed copper palette for all KiCad
  previews — red F.Cu, blue B.Cu, green In1.Cu, orange In2.Cu;
  theme LUT is bypassed; W=½ edges unfilled; paint order is B.Cu →
  inners → F.Cu → silk/edge/keepout; silk courtyards, Edge.Cuts, and
  keepout zones are stroked only — filling them painted a gray/yellow
  wash over the Antenna copper; B.Cu GND pours fill blue),
  `static/css/themes.css` + `static/js/theme.js` (Phase 4.5/6/7: seven
  runtime retro-monitor themes — mono/P1/AMB/PLS/DMG/CGB/VGA —
  `data-theme` + localStorage, `recolorCanvas` LUT for server PNGs (theme
  color + quantization ONLY); the LUT ramp is always FORWARD (lum 0 →
  darkest stop) on every theme, inverted ones included — server PNGs are
  fixed ink-on-black renders, and the old inverted walk made mostly-dark
  PNGs a solid light rectangle; LUT intensity is the MAX RGB channel, not
  Rec.601 luma (luma capped pure green at ~58% and never reached the
  lightest shade); exported
  `dmgLut(bivert)` pins the exact 4-shade DMG mapping for tests;
  `fillPlusOne(canvas)` paints a viewport as an all-+1 field (fill with
  the server's #22c55e +1 ink, then the normal retint path — exactly a
  retinted all-+1 PNG); every streamed-job matrix/field viewport (Matrix
  Lab, Search Studio, Micromag Sim, Antenna FIELDS/EVOLVE) calls it at
  job start / tab reset so it never sits blank or shows the previous run
  while the first compute frame is made;
  mono/green/amber ramps are pure bg→fg
  single-hue luminance lerps; DMG is the exact user palette
  #1b2a09/#0e450b/#496b22/#9a9e3f; CGB interpolates those anchors into a
  56-stop gradient (quantize "palette56" — the 5-bit snap is gone) and
  accepts palette packs via `setCgbPalette` (palette sorted by luminance,
  interpolated to 56; darkest→fg, lightest→bg, chrome overrides ride
  inline `--fg/--bg/--dim/--faint` on <html>, installed BEFORE the
  themechange dispatch so retint listeners already see the new ramp);
  DMG/CGB bivert is a FULL in-palette theme reversal of the entire screen
  (`_biverted`/`_chromeColors`/`_applyChrome`): fg↔bg and dim↔accent swap
  as inline vars on <html>, `themeColor`/`themeRampSample` follow, the
  canvas LUT ramp reverses (`bivertInLut`) — no invert(1) filter for
  dmg/cgb, colors stay exactly in-palette — while plasma bivert keeps the
  global #app invert filter;
  `themeRampSample` walks the ramp backwards on inverted themes (bivert
  un-reverses the walk, chrome is swapped too) and lifts inverted-family
  sampling to the upper 60% of the walked ramp so 3D views keep ≥40%
  contrast against the background;
  brightness scales the CRT phosphor glow via `--glow-eff` text-shadow
  alpha (mono/green/amber/plasma/vga; LCDs glow:null);
  VGA has four `data-subtheme` variants, each with its own bg/fg in
  `VGA_SUBTHEMES` — `themeColor("bg"/"fg")` and the 2D LUT follow the
  subtheme, not just the CSS chrome — CGB a
  `data-variant="dark"` flip; subtheme/variant/bivert switches re-fire
  `themechange` so canvas/three consumers re-render; persisted
  display/processing settings via `getSetting`/`setSetting` +
  `settingschange`, `themeRamp`/`themeRampSample` visualizer ramps,
  `ghostAmount` DMG LCD ghosting), `static/js/settings.js` (Phase 6:
  slide-in settings panel — theme buttons + per-theme display controls
  (incl. the CGB palette-pack dropdown fed by /api/palettes),
  dismissed by [X], Esc or click-outside; brightness/contrast/saturation/
  bivert apply ONCE to the whole UI as a single CSS `filter` on `#app`
  (`applyGlobalFilter`) — there is no per-canvas adjustment path),
  `static/js/controls.js` (Phase 7: `enhanceControls(root)` replaces
  native number spinners with HUD steppers (arrow SVG masks tinted by
  currentColor) and file inputs with TEXT-ONLY `[BROWSE]` buttons — hooked
  into every tab activation in main.js),
  `static/js/viz/shaders.js`
  (original GLSL: CRT/DMG post passes, electric/quantum fragment shaders,
  FLUX_FRAG domain-wall flow viz, half-res
  `makePostPipeline` + raw-WebGL `makeShaderCanvas`, both with ping-pong
  LCD ghosting when `ghostAmount() > 0`; DMG_FRAG quantizes through
  `uPal[8]`/`uPalLum[8]`/`uPalCount` — NEAREST-stop palette UNIFORM
  ARRAYS (`paletteStops`, installed by `setPalette(hexArray)` via
  `pipeline.setTheme` from the ACTIVE `themeRamp()`, palette packs/
  subthemes included, CGB's 56 stops downsampled to 8; the scene is
  already themed so nearest-stop snapping, not a linear lum→band remap —
  the old `lum<0.25/0.5/0.75` bands crushed the light LCD background.
  The earlier 256×1 `uPalTex` DataTexture LUT rendered BLACK on three
  r170's WebGL2 renderer (RGBFormat/UNSIGNED_BYTE falls through
  getInternalFormat to unsized gl.RGB) — do not reintroduce it;
  `uPalEnabled`=0 falls back to the 4-band `uRamp0..3` path;
  `uBivert` stays 0, bivert happens upstream in theme.js;
  QUANTUM_FRAG takes `uDensity`/`uDensityOn` for the orbitals radial
  profile coupling). All four three.js tabs pass `themeRamp()` (never
  `THEMES[...].ramp`) to `pipeline.setTheme`),
  `_png.py` (stdlib PNG encoder + 8-bit RGB/RGBA decoder with all five
  filter types — the decoder serves the Terrarium terrain tiles), `jobs.py` (JobManager thread pool,
  bounded `job.history` replay buffer, `job.matrix` holds verified results
  off-JSON), `static/` (no-build ES-module frontend; tabs: Matrix Lab,
  Search Studio, Micromag Sim, HOA Studio, Terrain, Orbitals, Library,
  Materials (`js/tabs/materials.js` — `#mat-layer-select` CLOTH/TOUCH/META:
  two-layer yarn, mutual-cap electrodes, spin-ice unit cell + Walsh
  lattice from `materials.py`; gap_frac and flux-tile size (4/8/16) are
  exposed in the Source panel; KiCad export via design_type=materials,
  FOOTPRINT/BOARD preview toggle after export — board preview is the
  generated `.kicad_pcb` with discrete B.Cu pads, no GND pour),
  Filter (`js/tabs/filter.js` — `#flt-layer-select` DESIGN/EVOLVE/KICAD:
  per-kind TOPOLOGY select (stepped-Z LPF, gap+stub HPF, hairpin BPF,
  open-stub BSF + lumped lc/dc_lc/c_shunt/qw_tl/rc/crc/rl); S21/S11
  canvas; BOM table with engineering values for lumped topos; KiCad
  preview + one-shot download incl. `.kicad_block` design-block zip for
  lumped; SA streams IL/RL/rejection and is gated to distributed topos),
  Antenna (`js/tabs/antenna.js` — `#ant-layer-select` DESIGN/PARTS/
  FIELDS/EVOLVE/KICAD/SMITH/ARRAY panels; FIELDS/EVOLVE stream job frames over
  `/ws/job/{id}`, DESIGN auto-fires the parts query and exposes one-shot
  KiCad downloads, PARTS formats `size_mm` as L×W×H (arrays must not be
  passed to `el()` — that threw and painted an empty table) and shows
  overlap % when `/parts` falls back from full-cover, KICAD is a
  standalone export form (PATCH/MEANDER IFA/MIFA/LOOP/LIBRARY/LAST
  EVOLVED) with a FOOTPRINT/BOARD canvas preview (red F.Cu on blue
  B.Cu via `drawKicadPrims`, not the theme LUT), JLCPCB design_params
  table, and per-file download links — LIBRARY scales
  `RF_Antenna.pretty`, EVOLVE has WIRE/PCB topology plus Export KiCad,
  SMITH renders the MoM Z_in sweep on an interactive Γ plane with hover
  readout; ARRAY is the phased-array designer (themed polar plot of
  total_db over af_db at a −40 dB floor, steer/grating-lobe guide rays,
  beam-metrics stat table);
  site survey lives on the Terrain tab (`[GENERATE]`/`[SURVEY]`);
  job results keep long payloads
  (pattern_png_b64/resonance_note/points) OUT of the DOM — whitelist
  stat rows + [RESULT JSON] blob download; viz canvases capped via
  `.ant-cap` in app.css), Library (`js/tabs/library.js` —
  `#lib-layer-select` MAP/CHALLENGES: MAP is the construction-DAG +
  det-bound chart; CHALLENGES is DARPA's 23 mathematical challenges
  with per-challenge status/engine chips that deep-link via
  `hoa64:open-tab`), Noise (`js/tabs/noise.js` — DiT classifier
  training with a loss/acc strip chart, muon|adamw selector, 19-class
  chip list (* = synthesized RF baseband), + WAV/live analysis
  (mic or local wifi/ble radio capture via a `#noi-a-source` selector,
  RF runs show capture stats in the status line)
  with mel spectrogram and class-probability bars), Microcontroller
  (`js/tabs/microcontroller.js` — `#mcu-layer-select` LED/MESH/EDGE:
  LED paints WS2812 frames (serpentine GRB) and downloads
  ESP32/Teensy/CircuitPython firmware or pushes frames to a device IP;
  MESH (ALPHA — RSSI tomography, no CSI) downloads ESP-NOW node sketches,
  collects the gateway RSSI matrix, draws link-strength/delta-vs-baseline
  viz, exports `mesh_field.json` for the external spatialxr project;
  EDGE downloads hadamard_core/flux_map/terrain_fbm kernels as
  CircuitPython/Rust no_std/bare-metal C from `mcu.py`) —
  cross-tab deep links via `hoa64:open-tab`/`hoa64:payload` events in
  `main.js`; `js/viz/stripchart.js`. Single-viewport UI convention: each
  tab keeps ONE main viewport and swaps content in place — Matrix Lab's
  ℍ³ transmute morphs the 2D matrix into the 3D ball via `startMorph`
  (`sp-overlay` inside `.canvas-wrap`), Search Studio merges
  status/charts/preview into one Run panel (`.run-viz`), Micromag/Terrain/
  Orbitals have `data-layer` selectors (`#sim-layer-select`,
  `#ter-layer-select`, `#orb-layer-select` — matrix/energy/grad/flux,
  heightfield/octaves, 3D/XZ-proj respectively); Micromag's waveforms are
  ONE strip chart with `data-series` toggles ([E][E_DEM][E_EXCH][E_ANIS][E_GOAL][T],
  stripchart.js `setVisible`; E_GOAL is flat 0 unless a library goal is
  active) and goal-attraction controls (`sim-goal` [EVOLVE TO LIBRARY
  GOAL] + `sim-lam-goal`, plus a `tune-lam-goal` live-retune slider); Terrain
  has a `[GENERATE]`/`[SURVEY]` sidebar switch — GENERATE is Hadamard fBm
  with per-octave [MUTE]/[SOLO]/[FULL] (`data-oct`, client-side
  recombination from `layers_f32`, in-place mesh update); SURVEY is the
  SRTM site survey (path profile left, satellite-textured 3-D link right,
  blank/same RX probes 8×400 m bearings); Orbitals' XZ
  projection is computed client-side from the point cloud (`splatXZ` →
  heatmap.js). three.js axis
  convention everywhere: ground = XZ, up = +Y — terrain displaces along
  +Y after `rotateX(-π/2)`, orbitals map physics coords as
  three(x,y,z) = (−p.y, p.z, −p.x)), `selftest.py`
  (`python -m hoa64.webapp.selftest`). Requires fastapi/uvicorn/httpx;
  the CLI imports it lazily.
- Hadamard modules: `hadamard.py` (bitset core, check/verify/normalize,
  Sylvester/Paley/Kronecker, ILS max-det search, `selftest()`),
  `finite_field.py` (11 prime-power extensions), `miyamoto.py`,
  `williamson.py`, `cw_construction.py`, `gcp_hadamard.py`,
  `row_builder.py`, `circulant_search.py`.
- Generative-lab modules (Phase 5a, library style with selftests):
  `terrain.py` (Hadamard-layered Perlin fBm: `perlin2d`,
  `hadamard_noise`, `terrain` → heightmap + display `layers` + signed
  amp-weighted `contribs` for mute/solo recombination), `orbitals.py` (hydrogenic real orbitals
  on the SN3D SH convention — `radial_wavefunction`, `orbital_grid`,
  `sample_orbital` rejection sampler; `_y_real_sn3d` is a vectorized
  replica of `basis._sn3d_one` pinned by selftest cross-check),
  `hadamard_space.py` (ℍ³ transmute: row-simplex PCA → Poincaré ball,
  orthogonal-circle geodesics, hyperboloid lattice lift; unit-ball frame
  with curvature in the metric, κ = 0 flat display mode),
  `em_physics.py` (antenna-design physics core: general lossy-dielectric
  propagation `medium_params` (α/β/γ, η, skin depth) over the `MEDIA`
  table [air/fresh water/seawater], Friis `link_budget` with medium
  attenuation, `ANTENNA_TYPES` registry of textbook builders — dipole,
  monopole, loop, patch, helix, yagi, slot, pifa/meander — each
  returning dimensions/Z/gain/BW + a vectorized normalized
  `pattern(theta, phi)`, and `stokes(ex, ey)` polarization
  analysis; all dimensions key off the medium wavelength) — plus the
  phased-array API: `array_taper` (uniform/binomial/Dolph–Chebyshev at a
  given SLL, `TAPERS` registry), `array_factor` (progressive-phase
  steering), `array_metrics` (HPBW / SLL / grating-lobe angle / taper-
  efficiency directivity), and `design_array` (pattern multiplication with
  the element pattern, θ/af_db/total_db grids + layout).
- Antenna-lab modules (ALPHA BUILD — needs extensive field testing and a
  larger parts database; library style with selftests, all physics-based —
  no heuristics): `antenna_design.py` (`SiteConditions` + `recommend` —
  ranks `ANTENNA_TYPES` by exact fit factors: fractional BW, size, link
  budget at f_lo, medium attenuation e^(−αλ), polarization mismatch,
  viability constraints; composite = product, `explain()` trace),
  `parts_db.py` + `webapp/data/antenna_parts.json` (local curated
  off-the-shelf antenna DB, 103 rows — everythingRF has no API and
  403-blocks scraping, so rows mirror their listing fields and carry
  `erf_url` deep links; dual-band parts are one row per lobe;
  `match(spec)` freq-coverage gate + gain/size scoring; `/api/antenna/parts`
  with `partial=None` returns full-cover hits, then every overlapping
  row if the span has no single covering part),
  `fdtd.py` (3-D Yee FDTD: lossy-dielectric Cayley updates, graded
  sponge BC [not CPML], soft sinusoidal point source, air/water/
  air-water-interface media, per-frame |E| mid-plane slices + Stokes
  phasor polarization, radial α_fit validated against theory; same
  callback/stop_flag/live_params streaming contract as `micromag_sa`),
  `antenna_evo.py` (thin-wire Method of Moments — Pocklington EFIE,
  pulse basis, reduced kernel, delta-gap feed; power-consistent, dipole
  Z_in ≈ 73+j42 Ω at thin radius, resonance at 0.48 λ — plus
  `antenna_sa`: Hadamard-row-seeded meander-walk topology annealing with
  length-invariant corner-flip/swap moves and a multi-term
  match/gain/compactness objective — `topology="pcb"` uses λ/4 length,
  20 mil trace, E_dfm/E_rl (JLCPCB 5 mil floor + RL ≥ 10 dB) and
  exports via `kicad_gen.footprint_from_walk`), `kicad_gen.py`
  (procedural KiCad 7 `.kicad_mod`/`.kicad_pcb` for
  patch/meander-IFA/MIFA/loop plus scaled `RF_Antenna.pretty` library
  bases and evolved walks; JLCPCB design_params + s-expr preview
  prims (`layer` prefers a real *.Cu token; `fill` is captured so
  silk/Edge.Cuts stay `none`); Wheeler/Hammerstad 50 Ω feedline, 6h ground, keep-out under
  the radiator; `footprint_from_layout` honors a per-rect `layer` so
  materials boards emit F.Cu/B.Cu pads (silk is `fp_rect`, never a
  pad); `board_from_footprint(..., gnd_pour=False)` for
  `design_type="materials"` — a full-board B.Cu pour would short the
  reverse-layer cells; also `footprint_filter` / `design_type="filter"` for
  stepped/hairpin/stub RF filters from `rf_filter.layout_mm` plus
  `schematic_lumped`/`design_block_lumped` — KiCad ≥8 `.kicad_block`
  folders (embedded-lib_symbols `.kicad_sch` + block.json + board + zip)
  for lumped filter topos; S-expr
  parser validity gate + `kicad-cli pcb upgrade` check when available),
  `materials.py` (cloth/touchpad/metamaterial layouts from
  `flux_tiles`; open-sheet 4-connected electrodes, wall-bond
  capacitors, H.8 spin-ice unit cell + Walsh lattice; `design` takes
  `gap_frac` and `tile` (4/8/16) options),
  `rf_filter.py` (PCB RF-filter physics: Butterworth/Chebyshev g-values,
  Wheeler microstrip, ABCD cascade S-params with tanδ+Rs loss, Marki
  IL≈4.343 Σg/(Δ Q_u), everythingRF digest targets RL≥10 dB and
  40 dBc @ 10 % from the edge; `filter_sa` Hadamard-seeded section
  perturbation (distributed topos only); `design_filter(..., topo=...)`
  dispatches per-kind `TOPOS` — distributed stepped/stub/hairpin plus
  lumped `lc` ladders (Pozar 8.3/8.4), `dc_lc` nodal C-coupled and
  `c_shunt` combline-style BPF, `qw_tl` Richards/Kuroda commensurate
  lines, passive `rc`/`crc`/`rl` staged ladders (approximate in a 50 Ω
  environment — corner shifts documented in the docstrings); lumped
  designs carry a `components[]` BOM and lay out as 0805 pad cascades),
  `site_survey.py` (virtual site survey: AWS Open Data SRTM Terrarium
  tiles [no key, cached to ~/.cache/hoa64/terrain], tight DEM window +
  Esri World Imagery JPEG mosaic [Pillow; cached as PNG under
  ~/.cache/hoa64/imagery], great-circle path profiles, 4/3-earth bulge
  + first-Fresnel geometry, Deygout knife-edge diffraction,
  `em_physics.link_budget` closure → verdict; coincident/blank RX
  probes eight 400 m bearings and keeps the clearest hop),
  `noise_data.py` (NOISEX-92 noise DB — 15 recorded .mat from
  spib.linse.ufsc.br, cached to ~/.cache/hoa64/noisex92, plus 4
  synthesized RF baseband envelopes `ble`/`wifi`/`zigbee`/`lora`
  (`SYNTH_CLASSES`; cadence/duty/spectral envelope at fs=19.98 kHz,
  not the RF carrier); HTK log-mel DSP with fixed dB normalization;
  `load_noise` dispatches synth classes locally, `download` refuses them),
  `dit_noise.py` (ALPHA BUILD — DiT-backbone noise classifier, Peebles & Xie
  adaLN-Zero blocks used discriminatively, lazy torch, `train_model`
  job-friendly with callback/stop_flag, checkpoint `dit_noise.pt`;
  default optimizer is `muon.Muon` (AdamW fallback on 1-D params);
  window-level train/val split over 15 long recordings + 4 synth
  classes means val_acc≈1 reflects within-recording familiarity —
  needs a larger database and unseen-source testing before
  generalization claims),
  `rf_capture.py` (ALPHA BUILD — live radio-telemetry capture for the
  analyzer: polls /proc/net/dev (wifi) and the HCIGETDEVINFO ioctl (ble)
  counters at ~100 Hz, renders the measured duty envelope into audio
  baseband with `noise_data._shaped_noise` — the synth_waveform
  envelope-equivalent family with measured, not scripted, timing;
  managed-mode wifi sees this station's traffic only; zigbee/lora
  report unavailable, idle captures are never fabricated).
- Program-frame / optimizer modules: `darpa_challenges.py` (the 23
  DARPA mathematical challenges + honest tooling alignment —
  `active`/`partial`/`latent`/`none`; statuses are *not* progress
  toward solutions; `rh.py` is the single `active` anchor),
  `muon.py` (Muon/Dion3 optimizer: cursed-quintic Newton–Schulz
  orthogonalization of the momentum matrix — S′ ∈ ~[0.5, 1.5], not
  exact U Vᵀ — optional Gram-NS + row-subsampled orthogonalization,
  AdamW fallback on ndim<2; lazy torch factory).
- Microcontroller-lab module (ALPHA — firmware is template-generated and
  hardware-tested only by the user): `mcu.py` — WS2812 GRB frame packing
  (`pack_frame`/`pack_frames`, serpentine remap), `led_firmware`
  (ESP32/Teensy Arduino FastLED + CircuitPython sketches with an HTTP
  `POST /frame` raw-GRB contract), `mesh_firmware` (ESP-NOW RSSI
  tomography nodes, gateway serves `GET /mesh`), and `export_engine`
  (hadamard_core / flux_map / terrain_fbm → CircuitPython / Rust
  `#![no_std]` / bare-metal C). Each kernel exists first as a plain
  Python reference (`py_*`) pinned against `hadamard.py`/`micromag.py`
  in the module selftest; the C/Rust/CircuitPython bodies are templates
  of the same algorithms.
- Search engines/daemons (standalone scripts, each does
  `sys.path.insert(0, parent)` then `from hoa64....` imports):
  `evolve.py` (gap-filling daemon, constructions only), `search_daemon.py`
  (aggressive: tries all engines on every gap), `micromag.py` (simulated
  annealing + swap moves, `flux_map` domain-wall density viz,
  `flux_tiles` H.8 tessellation catalog (Sylvester / A⊗H₈ → exactly
  four 8×8 wall tiles at every dyadic scale, Walsh placement so
  H.256 counts 341/171/171/341 and top-left = H.128; Paley does
  not tile; orders ≡ 4 mod 8 cannot; open uses noted in
  `flux_tiles` docstring; `lam_tile` SA prior is live;
  `lam_z` Gerzon H₂ prior + 45°/225° pair-flip when > 0;
  cloth/touch/meta live in the Materials tab),
  optional
  `goal`/`lam_goal` attraction — E_goal = lam_goal per entry disagreeing
  with ±goal, sign fixed at start since global sign is gauge), `tile_search.py` (2×2 H2-cell SA),
  `gerzon.py` (Gerzon 1975 AB module: A-format corners → WXYZ; H₄ after
  L_F↔L_B column swap; |Z_int| trichotomy cohesive/H₂/wall; SA + ILS
  engine in Search Studio; 45°/225° pair is the dedicated cancel move),
  `holographic.py` (S = A/(4ℓₚ²): S is the entropy of the region of
  volume V, A is the Planck-unit area of the surface bounding V —
  domain-wall area on the ±1 grid; SA + ILS),
  `crown.py` (Liu 2022 spherical-crown diffraction: RS kernel, OPSF
  occlusion utilizing, d_m / d_m-max, 2-D FFT propagate; SA + ILS),
  `rnn_hadamard.py` and `sig_predictor.py` (PyTorch LSTM fitness models),
  `game_of_hadamard.py` (construction DAG viz; `classify_orders` feeds
  /api/dag), `matrix_viz.py`
  (pixel-art matrix visualizer).
- `rh.py` — Riemann-hypothesis |Δₙ| bound consistency checker (unrelated
  side experiment; used by daemons as a checksum-style gate via `rh_check`).
- `rebuild.py` — decompresses `matrices/*.csv.gz` and rebuilds the whole
  CSV library via `evolve.py`.
- `sage/` — a full **SageMath source checkout** (own git repo, `.venv`,
  build tree). It is gitignored here and is only a tool: SageMath is used
  for orders > 2000. Do not treat it as part of this project; do not modify
  it unless explicitly asked.
- `matrices/` — 12 gzipped Alpoge Hadamard matrices (orders 668–1964,
  formerly open). `data/h668_mod64.npy` — Eliahou's 64-modular H(668) warm
  start. `~/open_hadamard/` (outside the repo, gitignored pattern) — the
  full generated CSV library the daemons read/write.
- `*.txt`, `*.pdf`, `Had28.tar.gz`, `Anaconda3-*.sh` — reference material,
  notes, and installers; gitignored, not code.

## Environment & running

There is **no** `pyproject.toml`, `setup.py`, or `requirements.txt` — the
package is not pip-installable and there is no build step. Run it in place
from the parent directory:

```bash
cd /home/bbear
python3 -m hoa64.cli --help
python3 -m hoa64.cli hadamard --selftest      # the test suite (see below)
python3 -m hoa64.cli hadamard --generate 668
python3 -m hoa64.cli analyze scene.wav --ambix -o report.json
python3 -m hoa64.cli serve                    # HTTP API on 127.0.0.1:8765
python3 hoa64/evolve.py                       # gap-filling daemon
python3 hoa64/search_daemon.py                # overnight search daemon
python3 hoa64/rebuild.py                      # rebuild CSV library (needs SageMath)
```

- **Working interpreter**: `~/miniforge3/envs/sage-dev/bin/python`
  (Python 3.12.13, NumPy 2.4.3, PyTorch 2.13.0+cu126). This is the env the
  project is developed and run with.
- The system `python3` (3.13) has **no NumPy** and will fail. The old
  `~/anaconda3` Python 3.9 has NumPy 1.21 and can import the package but has
  no PyTorch. Prefer the `sage-dev` env for everything.
- Required: NumPy. Optional: PyTorch (`rnn_hadamard.py`, `sig_predictor.py`),
  torchvision (`detector.py` image backend), a running ComfyUI server
  (`conditioning.py`), PulseAudio/`parecord` (`live_audio.py`), SageMath
  (orders > 2000, `rebuild.py`). Optional imports are done lazily inside
  functions — keep it that way.
- Generated model files (`rnn_hadamard.pt`, `sig_simple.pt`), CSVs, `.npy`
  and `.txt` artifacts are gitignored and may be absent from a fresh clone.

## Testing

There is **no pytest/unittest suite and no CI**. The canonical check is the
in-module selftest:

```bash
cd /home/bbear && ~/miniforge3/envs/sage-dev/bin/python -m hoa64.cli hadamard --selftest
```

It verifies Sylvester/Paley/construction correctness, the CSV-imported
orders 668/716/892, the incremental Gram-matrix update against brute force,
the dF flip formula, gauge invariants, and a small end-to-end max-det search.
It passes as of Aug 2026. Beyond that, correctness is established by
`hadamard.verify(H)` / `check(H, det=True)` (orthogonality, H2 balance,
determinant bound) — any new construction or search result **must** be
checked with `verify` before being exported to CSV (the daemons do this).
When changing core math, run the selftest; when adding a module, include a
small `if __name__ == "__main__":` demo/self-check in the prevailing style.

## Code style conventions

- NumPy-heavy, plain functions, minimal classes (`dataclass` where a record
  is needed, e.g. `report.py`, `stream.py`, `detector.py`).
- Modern typing with `from __future__ import annotations` at the top of
  library modules; standalone daemon scripts use compact imports
  (`import os, sys, time, math` on one line) and no type annotations.
- Docstrings are substantive and mathematical — modules open with the
  theory (formulas, references) before the code. Match that: explain the
  math in the module docstring when extending constructions/searches.
- Unicode punctuation (em-dashes, ≣, ×, subscripts) is used freely in
  docstrings and comments — follow the existing tone.
- Matrices are `np.int8` ±1 arrays; `hadamard.py` also has a bitset core
  (rows packed into Python ints). CSVs are comma-separated ±1
  (`np.savetxt(..., delimiter=",", fmt="%d")`).
- Minimal-change culture: small single-purpose modules, no abstraction
  frameworks, no linters/formatters configured. Keep new code stylistically
  consistent with the file it lives in.

## Deployment & operations

- Nothing is deployed as a package. "Deployment" is running `server.py`
  (`hoa64 serve`) or the daemons locally on this machine; additionally a
  LAN mirror runs on `192.168.1.107` ("generator"): plain rsync copy at
  `~/hoa64` (tracked files via `git ls-files`), venv `~/hoa64-venv`
  (numpy, torch+xpu, fastapi/uvicorn), serving
  `hoa64 webapp --host 192.168.1.107 --port 8770` with a ufw rule
  allowing 8770/tcp from 192.168.1.0/24. SSH deploy key:
  `~/.ssh/hoa64_deploy` (user bbear).
- Daemons write into `~/open_hadamard/` (outside the repo); `evolve.py`
  auto-cascades Kronecker multiples whenever a new order is filled.
- Git: single branch `master`, remote `github.com/bbeartheancient/hoa64`.
  Commit messages are short imperative sentences ("Add ...", "Fix ...").
  Data products are kept out of git via `.gitignore`.

## Security considerations

- `server.py` binds to `127.0.0.1` by default and is unauthenticated
  stdlib HTTP — do not expose it on a public interface without adding auth.
- `analyze_file`-style endpoints take filesystem paths from JSON bodies;
  treat the API as trusted-local only.
- Never commit the gitignored data/model artifacts (`*.csv`, `*.npy`,
  `*.pt`, `data/`, `matrices/*.gz` are the exceptions that ARE committed).
- The `sage/` tree and the large installers (`Anaconda3-*.sh`,
  `Had28.tar.gz`) are third-party material — don't modify or redistribute
  them as part of changes here.
