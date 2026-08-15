"""Evolutionary antenna discovery: thin-wire MoM evaluator + Hadamard-seeded
topology annealing.

The hoa64 toolkit's Hadamard/micromag machinery (Sylvester seeds,
invariant-preserving moves, multi-term SA with live retuning) applied to
wire-antenna geometry: the simulated annealer of `micromag.py` re-cast so
that a "spin configuration" is a wire walk, the conserved polarity count is
the conserved wire length (electrical length ≣ resonance), and the energy
is a real electromagnetic objective evaluated by a Method of Moments solve.

Half 1 — thin-wire Method of Moments (``wire_mom``)
===================================================

Pocklington's electric-field integral equation for a thin perfectly
conducting wire (Balanis, *Antenna Theory*, ch. 8; Harrington, *Field
Computation by Moment Methods*): enforcing the boundary condition that the
tangential scattered field cancels the incident field on the wire surface
gives, for current I(l) along the wire axis,

    E^i_l = jωμ ∫ I(l′) (φ̂·φ̂′) G dl′
            + 1/(jωε) · ∂/∂l ∫ I(l′) ∂G/∂l′ dl′,     G(R) = e^(−jkR)/(4πR),

with k = ω√(με) the medium wavenumber (complex in lossy media, via
`em_physics.medium_params`).  We expand I in **pulse basis functions**
(piecewise-constant on each segment) and **point match** at segment
centers (delta testing functions).  The ∂/∂l′ integral collapses onto the
segment endpoints and the outer ∂/∂l becomes a finite difference across
the test segment, yielding the classic pulse/point-matching matrix element

    Z_mn = j k η (φ̂_m·φ̂_n) Δl_n G(c_m, c_n)
           − j (η/k) [G(e_m,e_n) − G(e_m,s_n) − G(s_m,e_n) + G(s_m,s_n)] / Δl_m,

since kη = ωμ and η/k = 1/(ωε).  The kernel is the **reduced kernel**
R = √(d² + a²): the field point is never allowed closer than the wire
radius a, which removes the 1/R singularity and models the current as a
filament on the axis with the field observed on the cylinder surface.
The source is a 1 V **delta-gap generator** at the feed vertex: the
right-hand side is the incident *field*, V_m = (1 V)/Δl_m·δ_{m,feed} —
equivalently the Galerkin-pulse excitation ∫E^i dl = 1 V (the two differ
only by a row scaling of Z I = V and give identical currents).  Solved by
direct LU (n ≲ 10³ → the O(n²) build and O(n³) solve are both trivial).

Convergence notes (honest ones): the solution is exactly power-consistent
(the far-field-integrated radiated power equals the accepted power to
~6 digits) and the pattern/directivity are spot-on (half-wave dipole
D = 1.644 vs the theoretical 1.643).  The reduced kernel + delta gap
converges like the classic NEC-2-style thin-wire codes: R_in settles
within a few percent of the 73.1 Ω asymptote once Δl/a ≳ 8, while X_in
converges slowly from above — the 73.1 + j42.5 Ω value is the a → 0
asymptote, reached here with a = 1e-5 m (2h/a = 15000) and ~1000
segments (76.7 + j46.4 Ω, both within ±15 %); at the default physical
radius a = 1e-4 m the series resonance lands at L = 0.480 λ (|X| < 1 Ω),
in the canonical 0.475–0.48 λ window.

The far field of the pulse currents is the sum of short-dipole
contributions,

    E(r̂) = j k η e^(−jkr)/(4πr) Σ_n I_n Δl_n [φ̂_n − (φ̂_n·r̂) r̂] e^(jk r̂·c_n),

normalized to a 0..1 power pattern; the gain normalizes the pattern
maximum by the accepted power P = ½·Re{V·I_feed*} (efficiency 1, lossless
wire).

Half 2 — Hadamard-seeded topology annealing (``antenna_sa``)
============================================================

Search space for ``topology="meander"``: a planar walk of
N = `hadamard_order` unit steps, each step one of {+x, −x, +y, −y}
(codes 0..3).  The walk is seeded from Sylvester Hadamard rows
(`hoa64.hadamard.sylvester`): each ±1 row of N entries is folded pairwise
into N/2 direction codes by

    pair (h_2i, h_2i+1) → code = (1−h_2i)/2 + 2·(1−h_2i+1)/2
    (+1,+1) → 0 = +x,  (−1,+1) → 1 = −x,
    (+1,−1) → 2 = +y,  (−1,−1) → 3 = −y,

and two rows (chosen via `rng`) concatenate to the full N-step walk —
the Hadamard row's sign structure becomes the initial turn pattern, a
maximally-uncorrelated binary seed instead of a uniform-random one.

The step length is fixed so the total wire length is the resonant length:
N steps × Δ = λ_medium/2, i.e. Δ = λ_medium/(2N); the default bounding
box `bbox_m` is a λ_medium/4 square.

Search space for ``topology="pcb"``: the same planar walk, but total
length = λ/4 (printed IFA), trace radius = 10 mil (JLCPCB/Cypress
MIFA is 20 mil throughout), default bbox λ/8, plus DFM / return-loss
terms — E_dfm penalises folds tighter than 0.15 mm (5 mil floor) and
E_rl wants S11 ≤ −10 dB (90 % of incident power into the antenna).
The best walk exports as a KiCad footprint via
``kicad_gen.footprint_from_walk``.

**Invariant-preserving moves** (the analogue of micromag's
polarity-count-preserving swaps): every proposal keeps the total wire
length — hence the electrical length and rough resonance — exactly
constant, so the SA explores *shape* at fixed resonance:

* **90° corner flip** (70 %) — where steps i, i+1 are perpendicular, swap
  their order: the corner vertex mirrors across the pair's diagonal, both
  endpoints of the pair stay fixed, length is conserved;
* **step swap** (15 %) — micromag-style exchange of two arbitrary
  differing steps: preserves the whole step multiset, hence length AND
  total displacement;
* **end rotation** (15 %, and as fallback) — the first or last step
  rotates ±90° (needed for ergodicity; corner flips alone cannot move
  the endpoints).

Walks that retrace a unit segment (overlapping wires → coincident MoM
segments → singular Z; physically a junction the filament model cannot
represent) are rejected with a hard energy penalty before any solve.

**Objective** (multi-term, each term reported separately like
`micromag.total_energy`):

    E = w_z·|Z_in − 50|/50 + w_gain·(1 − min(gain_dbi, 6)/6)
        + w_size·(bbox_diagonal/bbox_m),

lower is better; a 50-Ω-matched, 6 dBi, point-sized antenna has E = 0.
The weights (and `cooling`) are re-tunable mid-run through `live_params`
keys "w_z" / "w_gain" / "w_size" / "cooling", read every 25-step frame;
terms are cached per geometry so a retune re-scores instantly.  MoM
solves are the evaluation budget, so every geometry (its step-code byte
string) is cached — revisits are free.

Streaming contract identical to `micromag_sa`: every 25 steps
``callback({"step", "T", "E", "best_E", "accepts", "geom"})`` with `geom`
the current-best geometry dict (points/z_in/gain/s11 plus the MoM
`pattern` callable — a conduit for live previews, including the mid-run
far-field pattern; the route layer pops it before JSON), and
`stop_flag.is_set()` is polled on the same cadence for early return.  Returns ``(best, info)`` with
``info = {steps, accepts, best_E, elapsed_s, best_design}``.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .em_physics import medium_params
from .hadamard import sylvester

# lattice direction table for the meander walk: codes 0..3 → ±x, ±y
_DIRS = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
_DIRS2 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int64)


# --- Half 1: thin-wire Method of Moments -------------------------------------

def _subdivide(points: np.ndarray, lam: float, seg_per_lambda: int) -> np.ndarray:
    """Refine a polyline so every straight piece samples λ at ≥ seg_per_lambda."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        raise ValueError("points must be an (M,3) array with M ≥ 2")
    out = [pts[0]]
    for p0, p1 in zip(pts[:-1], pts[1:]):
        length = float(np.linalg.norm(p1 - p0))
        if length < 1e-15:
            continue                        # drop zero-length pieces
        nsub = max(1, int(math.ceil(length / lam * seg_per_lambda)))
        for i in range(1, nsub + 1):
            out.append(p0 + (p1 - p0) * (i / nsub))
    out = np.asarray(out)
    if len(out) < 2:
        raise ValueError("polyline has no nonzero-length segments")
    return out


def wire_mom(points, f_hz: float, radius_m: float = 1e-4, medium="air",
             seg_per_lambda: int = 20, feed_idx: int | None = None) -> dict:
    """Thin-wire MoM analysis of a polyline wire antenna (Pocklington EFIE).

    Pulse basis + point matching, reduced kernel R = √(d² + a²), 1 V
    delta-gap source at the feed vertex.  `feed_idx` indexes the vertices
    of the *subdivided* wire and defaults to the middle vertex (the wire
    midpoint).  In lossy media the complex k = β − jα and η from
    `em_physics.medium_params` substitute directly (G = e^(−jkR)/4πR is
    then attenuated); the far-field phase uses the same complex k, an
    approximation valid while the antenna is small against 1/α.

    Returns dict with keys: z_in_ohm, i_segments (complex per-segment
    currents), s11 / s11_db (vs 50 Ω), resonance_note, pattern (vectorized
    callable pattern(theta, phi) → normalized power 0..1), gain_dbi
    (pattern maximum normalized by accepted power, lossless wire),
    directivity_dbi, points (subdivided vertices), n_segments, feed_idx,
    wavelength.
    """
    mp = medium_params(f_hz, medium)
    lam = mp["wavelength"]
    eta = mp["eta"]
    k = -1j * mp["gamma"]            # β − jα  (e^{jωt} convention)
    pts = _subdivide(points, lam, seg_per_lambda)
    s = pts[:-1]                     # segment starts
    e = pts[1:]                      # segment ends
    c = 0.5 * (s + e)                # centers (point-matching sites)
    vec = e - s
    dl = np.linalg.norm(vec, axis=1)
    u = vec / dl[:, None]            # segment unit vectors φ̂_n
    n = len(dl)
    a = float(radius_m)
    if feed_idx is None:
        feed_idx = n // 2            # middle vertex of the subdivided wire
    feed_idx = int(np.clip(feed_idx, 0, n - 1))

    def gmat(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """Reduced-kernel Green's function between point sets P (rows), Q (cols)."""
        d = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=-1)
        r = np.sqrt(d * d + a * a)
        return np.exp(-1j * k * r) / (4.0 * np.pi * r)

    gcc = gmat(c, c)
    # endpoint Green's functions for the ∂²G/∂l∂l′ finite difference
    g_ends = (gmat(e, e) - gmat(e, s) - gmat(s, e) + gmat(s, s))
    uu = u @ u.T
    z = (1j * k * eta * uu * dl[None, :] * gcc
         - 1j * (eta / k) * g_ends / dl[:, None])

    v = np.zeros(n, dtype=complex)
    v[feed_idx] = 1.0 / dl[feed_idx]   # 1 V delta gap → E = V/Δl at the feed
    i_seg = np.linalg.solve(z, v)
    i_feed = i_seg[feed_idx]
    z_in = 1.0 / i_feed                # 1 V applied
    s11 = (z_in - 50.0) / (z_in + 50.0)
    s11_db = 20.0 * math.log10(max(abs(s11), 1e-12))
    p_rad = 0.5 * float(np.real(np.conj(i_feed)))

    # --- far field of the pulse currents on an observation grid ---
    th = np.linspace(1e-4, np.pi - 1e-4, 61)
    ph = np.linspace(0.0, 2.0 * np.pi, 121, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    rhat = np.stack([np.sin(TH) * np.cos(PH),
                     np.sin(TH) * np.sin(PH),
                     np.cos(TH)], axis=-1)                  # (D,3)
    phase = np.exp(1j * k * (rhat @ c.T))                   # (D,n)
    ur = rhat @ u.T                                          # (D,n) φ̂_n·r̂
    w = (i_seg * dl)[None, :] * phase                        # (D,n)
    # Σ_n I_n Δl_n e^{jk r̂·c_n} [φ̂_n − (φ̂_n·r̂) r̂]
    e_far = (w @ u) - np.sum(w * ur, axis=-1)[..., None] * rhat
    e_far = e_far * (1j * k * eta / (4.0 * np.pi))           # e^{−jkr}/r dropped
    f_grid = np.sum(np.abs(e_far) ** 2, axis=-1)             # (D,)
    f_max = float(np.max(f_grid))
    f_grid = f_grid / f_max
    u_max = f_max / (2.0 * abs(eta))                         # r²-scaled intensity
    gain = 4.0 * np.pi * u_max / max(p_rad, 1e-30)
    omega_a = float(np.trapezoid(np.trapezoid(f_grid * np.sin(TH), th, axis=0), ph))
    directivity = 4.0 * np.pi / omega_a

    def pattern(theta, phi):
        t, p = np.broadcast_arrays(np.asarray(theta, dtype=float),
                                   np.asarray(phi, dtype=float))
        rh = np.stack([np.sin(t) * np.cos(p),
                       np.sin(t) * np.sin(p),
                       np.cos(t)], axis=-1).reshape(-1, 3)
        ph_n = np.exp(1j * k * (rh @ c.T))
        ur_n = rh @ u.T
        w_n = (i_seg * dl)[None, :] * ph_n
        ef = (w_n @ u) - np.sum(w_n * ur_n, axis=1)[:, None] * rh
        f = np.sum(np.abs(ef) ** 2, axis=-1) / f_max
        return np.clip(f.reshape(t.shape), 0.0, None)

    x_in = float(np.imag(z_in))
    note = (f"X_in = {x_in:+.2f} Ω — "
            + ("near resonance (|X| < 5 Ω)" if abs(x_in) < 5.0
               else "off resonance")
            + f", R_in = {float(np.real(z_in)):.2f} Ω")
    return {
        "z_in_ohm": complex(z_in),
        "i_segments": i_seg,
        "s11": complex(s11),
        "s11_db": s11_db,
        "resonance_note": note,
        "pattern": pattern,
        "gain_dbi": 10.0 * math.log10(max(gain, 1e-12)),
        "directivity_dbi": 10.0 * math.log10(max(directivity, 1e-12)),
        "points": pts,
        "n_segments": n,
        "feed_idx": feed_idx,
        "wavelength": lam,
    }


# --- Half 2: Hadamard-seeded topology annealing -------------------------------

def _seed_walk(order: int, rng: np.random.Generator):
    """Fold two Sylvester rows pairwise into N direction codes (see docstring)."""
    H = sylvester(order)
    if H is None:
        raise ValueError(f"hadamard_order {order} is not a power of 2 "
                         "(Sylvester seed)")
    r1 = int(rng.integers(1, order))
    r2 = int(rng.integers(1, order))
    while r2 == r1:
        r2 = int(rng.integers(1, order))
    rows = np.concatenate([H[r1].astype(np.int64), H[r2].astype(np.int64)])
    a = (1 - rows[0::2]) // 2
    b = (1 - rows[1::2]) // 2
    codes = (a + 2 * b).astype(np.int8)
    return codes, (r1, r2)


def _walk_points(codes: np.ndarray, step_len: float) -> np.ndarray:
    """Step codes → centered planar polyline vertices (z = 0)."""
    verts = np.vstack([np.zeros(3), np.cumsum(_DIRS[codes], axis=0)])
    verts = verts * step_len
    return verts - verts.mean(axis=0)


def _lattice_ok(codes: np.ndarray) -> bool:
    """False when the walk retraces a unit segment (overlapping wires).

    Coincident segments are singular for the filament MoM (touching wires
    form a junction the model cannot represent), so such walks are rejected
    with a hard energy penalty instead of a solve.
    """
    v = np.vstack([np.zeros(2, dtype=np.int64),
                   np.cumsum(_DIRS2[codes], axis=0)])
    edges = set()
    for a, b in zip(v[:-1].tolist(), v[1:].tolist()):
        edges.add((tuple(a), tuple(b)) if a <= b else (tuple(b), tuple(a)))
    return len(edges) == len(codes)


def _propose(codes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Length-preserving move: 90° corner flip (order swap of a perpendicular
    adjacent pair, 70 %), micromag-style swap of two arbitrary differing
    steps (15 %), or an end rotation (15 %, and as fallback)."""
    new = codes.copy()
    n = len(codes)

    def end_turn():
        j = 0 if rng.random() < 0.5 else n - 1
        # rotate step j by ±90°: x-type (0,1) ↔ y-type (2,3)
        new[j] = (2 + rng.integers(0, 2)) if new[j] < 2 else rng.integers(0, 2)

    r = rng.random()
    if r < 0.70 and n > 1:
        # 90° corner flip: swap a perpendicular adjacent pair
        i = int(rng.integers(0, n - 1))
        if (new[i] < 2) != (new[i + 1] < 2):     # perpendicular pair
            new[i], new[i + 1] = new[i + 1], new[i]
        else:                                    # parallel/antiparallel: no-op
            end_turn()
    elif r < 0.85 and n > 1:
        # micromag-style swap: exchange two arbitrary differing steps —
        # preserves the step multiset (length AND total displacement)
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if new[i] != new[j]:
            new[i], new[j] = new[j], new[i]
        else:
            end_turn()
    else:
        end_turn()
    return new


def antenna_sa(f_hz: float, medium: str = "air", topology: str = "meander",
               bbox_m: float | None = None, hadamard_order: int = 64,
               T_start: float = 2.0, T_end: float = 0.02,
               cooling: float = 0.995, max_steps: int = 2000,
               callback=None, stop_flag=None, live_params=None,
               rng=None) -> tuple[dict, dict]:
    """Hadamard-seeded simulated annealing over wire-walk topologies.

    See the module docstring for the encoding, the invariant-preserving
    move set and the objective terms.  Returns ``(best, info)`` where
    `best` = {points, z_in_ohm, gain_dbi, s11, s11_db, terms {E_z, E_gain,
    E_size}, seed_row} and `info` = {steps, accepts, best_E, elapsed_s,
    best_design} with best_design = best + the MoM `pattern` callable and
    run metadata.
    """
    if topology not in ("meander", "pcb"):
        raise ValueError(f"unknown topology {topology!r}; supported: 'meander', 'pcb'")
    rng = rng or np.random.default_rng()
    mp = medium_params(f_hz, medium)
    lam = mp["wavelength"]
    n_steps = int(hadamard_order)
    is_pcb = topology == "pcb"
    if is_pcb:
        # printed IFA: electrical length ≈ λ/4, 20 mil trace (JLCPCB MIFA)
        step_len = lam / (4.0 * n_steps)
        if bbox_m is None:
            bbox_m = lam / 8.0
        radius = 0.254e-3                     # 20 mil / 2
    else:
        step_len = lam / (2.0 * n_steps)      # total wire length = λ/2
        if bbox_m is None:
            bbox_m = lam / 4.0
        radius = lam / 1000.0                 # thin wire

    w = {"w_z": 1.0, "w_gain": 1.0, "w_size": 1.0}
    cache: dict[bytes, tuple[dict, dict]] = {}   # codes → (terms, mom result)

    def evaluate(codes: np.ndarray) -> tuple[dict, dict]:
        key = codes.tobytes()
        hit = cache.get(key)
        if hit is not None:
            return hit
        if not _lattice_ok(codes):
            terms = {"E_z": 4.0, "E_gain": 2.0, "E_size": 4.0}
            if is_pcb:
                terms["E_dfm"] = 4.0
                terms["E_rl"] = 4.0
            cache[key] = (terms, None)
            return terms, None
        pts = _walk_points(codes, step_len)
        res = wire_mom(pts, f_hz, radius_m=radius, medium=medium,
                       seg_per_lambda=20, feed_idx=n_steps // 2)
        span = pts.max(axis=0) - pts.min(axis=0)
        diag = float(np.linalg.norm(span))
        terms = {
            "E_z": abs(res["z_in_ohm"] - 50.0) / 50.0,
            "E_gain": 1.0 - min(res["gain_dbi"], 6.0) / 6.0,
            "E_size": diag / float(bbox_m),
        }
        if is_pcb:
            # JLCPCB 5 mil floor: penalise walks that fold tighter than 0.15 mm
            min_gap = 0.15e-3
            gap_pen = 0.0
            for i in range(0, len(pts) - 2, 2):
                d = float(np.linalg.norm(pts[i] - pts[i + 2]))
                if d < min_gap:
                    gap_pen += (min_gap - d) / min_gap
            terms["E_dfm"] = gap_pen
            terms["E_rl"] = max(0.0, (10.0 + res["s11_db"]) / 10.0)  # want S11 ≤ −10 dB
        cache[key] = (terms, res)
        return terms, res

    def energy(terms: dict) -> float:
        e = (w["w_z"] * terms["E_z"] + w["w_gain"] * terms["E_gain"]
             + w["w_size"] * terms["E_size"])
        if is_pcb:
            e += 0.5 * terms.get("E_dfm", 0.0) + 0.5 * terms.get("E_rl", 0.0)
        return e

    codes, (r1, r2) = _seed_walk(n_steps, rng)
    for _ in range(16):
        if _lattice_ok(codes):
            break
        codes, (r1, r2) = _seed_walk(n_steps, rng)
    else:
        # deterministic valid fallback: a staircase meander
        codes = np.tile(np.array([0, 2], dtype=np.int8), n_steps // 2)
    terms, res = evaluate(codes)
    e_cur = energy(terms)
    best_codes = codes.copy()
    best_terms, best_res, best_e = terms, res, e_cur

    t = T_start
    steps = accepts = 0
    t0 = time.monotonic()

    while steps < max_steps and t > T_end:
        trial = _propose(codes, rng)
        t_terms, t_res = evaluate(trial)
        e_new = energy(t_terms)
        delta = e_new - e_cur
        if delta < 0.0 or rng.random() < math.exp(-delta / max(t, 1e-10)):
            codes = trial
            terms, res = t_terms, t_res
            e_cur = e_new
            accepts += 1
            if e_new < best_e:
                best_codes = trial.copy()
                best_terms, best_res, best_e = t_terms, t_res, e_new
        steps += 1
        t *= cooling

        if steps % 25 == 0:
            if live_params is not None:
                retuned = False
                for key_w in ("w_z", "w_gain", "w_size"):
                    val = live_params.get(key_w)
                    if val is not None and float(val) != w[key_w]:
                        w[key_w] = float(val)
                        retuned = True
                cl = live_params.get("cooling")
                if cl is not None:
                    cooling = float(cl)
                if retuned:
                    # re-score incumbent and best under the new weights
                    e_cur = energy(terms)
                    best_e = energy(best_terms)
            if callback is not None:
                geom = {}
                if best_res is not None:
                    geom = {
                        "points": best_res["points"],
                        "z_in_ohm": best_res["z_in_ohm"],
                        "gain_dbi": best_res["gain_dbi"],
                        "s11": best_res["s11"],
                        # MoM pattern callable — lets the route layer stream
                        # the far-field pattern mid-run; popped with geom,
                        # never JSON-serialized
                        "pattern": best_res["pattern"],
                    }
                callback({
                    "step": steps, "T": t, "E": e_cur, "best_E": best_e,
                    "accepts": accepts,
                    "geom": geom,
                })
            if stop_flag is not None and stop_flag.is_set():
                break

    elapsed = time.monotonic() - t0
    if best_res is None:
        raise RuntimeError("antenna_sa: no valid geometry found")
    best = {
        "points": best_res["points"],
        "z_in_ohm": best_res["z_in_ohm"],
        "gain_dbi": best_res["gain_dbi"],
        "s11": best_res["s11"],
        "s11_db": best_res["s11_db"],
        "terms": dict(best_terms),
        "seed_row": {"order": n_steps, "row_indices": [r1, r2]},
    }
    best_design = dict(best)
    best_design.update({
        "pattern": best_res["pattern"],
        "resonance_note": best_res["resonance_note"],
        "f_hz": f_hz,
        "medium": medium,
        "step_len_m": step_len,
        "n_steps": n_steps,
        "step_codes": best_codes.tolist(),
        "kind": "pcb" if is_pcb else "wire",
    })
    info = {
        "steps": steps,
        "accepts": accepts,
        "best_E": best_e,
        "elapsed_s": elapsed,
        "best_design": best_design,
    }
    return best, info


# --- self-check ----------------------------------------------------------------

if __name__ == "__main__":
    import threading

    t_start = time.monotonic()

    # --- 1. center-fed half-wave dipole at 1 GHz: Z_in ≈ 73 + j42 Ω ---
    # 73.1 + j42.5 Ω is the a→0 thin-wire asymptote (induced-EMF /
    # exact-kernel limit), so push thin: a = 1e-5 m (2h/a = 15000) with
    # 960 segments (Δl/a ≈ 16, inside the thin-wire discretization rule).
    f0 = 1.0e9
    lam0 = medium_params(f0, "air")["wavelength"]
    half = lam0 / 2.0
    dipole_pts = np.array([[0.0, 0.0, -half / 2.0], [0.0, 0.0, half / 2.0]])
    d = wire_mom(dipole_pts, f0, radius_m=1e-5, seg_per_lambda=1920)
    z = d["z_in_ohm"]
    assert abs(z.real - 73.0) / 73.0 < 0.15, z
    assert abs(z.imag - 42.0) / 42.0 < 0.15, z
    assert d["pattern"](0.0, 0.0) < 1e-3                     # null on axis
    th_s = np.linspace(1e-3, np.pi - 1e-3, 2001)
    f_s = d["pattern"](th_s, 0.0)
    assert abs(th_s[int(np.argmax(f_s))] - np.pi / 2) < 0.01  # max broadside
    print(f"PASS  wire_mom dipole L=λ/2 @1 GHz: Z_in = {z.real:.1f} "
          f"{z.imag:+.1f}j Ω ≈ 73+j42 (±15 %), null on axis, max broadside, "
          f"gain = {d['gain_dbi']:.2f} dBi (D = {d['directivity_dbi']:.2f} dBi, "
          f"S11 = {d['s11_db']:.1f} dB)")

    # --- 2. resonance scan: |X_in| minimum near 0.475–0.48 λ ---
    # (physical wire radius a = 1e-4 m = 3.3e-4 λ — the resonant-length
    # end-effect shortening is radius-dependent)
    scan = []
    for frac in np.arange(0.44, 0.525, 0.005):
        L = frac * lam0
        pts = np.array([[0.0, 0.0, -L / 2.0], [0.0, 0.0, L / 2.0]])
        r = wire_mom(pts, f0, seg_per_lambda=960)
        scan.append((frac, abs(r["z_in_ohm"].imag), r["z_in_ohm"].imag))
    frac_r, x_min, x_signed = min(scan, key=lambda t: t[1])
    assert 0.46 <= frac_r <= 0.49, frac_r
    assert x_min < 5.0, x_min
    print(f"PASS  resonance scan: |X_in| minimum at L = {frac_r:.3f} λ "
          f"(X = {x_signed:+.2f} Ω), in the 0.475–0.48 λ thin-wire window")

    # --- 3. SA run at 2.45 GHz: valid geometry, E decreases, finite S11 ---
    sa_kw = dict(T_start=1.0, cooling=0.992)
    frames = []
    best, info = antenna_sa(2.45e9, max_steps=400, callback=frames.append,
                            rng=np.random.default_rng(11), **sa_kw)
    assert len(frames) >= 2
    assert frames[-1]["best_E"] < frames[0]["best_E"], (
        frames[0]["best_E"], frames[-1]["best_E"])
    assert best["points"].ndim == 2 and best["points"].shape[1] == 3
    assert np.isfinite(abs(best["s11"]))
    assert {"E_z", "E_gain", "E_size"} <= set(best["terms"])
    print(f"PASS  antenna_sa @2.45 GHz (400 steps): best_E {frames[0]['best_E']:.3f} "
          f"→ {info['best_E']:.3f} (accepts {info['accepts']}), "
          f"Z_in = {best['z_in_ohm'].real:.1f} {best['z_in_ohm'].imag:+.1f}j Ω, "
          f"S11 = {best['s11_db']:.1f} dB, gain = {best['gain_dbi']:.2f} dBi "
          f"[{info['elapsed_s']:.1f} s]")

    # --- 4. stop_flag honored mid-run ---
    flag = threading.Event()
    frames_stop = []

    def cb_stop(fr):
        frames_stop.append(fr)
        if len(frames_stop) >= 2:
            flag.set()

    _, info_stop = antenna_sa(2.45e9, max_steps=400, callback=cb_stop,
                              stop_flag=flag, rng=np.random.default_rng(5),
                              **sa_kw)
    assert info_stop["steps"] < 400, info_stop["steps"]
    print(f"PASS  stop_flag: early return after {info_stop['steps']} steps "
          f"(< 400)")

    # --- 5. determinism: same rng seed → same best_E ---
    _, i_a = antenna_sa(2.45e9, max_steps=200, rng=np.random.default_rng(7),
                        **sa_kw)
    _, i_b = antenna_sa(2.45e9, max_steps=200, rng=np.random.default_rng(7),
                        **sa_kw)
    assert i_a["best_E"] == i_b["best_E"], (i_a["best_E"], i_b["best_E"])
    print(f"PASS  determinism: seed 7 twice → best_E = {i_a['best_E']:.6f} both")

    # --- 6. PCB topology: λ/4 walk, 20 mil radius, DFM/RL terms ---
    best_pcb, info_pcb = antenna_sa(
        2.45e9, topology="pcb", max_steps=80,
        rng=np.random.default_rng(3), **sa_kw)
    assert info_pcb["best_design"]["kind"] == "pcb"
    assert {"E_dfm", "E_rl"} <= set(best_pcb["terms"])
    assert best_pcb["points"].ndim == 2 and best_pcb["points"].shape[1] == 3
    print(f"PASS  antenna_sa topology=pcb (80 steps): kind=pcb, "
          f"S11 = {best_pcb['s11_db']:.1f} dB, "
          f"E_dfm = {best_pcb['terms']['E_dfm']:.3f}, "
          f"E_rl = {best_pcb['terms']['E_rl']:.3f}")

    print(f"antenna_evo selftest: all checks passed "
          f"[{time.monotonic() - t_start:.1f} s]")
