"""3-D FDTD (Yee 1966) time-domain Maxwell solver for dipole radiation
in air / water / an air–water half-space.

**Yee scheme.**  Maxwell's curl equations in a lossy dielectric
(ε = εᵣε₀, μ = μ₀, conductivity σ),

    ∂E/∂t = (∇×H − σE)/ε,        ∂H/∂t = −(∇×E)/μ,

are discretized on the staggered Yee lattice: E components live at edge
centres, H at face centres, offset by Δt/2 in time (leapfrog).  With the
convention Ex[i,j,k] ≣ E_x(i+½, j, k), Hx[i,j,k] ≣ H_x(i, j+½, k+½) etc.,
the discrete curl updates are pure array slices, e.g.

    Ez(1:,1:,:-1) ← Ca·Ez + Cb·[(Hy[i,j,k] − Hy[i−1,j,k])/Δx
                               − (Hx[i,j,k] − Hx[i,j−1,k])/Δy].

The conductivity term is time-averaged (σE at the half step), giving the
standard lossy-dielectric Cayley coefficients

    Ca = (1 − σΔt/2ε)/(1 + σΔt/2ε),   Cb = (Δt/ε)/(1 + σΔt/2ε),

which stay stable and correct from σ = 0 (air) through the weakly
conducting fresh-water regime (σ/ωε ≈ 0.075 at 150 MHz).

**Courant stability.**  Δt = r·Δx/(v_max·√3), r = 0.95, with v_max the
largest phase velocity present (air in the interface configuration).
The cell size is chosen so the *smallest* medium wavelength spans
≥ 10 cells (default target 12): in the half-space run that is the
water wavelength, √80 shorter than in air.

**Absorbing boundary — graded sponge, not CPML.**  The outer
``SPONGE_CELLS`` (= 8) cells carry a polynomial-ramped artificial
conductivity σ_s(d) = σ_max·(1 − d/8)³ (d = distance from the wall),
with σ_max set by the usual round-trip reflection target R₀ = 10⁻²,
σ_max = −(p+1)·ln R₀/(2·η·8Δx), scaled by √εᵣ so the layer is locally
matched.  A *matched* magnetic loss σ* = σ_s·μ/ε is applied in the H
update (same Cayley form), which is what makes a pure sponge reasonably
reflectionless for near-normal incidence.  This is deliberately simple
and robust — it is NOT a convolutional PML; grazing-incidence waves and
late-time residuals reflect at the percent level, which shows up as a
mild standing-wave ripple on top of the radial decay.  Every frame is
labelled ``boundary_note = "sponge"`` for this reason.

**Source.**  A sinusoidal soft source Jz = A·sin(2πf·t) is *added* to Ez
at a single cell after the curl update (soft = the cell keeps its
curl-driven value plus the drive, so it radiates and scatters).  In the
interface run the feed sits three cells above the water surface, in the
air half.

**Phasor tracking / polarization.**  For polarization visualization the
solver keeps running sin/cos projections of the mid-plane field,
Σ E(t)·e^{−jωt} over one-period windows (W = round(T/Δt) steps), giving
complex phasors Ex, Ey, Ez at the source plane.  Ex/Ey feed
`em_physics.stokes` → per-pixel I, Q, U, V, axial ratio, tilt,
handedness.  The Ez phasor doubles as the *envelope* for the decay
measurement: after the last step, ln(|Ez_phasor|·r) along the four
mid-plane rays (±x, ±y through the source — the dipole far-field maximum;
the ±z rays sit in the pattern null and are excluded), starting at ≈ 0.7λ
to skip the dipole near field and stopping before the sponge, is
least-squares fit to ln A − α·r, yielding the measured attenuation α_fit.

**Numerical honesty.**  In air the wave spreads ballistically and
α_fit ≈ 0 (residuals come from sponge ripple and near-field leakage,
|α_fit| ≲ 0.03 Np/m at 12 cells/λ).  In fresh water α ≈ σ·η/2 ≈
1.05 Np/m is essentially frequency-independent in this low-loss regime,
so the self-test runs at 150 MHz where the ~0.8 m box spans a useful
number of Nepers; at this coarse resolution the fit reproduces the
analytic α to within **±35 %** (stated tolerance, dominated by the short
fit window and sponge ripple — not by the physics, which is exact Yee).

Streaming contract mirrors `micromag.micromag_sa`: `callback(frame)`
every `frame_every` steps, `stop_flag` (threading.Event) polled per
frame, `live_params["src_amplitude"]` re-read per frame.  `rng` is
accepted for API symmetry — the solver is deterministic.
"""

from __future__ import annotations

import math

import numpy as np

from .em_physics import EPS0, ETA0, MU0, MEDIA, medium_params, stokes

SPONGE_CELLS = 8          # graded-conductivity absorbing shell thickness
SPONGE_ORDER = 3          # polynomial ramp order of the sponge profile
SPONGE_R0 = 1e-2          # round-trip reflection target of the sponge
COURANT = 0.95            # fraction of the 3-D Courant limit
CELLS_PER_LAMBDA = 12     # default resolution target (≥ 10 required)
N_CAP = 96                # hard cap on the linear grid size
INTERFACE_SRC_OFFSET = 3  # feed height above the water surface, cells
FIT_R0_FRAC = 0.7         # decay fit starts at ≈ 0.7λ from the source


def _sponge_sigma(n, dx, eps_r):
    """Graded conductivity profile σ_s(i,j,k) for the absorbing shell.

    Polynomial ramp in the distance d to the nearest wall, amplitude
    matched to the local medium (∝ √εᵣ, i.e. ∝ 1/η) from the free-space
    round-trip-reflection formula.
    """
    ax = np.minimum(np.arange(n), n - 1 - np.arange(n)).astype(float)
    d = np.minimum(np.minimum(ax[:, None, None], ax[None, :, None]),
                   ax[None, None, :])
    ramp = np.clip((SPONGE_CELLS - d) / SPONGE_CELLS, 0.0, None) ** SPONGE_ORDER
    sig_max = (-(SPONGE_ORDER + 1.0) * math.log(SPONGE_R0)
               / (2.0 * ETA0 * SPONGE_CELLS * dx))
    return sig_max * np.sqrt(eps_r) * ramp


def _fit_alpha(phas, isrc, jsrc, n, dx, r0, rmax):
    """Fit |phasor| = A·e^(−αr)/r along the four mid-plane rays (±x, ±y).

    `phas` is the (n, n) Ez phasor on the source plane.  The rays stay in
    that plane (θ = 90° from the dipole axis) where the dipole pattern is
    maximal and clean 1/r far field; the ±z rays lie in the pattern null
    and are near-field dominated, so they are excluded.  Returns α (Np/m),
    or None when the fit window is too short.
    """
    if phas is None or rmax - r0 < 2:
        return None
    rs, ys = [], []
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for r in range(r0, rmax + 1):
            a = abs(phas[isrc + di * r, jsrc + dj * r])
            if a > 1e-30:
                rm = r * dx
                rs.append(rm)
                ys.append(math.log(a * rm))
    if len(rs) < 8:
        return None
    slope, _ = np.polyfit(np.asarray(rs), np.asarray(ys), 1)
    return float(-slope)


def fdtd_run(f_hz, medium="air", interface=False, n=48, span_m=None,
             frame_every=10, max_steps=400, pol_viz=True,
             callback=None, stop_flag=None, live_params=None, rng=None):
    """Run a 3-D Yee FDTD dipole-radiation simulation.  See module docstring.

    f_hz: drive frequency.  medium: key of `em_physics.MEDIA` (homogeneous
    box) — ignored when interface=True, which builds air over fresh water
    (z < z_mid water) with the source in the air half.  n: grid is n³,
    capped at 96.  span_m: optional physical box size; when given it
    overrides the λ/12 default cell size (a resolution warning lands in
    info if that drops below 10 cells per medium wavelength).

    Returns {"info": {...}, "frames": [frame, ...], "final_frame": frame}.
    info carries steps_run, stopped, dt_s, dx_m, cells_per_lambda,
    alpha_theory (source-medium α, Np/m), decay_measured (fitted α or
    None), plus geometry/resolution metadata.  Frames carry step, t_s,
    e_mid_xy / e_mid_xz ((n,n) float32 |E| slices through the source
    plane), emax, e_rms, boundary_note = "sponge"; with pol_viz they also
    carry "stokes" (dict of mid-plane arrays from `em_physics.stokes`),
    and interface runs add e_rms_lo / e_rms_hi (water / air halves,
    sponge excluded).
    """
    del rng  # deterministic solver; accepted for streaming-API symmetry

    n = int(min(max(int(n), 16), N_CAP))
    frame_every = max(1, int(frame_every))

    # --- media layout ------------------------------------------------------
    if interface:
        media_here = {"air": MEDIA["air"], "water": MEDIA["water"]}
        src_medium = "air"
    else:
        media_here = {str(medium) if isinstance(medium, str) else "custom":
                      MEDIA[medium] if isinstance(medium, str) else dict(medium)}
        src_medium = str(medium) if isinstance(medium, str) else "custom"
    mps = {k: medium_params(f_hz, v) for k, v in media_here.items()}
    lam_min = min(mp["wavelength"] for mp in mps.values())
    v_max = max(mp["phase_velocity"] for mp in mps.values())

    # --- cell size / time step ----------------------------------------------
    resolution_warning = None
    if span_m is None:
        dx = lam_min / CELLS_PER_LAMBDA
    else:
        dx = float(span_m) / n
        if lam_min / dx < 10.0:
            resolution_warning = (
                f"resolution {lam_min / dx:.1f} cells/λ_medium < 10 — "
                "dispersion and decay errors will be large")
    cells_per_lambda = lam_min / dx
    dt = COURANT * dx / (v_max * math.sqrt(3.0))
    omega = 2.0 * math.pi * f_hz
    period_steps = max(2, int(round(1.0 / (f_hz * dt))))

    eps_r = np.full((n, n, n), mps[src_medium]["eps_r"])
    sig_med = np.full((n, n, n), mps[src_medium]["sigma"])
    zmid = n // 2
    if interface:
        w = MEDIA["water"]
        eps_r[:, :, :zmid] = w["eps_r"]
        sig_med[:, :, :zmid] = w["sigma"]
    eps = eps_r * EPS0

    # --- update coefficients (lossy-dielectric Cayley form) ------------------
    sig_sp = _sponge_sigma(n, dx, eps_r)
    a_e = (sig_med + sig_sp) * dt / (2.0 * eps)      # electric: medium + sponge
    CaE = (1.0 - a_e) / (1.0 + a_e)
    CbE = (dt / eps) / (1.0 + a_e)
    b_h = sig_sp * dt / (2.0 * eps)                  # matched magnetic loss
    CaH = (1.0 - b_h) / (1.0 + b_h)
    CbH = (dt / MU0) / (1.0 + b_h)               # Δt/μ₀, μr = 1 everywhere

    # --- fields ---------------------------------------------------------------
    Ex = np.zeros((n, n, n)); Ey = np.zeros((n, n, n)); Ez = np.zeros((n, n, n))
    Hx = np.zeros((n, n, n)); Hy = np.zeros((n, n, n)); Hz = np.zeros((n, n, n))

    isrc = jsrc = zmid
    ksrc = min(zmid + INTERFACE_SRC_OFFSET, n - SPONGE_CELLS - 1) \
        if interface else zmid
    src_amp = 1.0

    # --- phasor accumulators (sin/cos projections over one-period windows) ---
    acc_c = {k: np.zeros((n, n)) for k in (("x", "y", "z") if pol_viz
                                           else ("z",))}
    acc_s = {k: np.zeros_like(acc_c[k]) for k in acc_c}
    phasor_done: dict[str, np.ndarray] = {}
    acc_cnt = 0

    def _mid(comp_arr):
        return comp_arr[:, :, ksrc]

    def _phasor(key):
        if key in phasor_done:
            return phasor_done[key]
        if acc_cnt > 0:
            return 2.0 * (acc_c[key] - 1j * acc_s[key]) / acc_cnt
        return np.zeros_like(acc_c[key], dtype=complex)

    frames = []
    stopped = False
    error = None
    step = 0
    D = SPONGE_CELLS

    for step in range(1, max_steps + 1):
        t = step * dt

        # H ← CaH·H − CbH·(∇×E)   (half-step, Faraday)
        Hx[:, :-1, :-1] = CaH[:, :-1, :-1] * Hx[:, :-1, :-1] - CbH[:, :-1, :-1] * (
            (Ez[:, 1:, :-1] - Ez[:, :-1, :-1])
            - (Ey[:, :-1, 1:] - Ey[:, :-1, :-1])) / dx
        Hy[:-1, :, :-1] = CaH[:-1, :, :-1] * Hy[:-1, :, :-1] - CbH[:-1, :, :-1] * (
            (Ex[:-1, :, 1:] - Ex[:-1, :, :-1])
            - (Ez[1:, :, :-1] - Ez[:-1, :, :-1])) / dx
        Hz[:-1, :-1, :] = CaH[:-1, :-1, :] * Hz[:-1, :-1, :] - CbH[:-1, :-1, :] * (
            (Ey[1:, :-1, :] - Ey[:-1, :-1, :])
            - (Ex[:-1, 1:, :] - Ex[:-1, :-1, :])) / dx

        # E ← CaE·E + CbE·(∇×H)   (full-step, Ampère with σE time-averaged)
        Ex[:-1, 1:, 1:] = CaE[:-1, 1:, 1:] * Ex[:-1, 1:, 1:] + CbE[:-1, 1:, 1:] * (
            (Hz[:-1, 1:, 1:] - Hz[:-1, :-1, 1:])
            - (Hy[:-1, 1:, 1:] - Hy[:-1, 1:, :-1])) / dx
        Ey[1:, :-1, 1:] = CaE[1:, :-1, 1:] * Ey[1:, :-1, 1:] + CbE[1:, :-1, 1:] * (
            (Hx[1:, :-1, 1:] - Hx[1:, :-1, :-1])
            - (Hz[1:, :-1, 1:] - Hz[:-1, :-1, 1:])) / dx
        Ez[1:, 1:, :-1] = CaE[1:, 1:, :-1] * Ez[1:, 1:, :-1] + CbE[1:, 1:, :-1] * (
            (Hy[1:, 1:, :-1] - Hy[:-1, 1:, :-1])
            - (Hx[1:, 1:, :-1] - Hx[1:, :-1, :-1])) / dx

        # soft sinusoidal Jz drive at the feed cell
        Ez[isrc, jsrc, ksrc] += src_amp * math.sin(omega * t)

        # phasor projections:  phasor = (2/W)·Σ E(t)·e^{−jωt}
        cw, sw = math.cos(omega * t), math.sin(omega * t)
        if pol_viz:
            acc_c["x"] += _mid(Ex) * cw; acc_s["x"] += _mid(Ex) * sw
            acc_c["y"] += _mid(Ey) * cw; acc_s["y"] += _mid(Ey) * sw
            acc_c["z"] += _mid(Ez) * cw; acc_s["z"] += _mid(Ez) * sw
        else:
            acc_c["z"] += _mid(Ez) * cw; acc_s["z"] += _mid(Ez) * sw
        acc_cnt += 1
        if acc_cnt == period_steps:
            for k in acc_c:
                phasor_done[k] = 2.0 * (acc_c[k] - 1j * acc_s[k]) / period_steps
                acc_c[k][:] = 0.0; acc_s[k][:] = 0.0
            acc_cnt = 0

        if step % frame_every == 0:
            emag2 = Ex * Ex + Ey * Ey + Ez * Ez
            emax = float(np.sqrt(emag2.max()))
            if not np.isfinite(emax):
                error = f"field blew up at step {step} (emax not finite)"
                break
            frame = {
                "step": step,
                "t_s": t,
                "e_mid_xy": np.sqrt(emag2[:, :, ksrc]).astype(np.float32),
                "e_mid_xz": np.sqrt(emag2[:, jsrc, :]).astype(np.float32),
                "emax": emax,
                "e_rms": float(np.sqrt(emag2.mean())),
                "boundary_note": "sponge",
            }
            if interface:
                inner = emag2[D:n - D, D:n - D, :]
                frame["e_rms_lo"] = float(np.sqrt(inner[:, :, D:zmid].mean()))
                frame["e_rms_hi"] = float(np.sqrt(inner[:, :, zmid:n - D].mean()))
            if pol_viz:
                st = stokes(_phasor("x"), _phasor("y"))
                frame["stokes"] = {
                    k: (v.astype(np.float32)
                        if isinstance(v, np.ndarray) and v.dtype.kind == "f"
                        else v)
                    for k, v in st.items()
                }
            frames.append(frame)
            if callback is not None:
                callback(frame)
            if live_params is not None:
                a = live_params.get("src_amplitude")
                if a is not None:
                    src_amp = float(a)
            if stop_flag is not None and stop_flag.is_set():
                stopped = True
                break

    # --- decay measurement: ln(|Ez|·r) = ln A − α·r on the last period ------
    mid_phasor = _phasor("z")
    lam_src = mps[src_medium]["wavelength"]
    cells_per_lambda_src = lam_src / dx
    r0 = max(3, int(round(FIT_R0_FRAC * cells_per_lambda_src)))
    rmax = min(isrc, jsrc, n - 1 - isrc, n - 1 - jsrc) - D - 1
    alpha_fit = _fit_alpha(mid_phasor, isrc, jsrc, n, dx, r0, rmax)

    info = {
        "steps_run": step,
        "stopped": stopped,
        "dt_s": dt,
        "dx_m": dx,
        "cells_per_lambda": cells_per_lambda,
        "alpha_theory": mps[src_medium]["alpha"],
        "decay_measured": alpha_fit,
        "n": n,
        "f_hz": f_hz,
        "medium": src_medium if not interface else "air/water",
        "interface": bool(interface),
        "lambda_src_m": lam_src,
        "span_m": n * dx,
        "src_index": [isrc, jsrc, ksrc],
        "sponge_cells": SPONGE_CELLS,
        "period_steps": period_steps,
        "frames_emitted": len(frames),
        "resolution_warning": resolution_warning,
    }
    if interface:
        info["alpha_theory_water"] = mps["water"]["alpha"]
        info["z_interface"] = zmid
    if error is not None:
        info["error"] = error
    return {"info": info, "frames": frames,
            "final_frame": frames[-1] if frames else None}


def _finite_lists(arr):
    """Nested float lists with non-finite values mapped to None."""
    def rec(x):
        if isinstance(x, list):
            return [rec(y) for y in x]
        return x if math.isfinite(x) else None
    return rec(np.asarray(arr, dtype=np.float64).tolist())


def _jsonify(v):
    if isinstance(v, np.ndarray):
        if v.dtype.kind in ("U", "S"):
            return v.tolist()
        return _finite_lists(v)
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def fdtd_slices_to_lists(frame):
    """JSON-safe copy of an fdtd_run frame (arrays → nested lists,
    ±inf/NaN → None, string arrays → lists of str)."""
    out = {}
    for k, v in frame.items():
        if k == "stokes" and isinstance(v, dict):
            out[k] = {sk: _jsonify(sv) for sk, sv in v.items()}
        else:
            out[k] = _jsonify(v)
    return out


# --- self-check --------------------------------------------------------------

if __name__ == "__main__":
    import json
    import threading
    import time

    # 150 MHz: fresh-water α ≈ ση/2 ≈ 1.05 Np/m is essentially f-independent
    # in this low-loss regime, so a low f makes the box physically larger and
    # the decay fit window spans a useful number of Nepers (see docstring).
    f = 150e6
    TOL = 0.35
    t0 = time.monotonic()
    mp_air = medium_params(f, "air")
    mp_w = medium_params(f, "water")

    # --- air: ballistic spread, α_fit ≈ 0, bounded energy -------------------
    air_frames = []
    res_air = fdtd_run(f, "air", n=40, frame_every=10, max_steps=300,
                       callback=air_frames.append)
    ia = res_air["info"]
    assert ia["resolution_warning"] is None
    assert len(air_frames) >= 25, len(air_frames)
    emaxes = np.array([fr["emax"] for fr in air_frames])
    assert np.all(np.isfinite(emaxes)), "NaN/Inf in air emax"
    assert emaxes[-5:].max() < 3.0 * emaxes[10:15].max(), \
        f"air energy blowing up: {emaxes[-5:].max()} vs {emaxes[10:15].max()}"
    a_air = ia["decay_measured"]
    assert a_air is not None
    assert abs(a_air) < 0.1 * mp_w["alpha"], (a_air, mp_w["alpha"])
    st = air_frames[-1]["stokes"]
    assert st["I"].shape == (40, 40) and st["handedness"].shape == (40, 40)
    json.dumps(fdtd_slices_to_lists(air_frames[-1]))   # JSON-safe helper
    print(f"PASS  air @{f/1e6:.0f} MHz, n=40: {len(air_frames)} frames, "
          f"emax bounded ({emaxes.max():.3g}), α_fit = {a_air:+.4f} Np/m ≈ 0 "
          f"(|·| < 0.1·α_water = {0.1*mp_w['alpha']:.4f}); stokes + JSON OK")

    # --- water: α_fit within ±35 % of theory ---------------------------------
    res_w = fdtd_run(f, "water", n=44, frame_every=10, max_steps=300,
                     callback=lambda fr: None, pol_viz=False)
    iw = res_w["info"]
    a_w = iw["decay_measured"]
    assert a_w is not None and a_w > 0.0, a_w
    assert abs(a_w - mp_w["alpha"]) / mp_w["alpha"] < TOL, \
        (a_w, mp_w["alpha"])
    print(f"PASS  water @{f/1e6:.0f} MHz, n=44: α_fit = {a_w:.4f} Np/m vs "
          f"theory {mp_w['alpha']:.4f} Np/m "
          f"({100*(a_w/mp_w['alpha'] - 1):+.1f} %, tolerance ±{TOL*100:.0f} %)")

    # --- interface: water half damps harder than air half --------------------
    iframes = []
    res_i = fdtd_run(f, interface=True, n=44, frame_every=10, max_steps=300,
                     callback=iframes.append, pol_viz=False)
    last = iframes[-1]
    assert last["e_rms_lo"] < last["e_rms_hi"], \
        (last["e_rms_lo"], last["e_rms_hi"])
    print(f"PASS  interface @{f/1e6:.0f} MHz, n=44: e_rms water half "
          f"{last['e_rms_lo']:.3g} < air half {last['e_rms_hi']:.3g} "
          f"(α_water = {res_i['info']['alpha_theory_water']:.3g} Np/m)")

    # --- stop_flag: counting callback aborts the run early --------------------
    flag = threading.Event()
    seen = {"n": 0}

    def cb(fr):
        seen["n"] += 1
        if seen["n"] >= 3:
            flag.set()

    res_s = fdtd_run(f, "air", n=24, frame_every=5, max_steps=300,
                     callback=cb, stop_flag=flag, pol_viz=False)
    assert res_s["info"]["stopped"] and res_s["info"]["steps_run"] < 300
    print(f"PASS  stop_flag: halted at step {res_s['info']['steps_run']}/300 "
          f"after {seen['n']} frames, stopped=True")

    print(f"fdtd selftest: all checks passed "
          f"({time.monotonic() - t0:.1f} s)")
