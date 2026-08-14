"""Electromagnetics core for antenna design in lossy media.

Textbook electromagnetics (Balanis, *Antenna Theory*; Pozar, *Microwave
Engineering*) — no heuristics beyond the standard analytic design
equations, each of which is cited in the per-antenna ``notes`` string.

**Plane-wave propagation in a lossy dielectric.**  A uniform plane wave
in a homogeneous medium (ε = εᵣε₀, μ = μᵣμ₀, conductivity σ) has
propagation constant

    γ = α + jβ = √[jωμ(σ + jωε)],

with the general lossy-dielectric attenuation and phase constants

    α = ω√(με/2) · √[ √(1 + (σ/ωε)²) − 1 ],
    β = ω√(με/2) · √[ √(1 + (σ/ωε)²) + 1 ],

valid continuously from the lossless limit (σ → 0: α → 0,
β → ω√(με)) to the good-conductor limit (σ ≫ ωε: α ≈ β ≈ √(πfμσ),
skin depth δ = 1/α).  The intrinsic impedance is

    η = √[ jωμ / (σ + jωε) ]          (complex in lossy media),

the wavelength in the medium is λ = 2π/β, the phase velocity
v_p = ω/β, and the loss tangent tanδ = σ/(ωε).

**Link budget.**  Friis transmission in an attenuating medium: the
spreading factor uses the medium wavelength, the medium adds
e^(−2αr) on the power (field ∝ e^(−αr)):

    P_r/P_t = G_t G_r (λ/4πr)² · e^(−2αr),
    L_med(dB) = 20·α·r·log₁₀(e) = 8.6859·α·r.

**Canonical antennas** (`ANTENNA_TYPES`).  All dimensions key off the
medium wavelength λ = 2π/β, so a design in water shrinks automatically
relative to air.  Models implemented:

* **dipole** — thin half-wave dipole, sinusoidal current distribution.
  Pattern F(θ) = [cos(π/2·cosθ)/sinθ]²; thin-wire Z_in ≈ 73 + j42.5 Ω
  at exactly λ/2, brought to resonance (X ≈ 0, R ≈ 65–70 Ω) by ~5 %
  end-effect shortening → L = 0.475 λ.  Directivity 1.643 → 2.15 dBi;
  fractional BW ≈ 8 % (thin-wire rule, VSWR 2:1).
* **monopole** — λ/4 over infinite ground plane: image theory gives
  half the dipole resistance (36.5 + j21.25 Ω), twice the directivity
  (5.15 dBi), pattern = dipole restricted to the upper hemisphere.
* **loop** — uniform-current circular loop, exact pattern
  E_φ ∝ J₁(ka·sinθ) (Balanis ch. 5); small loop (ka < 0.5) reduces to
  sin²θ with R_r = 320π⁴(A/λ²)² (single turn) and a documented
  copper-loss efficiency; the 1-wavelength loop (C = λ) option uses the
  same J₁ pattern at ka = 1.
* **patch** — rectangular microstrip patch, transmission-line/cavity
  model.  W = (c/2f)√(2/(εᵣ+1)); ε_eff = (εᵣ+1)/2 + (εᵣ−1)/2·(1 +
  12h/W)^(−1/2); Hammerstad fringing ΔL = 0.412h·(ε_eff+0.3)(W/h+0.264)
  / ((ε_eff−0.258)(W/h+0.8)); L = c/(2f√ε_eff) − 2ΔL.  Pattern from the
  two-slot model (magnetic currents at the radiating edges, array
  factor cos[(kL_eff/2)cosθ] × slot element sin[(kh/2)sinθ]/[…]).
  Radiation Q_r = c√ε_eff/(4fh), dielectric Q_d = 1/tanδ, conductor
  Q_c = h√(πfμ₀σ_c); efficiency η = Q_t/Q_r and VSWR-2:1 fractional
  BW = (S−1)/(Q_t√S) with S = 2 (Pozar/Balanis cavity model).
* **helix** — axial mode (C ≈ λ, pitch 13°): Kraus gain
  D = 15(C/λ)²N(S/λ), HPBW ≈ 52°/[(C/λ)√(N·S/λ)], R_in ≈ 140·C/λ Ω,
  circular polarization along the axis; pattern a cosⁿθ endfire fit
  matched to the Kraus beamwidth.  (Normal-mode helix C ≪ λ noted, not
  modelled.)
* **yagi** — N-element Yagi-Uda: reflector 0.50λ, driven 0.475λ,
  directors 0.44λ on ~0.2λ spacing; gain from the standard
  NBS/Viezbicke gain-vs-element-count table (interpolated); pattern an
  endfire cardioid power pattern [(1+cosθ)/2]^p whose exponent is set
  so its exact directivity 4π/Ω_A = p+1 equals the tabulated gain.
* **slot** — half-wave slot in a ground plane, Babinet–Booker
  complement of the dipole: identical pattern shape (rotated
  polarization, bidirectional), Z_slot = η²/(4·Z_dipole).
* **pifa** — quarter-wave shorted patch: L ≈ λ₀/(4√ε_eff) with
  ε_eff ≈ (εᵣ+1)/2, shorting wall at one edge; short-monopole pattern
  over ground, gain ≈ 2 dBi, BW ~3 % (PCB-loss dominated).

**Polarization** (`stokes`).  From two orthogonal complex field
components (suppressed e^{jωt}, wave propagating toward +z):

    I = |E_x|² + |E_y|²,   Q = |E_x|² − |E_y|²,
    U = 2 Re(E_x*·E_y),    V = 2 Im(E_x*·E_y),

polarization-ellipse tilt tan 2τ = U/Q, ellipticity sin 2ε = V/I,
axial ratio AR = |cot ε| (0 dB = circular, ∞ = linear).  Handedness
follows the IEEE right-hand rule (thumb along +z propagation): with
the e^{jωt} convention used here, V < 0 ≣ RHCP, V > 0 ≣ LHCP.
"""

from __future__ import annotations

import math

import numpy as np

# --- physical constants (SI) ----------------------------------------------

C0 = 299_792_458.0            # speed of light in vacuum, m/s (exact)
MU0 = 4.0e-7 * math.pi        # vacuum permeability, H/m
EPS0 = 1.0 / (MU0 * C0 * C0)  # vacuum permittivity, F/m
ETA0 = math.sqrt(MU0 / EPS0)  # free-space intrinsic impedance, Ω ≈ 376.73

SIGMA_CU = 5.8e7              # copper conductivity, S/m (conductor Q)

# --- media table -----------------------------------------------------------

MEDIA: dict[str, dict[str, float]] = {
    "air":       {"eps_r": 1.0,  "mu_r": 1.0, "sigma": 0.0},
    "water":     {"eps_r": 80.0, "mu_r": 1.0, "sigma": 0.05},   # fresh water
    "water_sea": {"eps_r": 72.0, "mu_r": 1.0, "sigma": 4.0},    # seawater
}


def _medium_dict(medium) -> dict[str, float]:
    """Resolve a medium name or a custom {eps_r, mu_r, sigma} dict."""
    if isinstance(medium, str):
        if medium not in MEDIA:
            raise KeyError(f"unknown medium {medium!r}; known: {sorted(MEDIA)}")
        return MEDIA[medium]
    return dict(medium)


def medium_params(f_hz: float, medium="air") -> dict:
    """General lossy-dielectric propagation parameters at frequency `f_hz`.

    Returns eps_r, mu_r, sigma, alpha (Np/m), beta (rad/m), gamma
    (complex), wavelength (m, in medium), lambda0 (free-space), phase_velocity,
    eta (complex Ω), skin_depth (m, inf when σ ≈ 0), loss_tangent.
    """
    m = _medium_dict(medium)
    eps_r, mu_r, sigma = m["eps_r"], m["mu_r"], m["sigma"]
    omega = 2.0 * math.pi * f_hz
    eps = eps_r * EPS0
    mu = mu_r * MU0
    x = sigma / (omega * eps) if sigma > 0.0 else 0.0   # σ/ωε = loss tangent
    root = math.sqrt(1.0 + x * x)
    head = omega * math.sqrt(mu * eps / 2.0)
    alpha = head * math.sqrt(root - 1.0)
    beta = head * math.sqrt(root + 1.0)
    gamma = complex(alpha, beta)
    eta = np.sqrt(1j * omega * mu / complex(sigma, omega * eps))
    return {
        "eps_r": eps_r,
        "mu_r": mu_r,
        "sigma": sigma,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "wavelength": 2.0 * math.pi / beta,
        "lambda0": C0 / f_hz,
        "phase_velocity": omega / beta,
        "eta": complex(eta),
        "skin_depth": math.inf if alpha == 0.0 else 1.0 / alpha,
        "loss_tangent": x,
    }


# --- link budget -----------------------------------------------------------

def link_budget(p_tx_dbw: float, g_tx_dbi: float, g_rx_dbi: float,
                range_m: float, f_hz: float, medium: str = "air") -> dict:
    """Full Friis link budget in a (possibly lossy) medium.

    Spreading uses the medium wavelength; the medium attenuation adds
    L_med = 8.6859·α·r dB (power ∝ e^(−2αr)).
    """
    mp = medium_params(f_hz, medium)
    lam = mp["wavelength"]
    fspl_db = 20.0 * math.log10(4.0 * math.pi * range_m / lam)
    medium_loss_db = 20.0 * mp["alpha"] * range_m * math.log10(math.e)
    path_loss_db = fspl_db + medium_loss_db
    received_dbw = p_tx_dbw + g_tx_dbi + g_rx_dbi - path_loss_db
    return {
        "p_tx_dbw": p_tx_dbw,
        "g_tx_dbi": g_tx_dbi,
        "g_rx_dbi": g_rx_dbi,
        "range_m": range_m,
        "f_hz": f_hz,
        "medium": medium,
        "wavelength_m": lam,
        "alpha_np_per_m": mp["alpha"],
        "fspl_db": fspl_db,
        "medium_loss_db": medium_loss_db,
        "path_loss_db": path_loss_db,
        "received_dbw": received_dbw,
    }


def friis_received_dbw(p_tx_dbw: float, g_tx_dbi: float, g_rx_dbi: float,
                       range_m: float, f_hz: float, medium: str = "air") -> float:
    """Received power (dBW) from the Friis equation with medium attenuation."""
    return link_budget(p_tx_dbw, g_tx_dbi, g_rx_dbi, range_m, f_hz,
                       medium)["received_dbw"]


# --- pattern helpers -------------------------------------------------------

def _safe_div_pattern(num: np.ndarray, den: np.ndarray, fill: float = 0.0):
    """num/den with den → 0 mapped to `fill` (pattern nulls at sinθ = 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return np.where(np.abs(den) < 1e-14, fill, out)


def _j1(x):
    """Bessel J₁ by its power series — |x| ≤ 2 here (loop ka·sinθ), no scipy."""
    x = np.asarray(x, dtype=float)
    t = (x / 2.0) ** 2
    s = x / 2.0
    term = np.ones_like(x)
    for k in range(1, 12):           # terms (x/2)^{2k+1}/(k!(k+1)!)
        term = term * (-t) / (k * (k + 1))
        s = s + x / 2.0 * term
    return s


def _directivity(pattern, ground_plane: bool = False) -> float:
    """Directivity 4π/Ω_A of a normalized power pattern by grid integration."""
    th = np.linspace(1e-6, (0.5 if ground_plane else 1.0) * np.pi - 1e-6, 361)
    ph = np.linspace(0.0, 2.0 * np.pi, 361, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    F = np.asarray(pattern(TH, PH), dtype=float) * np.sin(TH)
    omega_a = np.trapezoid(np.trapezoid(F, th, axis=0), ph)
    return 4.0 * np.pi / omega_a


def _db(x: float) -> float:
    return 10.0 * math.log10(x)


# --- canonical antenna builders ---------------------------------------------
# Each builder takes (f_hz, medium, **opts) and returns a dict with keys
# type, dimensions_m, z_in_ohm, gain_dbi, bandwidth_frac, polarization,
# pattern (vectorized callable pattern(theta, phi) → 0..1), notes.


def build_dipole(f_hz: float, medium: str = "air", **opts) -> dict:
    """Thin half-wave dipole along ẑ: F(θ) = [cos(π/2 cosθ)/sinθ]²."""
    mp = medium_params(f_hz, medium)
    lam = mp["wavelength"]
    length = 0.475 * lam                    # ~5 % end-effect shortening
    wire_radius = opts.get("wire_radius", length / 500.0)

    def pattern(theta, phi):
        th = np.asarray(theta, dtype=float)
        f = _safe_div_pattern(np.cos(0.5 * np.pi * np.cos(th)), np.sin(th))
        return f * f

    return {
        "type": "dipole",
        "f_hz": f_hz,
        "medium": medium,
        "dimensions_m": {"length": length, "wire_radius": wire_radius},
        "z_in_ohm": complex(70.0, 0.0),
        "gain_dbi": 2.15,
        "bandwidth_frac": 0.08,
        "polarization": "linear",
        "pattern": pattern,
        "notes": (
            "Sinusoidal-current thin dipole (Balanis 4.5–4.6): "
            "F(θ) = [cos(π/2·cosθ)/sinθ]², D = 1.643 → 2.15 dBi. "
            "Exact-λ/2 thin wire has Z_in = 73 + j42.5 Ω; the 5 % "
            "end-effect shortening to L = 0.475λ resonates it "
            "(X ≈ 0, R ≈ 65–70 Ω, here 70 Ω). Fractional BW ≈ 8 % "
            "thin-wire rule (VSWR 2:1). L keys off the medium "
            f"wavelength λ = 2π/β = {lam:.4g} m."
        ),
    }


def build_monopole(f_hz: float, medium: str = "air", **opts) -> dict:
    """Quarter-wave monopole over ground plane (image theory)."""
    mp = medium_params(f_hz, medium)
    lam = mp["wavelength"]
    length = 0.25 * lam
    ground_plane = opts.get("ground_plane", True)
    gp_note = ("infinite ground plane assumed" if ground_plane else
               "WARNING: no ground plane requested — image-theory values "
               "(gain, Z) are optimistic")

    def pattern(theta, phi):
        th = np.asarray(theta, dtype=float)
        f = _safe_div_pattern(np.cos(0.5 * np.pi * np.cos(th)), np.sin(th))
        return np.where(th <= 0.5 * np.pi + 1e-12, f * f, 0.0)

    return {
        "type": "monopole",
        "f_hz": f_hz,
        "medium": medium,
        "dimensions_m": {"length": length,
                         "wire_radius": opts.get("wire_radius", length / 500.0)},
        "z_in_ohm": complex(36.5, 21.25),
        "gain_dbi": 5.15,
        "bandwidth_frac": 0.08,
        "polarization": "linear",
        "pattern": pattern,
        "notes": (
            "λ/4 monopole over ground (Balanis 4.9, image theory): pattern = "
            "dipole pattern restricted to the upper hemisphere, R_r half of "
            "the dipole (73/2 = 36.5 Ω, X = 42.5/2 = 21.25 Ω — resonant at "
            "≈0.24λ), directivity doubled → 2.15 + 3.01 = 5.15 dBi. "
            f"{gp_note}. λ_medium = {lam:.4g} m."
        ),
    }


def build_loop(f_hz: float, medium: str = "air", **opts) -> dict:
    """Uniform-current circular loop; exact pattern E_φ ∝ J₁(ka sinθ).

    opts: mode "small" (ka = 0.3 default, opt ka) or "resonant" (C = λ);
    wire_radius default 1 mm (copper, for the small-loop loss efficiency).
    """
    mp = medium_params(f_hz, medium)
    lam = mp["wavelength"]
    mode = opts.get("mode", "small")
    b = opts.get("wire_radius", 1e-3)
    if mode == "resonant":
        a = lam / (2.0 * math.pi)                 # circumference C = λ, ka = 1
    else:
        ka = float(opts.get("ka", 0.3))
        if ka >= 0.5:
            raise ValueError("small-loop mode requires ka < 0.5")
        a = ka * lam / (2.0 * math.pi)
    ka = 2.0 * math.pi * a / lam
    area = math.pi * a * a

    def raw(theta, phi):
        th = np.asarray(theta, dtype=float)
        j = _j1(ka * np.sin(th))
        return j * j

    # normalize against the pattern's own max (J₁ peaks at ka sinθ = 1.841)
    th_g = np.linspace(0.0, np.pi, 721)
    fmax = float(np.max(_j1(ka * np.sin(th_g)) ** 2))
    pattern = lambda theta, phi: raw(theta, phi) / fmax   # noqa: E731

    if mode == "resonant":
        z_in = complex(100.0, 100.0)
        gain_dbi = _db(_directivity(pattern))
        eff_note = "lossless (radiation-resistance dominated at ka = 1)"
        rr = None
        bw = 0.10
    else:
        rr = 320.0 * math.pi**4 * (area / lam**2) ** 2   # Balanis 5-24, 1 turn
        # copper loss: R_loss = (C/2πb)·R_s, R_s = √(πfμ₀/σ_Cu)
        r_s = math.sqrt(math.pi * f_hz * MU0 / SIGMA_CU)
        r_loss = (2.0 * math.pi * a) / (2.0 * math.pi * b) * r_s
        eff = rr / (rr + r_loss)
        # inductive loop reactance: L = μ₀a[ln(8a/b) − 2] (Balanis 5-36)
        ind = MU0 * a * (math.log(8.0 * a / b) - 2.0)
        z_in = complex(rr + r_loss, 2.0 * math.pi * f_hz * ind)
        gain_dbi = _db(1.5 * eff)                # small-loop D = 1.5 → 1.76 dBi
        eff_note = f"η = R_r/(R_r+R_loss) = {eff:.3g}"
        bw = 0.01

    return {
        "type": "loop",
        "f_hz": f_hz,
        "medium": medium,
        "dimensions_m": {"radius": a, "circumference": 2.0 * math.pi * a,
                         "wire_radius": b},
        "z_in_ohm": z_in,
        "gain_dbi": gain_dbi,
        "bandwidth_frac": bw,
        "polarization": "linear",
        "pattern": pattern,
        "notes": (
            f"Uniform-current circular loop, mode={mode} (ka = {ka:.3g}). "
            "Exact far field E_φ ∝ J₁(ka·sinθ) (Balanis 5.3); small loop "
            "reduces to sin²θ with R_r = 320π⁴(A/λ²)² "
            + (f"= {rr:.3g} Ω. " if rr is not None else "")
            + eff_note + ". Resonant loop (C = λ): Z_in ≈ 100 + j100 Ω, "
            "directivity from grid integration of the uniform-current J₁ "
            "pattern — the uniform-current model underestimates the true "
            "resonant loop (near-sinusoidal current, D ≈ 2.2 ≣ 3.4 dBi). "
            f"λ_medium = {lam:.4g} m."
        ),
    }


def build_patch(f_hz: float, medium: str = "air", **opts) -> dict:
    """Rectangular microstrip patch — transmission-line / cavity model.

    opts: eps_r (substrate, default 4.4 FR4), h (default 1.6 mm),
    tan_delta (default 0.02 FR4), sigma_c (copper).
    """
    mp = medium_params(f_hz, medium)
    lam = mp["wavelength"]
    lam0 = mp["lambda0"]
    er = float(opts.get("eps_r", 4.4))
    h = float(opts.get("h", 1.6e-3))
    tan_d = float(opts.get("tan_delta", 0.02))
    sig_c = float(opts.get("sigma_c", SIGMA_CU))

    w = lam0 / 2.0 * math.sqrt(2.0 / (er + 1.0))          # Balanis 14-6
    e_eff = ((er + 1.0) / 2.0
             + (er - 1.0) / 2.0 * (1.0 + 12.0 * h / w) ** -0.5)   # 14-1
    dl = (0.412 * h * (e_eff + 0.3) * (w / h + 0.264)
          / ((e_eff - 0.258) * (w / h + 0.8)))                     # 14-2
    l_eff = lam0 / (2.0 * math.sqrt(e_eff))
    length = l_eff - 2.0 * dl                                      # 14-7

    k0 = 2.0 * math.pi / lam0

    def pattern(theta, phi):
        th = np.asarray(theta, dtype=float)
        ph = np.asarray(phi, dtype=float)
        x = 0.5 * k0 * h * np.sin(th)
        z = 0.5 * k0 * l_eff * np.cos(th)
        sx = np.where(np.abs(x) < 1e-9, 1.0, np.sin(x) / np.where(x == 0, 1, x))
        af = np.cos(z)                       # two-slot array factor
        e_th = np.sin(ph) * sx * af
        e_ph = np.cos(ph) * np.cos(th) * sx * af
        f = np.where(th <= 0.5 * np.pi + 1e-12, e_th * e_th + e_ph * e_ph, 0.0)
        return f

    th_g = np.linspace(0.0, 0.5 * np.pi, 361)
    ph_g = np.linspace(0.0, 2.0 * np.pi, 361, endpoint=False)
    TH, PH = np.meshgrid(th_g, ph_g, indexing="ij")
    fmax = float(np.max(pattern(TH, PH)))
    raw = pattern
    # clip: the normalization grid can undershoot the true max (≈ 0.002 %
    # at the E-plane horizon) — the 0..1 pattern contract is exact.
    pattern = lambda theta, phi: np.clip(raw(theta, phi) / fmax, 0.0, 1.0)  # noqa: E731
    directivity = _directivity(pattern, ground_plane=True)

    # cavity-model Qs (Balanis 14.3 / Pozar): radiation, dielectric, conductor
    q_r = C0 * math.sqrt(e_eff) / (4.0 * f_hz * h)
    q_d = 1.0 / tan_d
    q_c = h * math.sqrt(math.pi * f_hz * MU0 * sig_c)
    q_t = 1.0 / (1.0 / q_r + 1.0 / q_d + 1.0 / q_c)
    efficiency = q_t / q_r
    gain_dbi = _db(directivity * efficiency)
    s_vswr = 2.0
    bw = (s_vswr - 1.0) / (q_t * math.sqrt(s_vswr))

    # resonant edge resistance from the slot conductance G₁ = W²/(90λ₀²)
    g_slot = w * w / (90.0 * lam0 * lam0)
    r_in = 1.0 / (2.0 * g_slot)

    return {
        "type": "patch",
        "f_hz": f_hz,
        "medium": medium,
        "dimensions_m": {"W": w, "L": length, "L_eff": l_eff, "h": h,
                         "delta_L": dl},
        "z_in_ohm": complex(r_in, 0.0),
        "gain_dbi": gain_dbi,
        "directivity_dbi": _db(directivity),
        "efficiency": efficiency,
        "bandwidth_frac": bw,
        "polarization": "linear",
        "pattern": pattern,
        "notes": (
            "Transmission-line/cavity model (Balanis ch. 14): "
            f"W = (c/2f)√(2/(εᵣ+1)) = {w*1e3:.2f} mm; "
            f"ε_eff = (εᵣ+1)/2 + (εᵣ−1)/2·(1+12h/W)^(−1/2) = {e_eff:.3f}; "
            f"ΔL (Hammerstad) = {dl*1e3:.3f} mm; L = c/(2f√ε_eff) − 2ΔL "
            f"= {length*1e3:.2f} mm. Pattern: two-slot model, "
            "E ∝ sin(½k₀h sinθ)/(½k₀h sinθ)·cos(½k₀L_eff cosθ), directivity "
            f"{_db(directivity):.2f} dBi by grid integration. Q_r = "
            f"c√ε_eff/(4fh) = {q_r:.1f}, Q_d = 1/tanδ = {q_d:.1f}, Q_c = "
            f"h√(πfμ₀σ_c) = {q_c:.0f} → η = Q_t/Q_r = {efficiency:.3f}, "
            f"BW(VSWR 2:1) = 1/(√2·Q_t) = {bw*100:.2f} %. Edge resistance "
            f"R_in = 1/(2G₁), G₁ = W²/(90λ₀²) → {r_in:.0f} Ω "
            "(inset feed to 50 Ω in practice)."
        ),
    }


def build_helix(f_hz: float, medium: str = "air", **opts) -> dict:
    """Axial-mode helix (Kraus): C ≈ λ, 13° pitch, circular polarization.

    opts: turns N (default 6), pitch_deg (default 13°).
    """
    mp = medium_params(f_hz, medium)
    lam = mp["wavelength"]
    n_turns = int(opts.get("turns", 6))
    pitch = math.radians(float(opts.get("pitch_deg", 13.0)))
    circ = lam
    spacing = circ * math.tan(pitch)
    gain_lin = 15.0 * (circ / lam) ** 2 * n_turns * (spacing / lam)   # Kraus
    gain_dbi = _db(gain_lin)
    hpbw_deg = 52.0 / ((circ / lam) * math.sqrt(n_turns * spacing / lam))
    # cosⁿθ endfire fit: cosⁿ(HPBW/2) = 1/2
    n_exp = math.log(0.5) / math.log(math.cos(math.radians(hpbw_deg) / 2.0))
    radius = circ / (2.0 * math.pi)

    def pattern(theta, phi):
        th = np.asarray(theta, dtype=float)
        return np.clip(np.cos(th), 0.0, None) ** n_exp

    return {
        "type": "helix",
        "f_hz": f_hz,
        "medium": medium,
        "dimensions_m": {"circumference": circ, "radius": radius,
                         "turn_spacing": spacing, "turns": n_turns,
                         "axial_length": n_turns * spacing},
        "z_in_ohm": complex(140.0 * circ / lam, 0.0),
        "gain_dbi": gain_dbi,
        "bandwidth_frac": 0.20,
        "polarization": "circular",
        "pattern": pattern,
        "notes": (
            "Axial-mode helix, Kraus design equations: C ≈ λ, pitch 13°, "
            "D = 15(C/λ)²·N·(S/λ) = "
            f"{gain_lin:.1f} ({gain_dbi:.2f} dBi, N = {n_turns}); HPBW ≈ "
            f"52°/[(C/λ)√(N·S/λ)] = {hpbw_deg:.1f}°, pattern cos^nθ with "
            f"n = {n_exp:.2f} matched to that beamwidth; R_in ≈ 140·C/λ Ω. "
            "Endfire circular polarization along the axis (handedness = "
            "winding sense). Normal-mode helix (C ≪ λ) not modelled. "
            f"λ_medium = {lam:.4g} m."
        ),
    }


# Standard NBS / Viezbicke Yagi gain table (element count → dBi), the
# documented gain-vs-element-count reference; interpolated between points.
_YAGI_GAIN_TABLE = {3: 7.5, 4: 9.0, 5: 10.2, 6: 11.0, 8: 12.5,
                    10: 13.5, 12: 14.2, 15: 15.4}


def build_yagi(f_hz: float, medium: str = "air", **opts) -> dict:
    """N-element Yagi-Uda array, endfire along its boom (θ = 0)."""
    mp = medium_params(f_hz, medium)
    lam = mp["wavelength"]
    n_el = int(opts.get("elements", 5))
    if n_el < 3:
        raise ValueError("yagi needs at least 3 elements")
    ks = sorted(_YAGI_GAIN_TABLE)
    gain_dbi = float(np.interp(n_el, ks, [_YAGI_GAIN_TABLE[k] for k in ks]))
    # endfire cardioid power pattern [(1+cosθ)/2]^p: Ω_A = 4π/(p+1), so
    # p = D − 1 makes its exact directivity equal the tabulated gain.
    p = 10.0 ** (gain_dbi / 10.0) - 1.0
    spacings = {"reflector_driven": 0.25 * lam, "driven_director": 0.2 * lam}

    def pattern(theta, phi):
        th = np.asarray(theta, dtype=float)
        return ((1.0 + np.cos(th)) / 2.0) ** p

    return {
        "type": "yagi",
        "f_hz": f_hz,
        "medium": medium,
        "dimensions_m": {
            "reflector": 0.50 * lam,
            "driven": 0.475 * lam,
            "director": 0.44 * lam,
            "elements": n_el,
            "boom_length": spacings["reflector_driven"]
            + (n_el - 2) * spacings["driven_director"],
        },
        "z_in_ohm": complex(25.0, 0.0),
        "gain_dbi": gain_dbi,
        "bandwidth_frac": 0.03,
        "polarization": "linear",
        "pattern": pattern,
        "notes": (
            f"{n_el}-element Yagi-Uda: reflector 0.50λ, driven dipole "
            "0.475λ, directors 0.44λ, spacings 0.25λ (refl–driven) / 0.2λ. "
            f"Gain {gain_dbi:.1f} dBi from the standard NBS/Viezbicke "
            "gain-vs-element-count table (interpolated). Pattern: endfire "
            f"[(1+cosθ)/2]^p with p = {p:.1f} chosen so its directivity "
            "p+1 equals the tabulated gain. Driven-element Z drops to "
            "≈ 20–30 Ω in the array (folded dipole match in practice). "
            f"λ_medium = {lam:.4g} m."
        ),
    }


def build_slot(f_hz: float, medium: str = "air", **opts) -> dict:
    """Half-wave slot in a ground plane — Babinet–Booker complement of the
    dipole: same pattern shape (bidirectional, orthogonal polarization),
    Z_slot·Z_dipole = η²/4."""
    mp = medium_params(f_hz, medium)
    lam = mp["wavelength"]
    eta = mp["eta"]
    length = 0.475 * lam
    width = opts.get("width", length / 20.0)
    z_dipole = complex(73.0, 0.0)           # resonant (shortened) thin dipole
    z_slot = eta * eta / (4.0 * z_dipole)   # Booker relation

    def pattern(theta, phi):
        th = np.asarray(theta, dtype=float)
        f = _safe_div_pattern(np.cos(0.5 * np.pi * np.cos(th)), np.sin(th))
        return f * f

    return {
        "type": "slot",
        "f_hz": f_hz,
        "medium": medium,
        "dimensions_m": {"length": length, "width": width},
        "z_in_ohm": complex(z_slot),
        "gain_dbi": 2.15,
        "bandwidth_frac": 0.08,
        "polarization": "linear",
        "pattern": pattern,
        "notes": (
            "Half-wave slot, Babinet–Booker complement of the dipole "
            "(Balanis 12.8): identical pattern shape, radiating both sides "
            "of the screen, E/H (polarization) interchanged. Booker: "
            f"Z_slot = η²/(4·Z_dipole) with the resonant complement "
            f"Z_dipole = 73 + j0 Ω (the unshortened thin dipole is "
            f"73 + j42.5 Ω) and η = {abs(eta):.3g} Ω → Z_slot = "
            f"{z_slot.real:.1f} {z_slot.imag:+.1f}j Ω."
        ),
    }


def build_pifa(f_hz: float, medium: str = "air", **opts) -> dict:
    """Planar inverted-F / meander PCB antenna: quarter-wave shorted patch.

    opts: eps_r (default 4.4), h (default 1.6 mm).
    """
    mp = medium_params(f_hz, medium)
    lam = mp["wavelength"]
    lam0 = mp["lambda0"]
    er = float(opts.get("eps_r", 4.4))
    h = float(opts.get("h", 1.6e-3))
    e_eff = 0.5 * (er + 1.0)                # quarter-wave patch estimate
    length = lam0 / (4.0 * math.sqrt(e_eff))
    width = length                          # square plate default

    def pattern(theta, phi):
        th = np.asarray(theta, dtype=float)
        s = np.sin(th)
        return np.where(th <= 0.5 * np.pi + 1e-12, s * s, 0.0)

    return {
        "type": "pifa",
        "f_hz": f_hz,
        "medium": medium,
        "dimensions_m": {"length": length, "width": width, "h": h},
        "z_in_ohm": complex(50.0, 0.0),
        "gain_dbi": 2.0,
        "bandwidth_frac": 0.03,
        "polarization": "linear",
        "pattern": pattern,
        "notes": (
            "PIFA: quarter-wave patch with a shorting wall at one edge "
            "(Balanis 14.8 style): L ≈ λ₀/(4√ε_eff), ε_eff ≈ (εᵣ+1)/2 = "
            f"{e_eff:.2f} → L = {length*1e3:.2f} mm. Short-monopole sin²θ "
            "pattern over the ground plane; gain ≈ 1–3 dBi (2 dBi taken, "
            "PCB-loss dominated), BW ~2–5 % (3 % taken); 50 Ω via feed-pin "
            "placement between the shorted and open edges."
        ),
    }


ANTENNA_TYPES: dict[str, callable] = {
    "dipole": build_dipole,
    "monopole": build_monopole,
    "loop": build_loop,
    "patch": build_patch,
    "helix": build_helix,
    "yagi": build_yagi,
    "slot": build_slot,
    "pifa": build_pifa,
    "meander": build_pifa,          # meander PCB antenna ≈ PIFA model
}


# --- polarization ------------------------------------------------------------

def stokes(ex, ey) -> dict:
    """Stokes parameters and polarization-ellipse data from two orthogonal
    complex field components (arrays OK; suppressed e^{jωt}, propagation +z).

    I = |Ex|²+|Ey|², Q = |Ex|²−|Ey|², U = 2Re(Ex*Ey), V = 2Im(Ex*Ey).
    Tilt tan2τ = U/Q; ellipticity sin2ε = V/I; axial ratio = |cot ε| in dB.
    Handedness (IEEE right-hand rule, thumb along +z): V < 0 → RHCP,
    V > 0 → LHCP; "linear" when |V| ≪ I (axial ratio > 15 dB).
    """
    ex = np.asarray(ex, dtype=complex)
    ey = np.asarray(ey, dtype=complex)
    i_s = np.abs(ex) ** 2 + np.abs(ey) ** 2
    q_s = np.abs(ex) ** 2 - np.abs(ey) ** 2
    cross = np.conj(ex) * ey
    u_s = 2.0 * np.real(cross)
    v_s = 2.0 * np.imag(cross)

    with np.errstate(invalid="ignore", divide="ignore"):
        sin2e = np.where(i_s > 0.0, np.clip(v_s / i_s, -1.0, 1.0), 0.0)
    eps_ell = 0.5 * np.arcsin(sin2e)
    with np.errstate(divide="ignore", invalid="ignore"):
        ar = np.abs(1.0 / np.tan(eps_ell))
    ar = np.where(np.abs(eps_ell) < 1e-12, np.inf, ar)
    axial_ratio_db = 20.0 * np.log10(ar)
    tilt_deg = 0.5 * np.degrees(np.arctan2(u_s, q_s))

    handedness = np.where(axial_ratio_db > 15.0, "linear",
                          np.where(v_s < 0.0, "RHCP", "LHCP"))

    out = {
        "I": i_s, "Q": q_s, "U": u_s, "V": v_s,
        "axial_ratio_db": axial_ratio_db,
        "tilt_deg": tilt_deg,
        "handedness": handedness,
    }
    if np.ndim(i_s) == 0:
        out = {k: (v.item() if isinstance(v, np.ndarray) and v.ndim == 0 else v)
               for k, v in out.items()}
        out["handedness"] = str(handedness.item()) if np.ndim(handedness) == 0 \
            else handedness
    return out


# --- self-check --------------------------------------------------------------

if __name__ == "__main__":
    # --- air propagation at 2.45 GHz ---
    mp = medium_params(2.45e9, "air")
    lam_air = mp["wavelength"]
    assert abs(lam_air - 0.1224) / 0.1224 < 0.01, lam_air
    assert mp["alpha"] == 0.0 and mp["skin_depth"] == math.inf
    assert abs(ETA0 - 376.730313461) < 1e-6, ETA0
    print(f"PASS  air @2.45 GHz: λ = {lam_air:.5f} m ≈ 0.1224 m, α = 0, "
          f"η₀ = {ETA0:.6f} Ω")

    # --- fresh water: dispersion shrinks λ by ≈ √80, σ gives α > 0 ---
    mw = medium_params(2.45e9, "water")
    shrink = lam_air / mw["wavelength"]
    assert 8.5 < shrink < 9.4, shrink          # √80 = 8.944
    assert mw["alpha"] > 0.0
    print(f"PASS  water @2.45 GHz: λ shrink ×{shrink:.3f} ≈ √80, "
          f"α = {mw['alpha']:.4g} Np/m > 0")

    # --- seawater @ 100 kHz: good-conductor skin depth ---
    ms = medium_params(1e5, "water_sea")
    delta_gc = 1.0 / math.sqrt(math.pi * 1e5 * MU0 * ms["sigma"])
    assert abs(ms["skin_depth"] - delta_gc) / delta_gc < 0.05, ms["skin_depth"]
    print(f"PASS  seawater @100 kHz: δ = {ms['skin_depth']:.4f} m vs "
          f"good-conductor {delta_gc:.4f} m")

    # --- dipole ---
    d = build_dipole(2.45e9)
    assert d["gain_dbi"] == 2.15
    th = np.linspace(1e-6, np.pi - 1e-6, 2001)
    f = d["pattern"](th, 0.0)
    assert d["pattern"](0.0, 0.0) == 0.0                      # null on axis
    assert abs(th[np.argmax(f)] - np.pi / 2) < 0.01           # max broadside
    assert abs(np.max(f) - 1.0) < 1e-9
    print(f"PASS  dipole: null at θ=0, max at θ=π/2, gain = 2.15 dBi, "
          f"L = {d['dimensions_m']['length']*1e3:.2f} mm")

    # --- patch on FR4 at 2.45 GHz ---
    p = build_patch(2.45e9, "air", eps_r=4.4, h=1.6e-3)
    w_mm, l_mm = p["dimensions_m"]["W"] * 1e3, p["dimensions_m"]["L"] * 1e3
    assert abs(w_mm - 37.3) / 37.3 < 0.02, w_mm
    assert 28.0 <= l_mm <= 30.0, l_mm
    assert l_mm < w_mm
    print(f"PASS  patch FR4 @2.45 GHz: W = {w_mm:.2f} mm ≈ 37.3 mm, "
          f"L = {l_mm:.2f} mm < W (gain {p['gain_dbi']:.2f} dBi, "
          f"BW {p['bandwidth_frac']*100:.2f} %)")

    # --- slot: Booker relation ---
    s = build_slot(2.45e9)
    z_ref = ETA0 * ETA0 / (4.0 * 73.0)
    assert abs(abs(s["z_in_ohm"]) - z_ref) / z_ref < 0.05, s["z_in_ohm"]
    print(f"PASS  slot: |Z_slot| = {abs(s['z_in_ohm']):.1f} Ω ≈ η₀²/(4·73) "
          f"= {z_ref:.1f} Ω")

    # --- stokes of a pure circular wave ---
    st = stokes(1.0 / math.sqrt(2.0), 1j / math.sqrt(2.0))
    assert abs(abs(st["V"]) - st["I"]) < 1e-12, st
    assert abs(st["axial_ratio_db"]) < 1e-9, st["axial_ratio_db"]
    assert st["handedness"] in ("RHCP", "LHCP")
    stl = stokes(1.0, 0.0)
    assert stl["handedness"] == "linear" and stl["axial_ratio_db"] == np.inf
    print(f"PASS  stokes (1, j)/√2: |V| = I = {st['I']:.3f}, "
          f"AR = {st['axial_ratio_db']:.2e} dB ({st['handedness']}); "
          f"(1, 0) → linear")

    # --- Friis: two dipoles, 1 km, 2.45 GHz, air → FSPL ≈ 100.2 dB ---
    lb = link_budget(0.0, 2.15, 2.15, 1000.0, 2.45e9)
    assert abs(lb["fspl_db"] - 100.2) < 0.5, lb["fspl_db"]
    assert abs(friis_received_dbw(0.0, 2.15, 2.15, 1000.0, 2.45e9)
               - (4.3 - lb["path_loss_db"])) < 1e-12
    print(f"PASS  friis 1 km @2.45 GHz: FSPL = {lb['fspl_db']:.2f} dB ≈ "
          f"100.2 dB, P_r = {lb['received_dbw']:.2f} dBW")

    # --- every registered builder produces a valid record ---
    for name, builder in ANTENNA_TYPES.items():
        ant = builder(2.45e9)
        assert {"type", "dimensions_m", "z_in_ohm", "gain_dbi",
                "bandwidth_frac", "polarization", "pattern",
                "notes"} <= set(ant), name
        assert 0.0 <= float(ant["pattern"](np.pi / 2, 0.0)) <= 1.0 + 1e-9
    print(f"PASS  ANTENNA_TYPES registry: {sorted(ANTENNA_TYPES)} all build")
    print("em_physics selftest: all checks passed")
