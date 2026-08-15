"""PCB RF-filter synthesis, distributed S-parameter evaluation, and SA.

Planar laminate filters (the Marki / Mini-Circuits / everythingRF digest
space: Butterworth & Chebyshev prototypes, resonator Q and order, 50 Ω
matching, RL ≥ 10 dB, 40 dBc at 10 % from the band edge, stripline vs
CPW launch).  Source: *RF Filter Digest* (everythingRF, 2025)
https://cdn.everythingrf.com/live/Filters_updated_version_638463728945358500_1_3_638761394513756674.pdf

**Prototypes.**  Low-pass g-values (Pozar 8.3).  Butterworth (maximally
flat):

    g₀ = 1,   g_k = 2 sin((2k−1)π / 2N),   g_{N+1} = 1.

Chebyshev (ripple L_r dB): β = ln[coth(L_r / 17.37)], γ = sinh(β/2N),
a_k = sin((2k−1)π/2N), b_k = γ² + sin²(kπ/N),

    g₁ = 2 a₁/γ,   g_k = 4 a_{k−1} a_k / (b_{k−1} g_{k−1}),
    g_{N+1} = 1 (N odd) or coth²(β/4) (N even).

Roll-off of an Nth-order Butterworth is −20 N dB/decade (digest).

**Distributed PCB realisations** (Hong & Lancaster, *Microstrip Filters
for RF/Microwave Applications*; Pozar ch. 8):

* **LPF** — stepped-impedance: series g_k → high-Z line
  ℓ = (λ_g/2π)·(g_k Z₀/Z_H); shunt g_k → low-Z line
  ℓ = (λ_g/2π)·(g_k Z_L/Z₀).  Z_H ≈ 100 Ω, Z_L ≈ 20 Ω.
* **HPF** — shunt shorted stubs of length λ_g/4 at f_c, spaced by
  unit elements.  Zin = j Z tanθ is inductive below f_c and open at
  f_c, so the cascade is high-pass with a spurious notch at 2 f_c.
* **BPF** — J-inverter cascade of λ_g/2 resonators
  J_{0,1}/Y₀ = √(π Δ / 2 g₀ g₁),  J_{i,i+1}/Y₀ = π Δ / 2√(g_i g_{i+1})
  (Pozar 8.7).  Laid out as hairpin (folded λ/2) U-resonators.
* **BSF** — open λ_g/4 stubs on a 50 Ω through-line; stub admittance
  from the bandstop prototype Y_k/Y₀ = (4 g_k / π) Δ.

**S-parameters.**  Each section is an ABCD matrix of a lossy TEM line
(θ = βℓ − jαℓ, α from dielectric tanδ + copper Rs) or a shunt stub /
series gap / J-inverter.  Cascade → S via the bilinear map.  Insertion
loss at band centre is compared to the digest formula

    IL(dB) ≈ 4.343 · (Σ g_k) / (Δ · Q_u)     (Marki eq. 2),

with Q_u = β/(2α) of the 50 Ω microstrip.

**Evolution** (`filter_sa`).  Hadamard-seeded multiplicative
perturbations of section lengths and widths/gaps; length-preserving
corner of the design is *not* required (electrical length *is* the
knob).  Energy: passband IL, passband RL vs 10 dB, stopband rejection
vs 40 dBc, DFM (5 mil), size.

**Topologies.**  `design_filter(kind, …, topo=…)` selects the
realisation per kind (`TOPOS`); the default is always the distributed
microstrip form above, so existing calls are unchanged:

* **lc** — lumped LC ladder from the same g-values (Pozar 8.4: LP→HP
  dual, LP→BP / LP→BS reactance transforms, f₀ = √(f_lo·f_hi)).
* **qw_tl** — commensurate quarter-wave TEM filter: Richards'
  transform + Kuroda identities (LPF all-shunt open stubs / BPF
  shorted-stub resonators) separated by λ/4 unit elements.
* **dc_lc** — capacitively-coupled shunt-LC resonator BPF (nodal
  J-inverter ladder, Pozar 8.7 with J ≈ ω₀C_c).
* **c_shunt** — combline-style BPF: shorted stubs < λ/4 resonated by
  shunt loading caps, gap-capacitor J-inverters.
* **rc / crc / rl** — passive audio/LF ladders (staged 1st-order
  corners; honest approximations, see `design_rc`).

Lumped designs carry a `components` BOM ({ref,type,value,unit,role})
and lay out as an 0805 SMD pad cascade; every design dict sets `topo`.
All topologies evaluate through the same ABCD → S-parameter path —
the lumped section kinds are ideal two-terminal elements.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .em_physics import C0, EPS0, ETA0, MU0, SIGMA_CU
from .hadamard import sylvester

KINDS = ("lpf", "hpf", "bpf", "bsf")
PROTOS = ("butterworth", "chebyshev")
TOPOS = {
    "lpf": ("stepped", "lc", "qw_tl", "rc", "crc", "rl"),
    "hpf": ("stub", "lc", "rc", "rl"),
    "bpf": ("hairpin", "lc", "dc_lc", "qw_tl", "c_shunt"),
    "bsf": ("stub", "lc"),
}
# lumped topologies: audio/LF-capable, SMD BOM, schematic export
LUMPED_TOPOS = frozenset({"lc", "rc", "crc", "rl", "dc_lc"})
MIN_TRACE_M = 0.127e-3          # 5 mil JLCPCB floor
Z_HIGH = 100.0
Z_LOW = 20.0
RL_TARGET_DB = 10.0
REJ_TARGET_DBC = 40.0           # digest: 40 dBc at 10 % from the edge
SOURCE = (
    "https://cdn.everythingrf.com/live/"
    "Filters_updated_version_638463728945358500_1_3_638761394513756674.pdf"
)


# --- microstrip --------------------------------------------------------------


def eps_eff(eps_r: float, w: float, h: float) -> float:
    """Hammerstad effective permittivity (Pozar 3.195)."""
    er, u = float(eps_r), max(float(w) / max(float(h), 1e-12), 1e-6)
    base = 0.5 * (er + 1.0) + 0.5 * (er - 1.0) / math.sqrt(1.0 + 12.0 / u)
    if u < 1.0:
        base += 0.5 * (er - 1.0) * 0.04 * (1.0 - u) ** 2
    return base


def microstrip_z0(eps_r: float, w: float, h: float) -> float:
    """Wheeler analysis: Z₀ of a microstrip of width w on height h."""
    u = max(float(w) / max(float(h), 1e-12), 1e-6)
    ee = eps_eff(eps_r, w, h)
    if u <= 1.0:
        return (ETA0 / (2.0 * math.pi * math.sqrt(ee))) * math.log(
            8.0 / u + 0.25 * u)
    return ETA0 / (math.sqrt(ee) * (u + 1.393 + 0.667 * math.log(u + 1.444)))


def microstrip_width(eps_r: float, h_m: float, z0: float) -> float:
    """Wheeler/Hammerstad synthesis: width (m) for target Z₀.

    Same formula as `kicad_gen.microstrip_width_50ohm` — kept here so
    this module does not import the KiCad emitter (the emitter imports
    us for filter layouts).
    """
    er = float(eps_r)
    z0 = float(z0)
    a = (z0 / 60.0) * math.sqrt((er + 1.0) / 2.0) \
        + (er - 1.0) / (er + 1.0) * (0.23 + 0.11 / er)
    wh = 8.0 * math.exp(a) / (math.exp(2.0 * a) - 2.0)
    if wh > 2.0:
        b = ETA0 * math.pi / (2.0 * z0 * math.sqrt(er))
        wh = (2.0 / math.pi) * (
            b - 1.0 - math.log(2.0 * b - 1.0)
            + (er - 1.0) / (2.0 * er)
            * (math.log(b - 1.0) + 0.39 - 0.61 / er))
    return max(MIN_TRACE_M, wh * float(h_m))


def microstrip_alpha(f_hz: float, eps_r: float, w: float, h: float,
                     tan_delta: float, z0: float) -> float:
    """Np/m: dielectric + conductor (Pozar 3.198 / 3.199)."""
    ee = eps_eff(eps_r, w, h)
    alpha_d = (math.pi * f_hz * math.sqrt(ee) * max(tan_delta, 0.0)) / C0
    rs = math.sqrt(math.pi * f_hz * MU0 / SIGMA_CU)
    alpha_c = rs / max(z0 * max(w, MIN_TRACE_M), 1e-12)
    return alpha_d + alpha_c


def q_unloaded(f_hz: float, eps_r: float, h_m: float,
               tan_delta: float = 0.02, z0: float = 50.0) -> float:
    """Microstrip Q_u = β / (2α) on the 50 Ω line."""
    w = microstrip_width(eps_r, h_m, z0)
    ee = eps_eff(eps_r, w, h_m)
    beta = 2.0 * math.pi * f_hz * math.sqrt(ee) / C0
    alpha = microstrip_alpha(f_hz, eps_r, w, h_m, tan_delta, z0)
    return beta / max(2.0 * alpha, 1e-12)


# --- prototypes --------------------------------------------------------------


def prototype_g(kind: str, n: int, ripple_db: float = 0.1) -> list[float]:
    """Low-pass prototype g₀…g_{N+1} (Pozar 8.3).  g₀ is the source."""
    if kind not in PROTOS:
        raise ValueError(f"unknown prototype {kind!r}; expected {PROTOS}")
    n = int(n)
    if not (1 <= n <= 11):
        raise ValueError("filter order must be 1..11")
    g = [1.0]
    if kind == "butterworth":
        for k in range(1, n + 1):
            g.append(2.0 * math.sin((2 * k - 1) * math.pi / (2.0 * n)))
        g.append(1.0)
        return g
    # Chebyshev
    lr = max(float(ripple_db), 1e-6)
    beta = math.log(1.0 / math.tanh(lr / 17.37))
    gamma = math.sinh(beta / (2.0 * n))
    a = [math.sin((2 * k - 1) * math.pi / (2.0 * n)) for k in range(1, n + 1)]
    b = [gamma ** 2 + math.sin(k * math.pi / n) ** 2 for k in range(1, n + 1)]
    g.append(2.0 * a[0] / gamma)
    for k in range(2, n + 1):
        g.append(4.0 * a[k - 2] * a[k - 1] / (b[k - 2] * g[-1]))
    g.append(1.0 if (n % 2) else math.tanh(beta / 4.0) ** -2)
    return g


# --- ABCD / S ----------------------------------------------------------------


def _abcd_line(z0: float, theta: complex) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 1j * z0 * s], [1j * s / z0, c]], dtype=np.complex128)


def _abcd_shunt_y(y: complex) -> np.ndarray:
    return np.array([[1.0, 0.0], [y, 1.0]], dtype=np.complex128)


def _abcd_series_z(z: complex) -> np.ndarray:
    return np.array([[1.0, z], [0.0, 1.0]], dtype=np.complex128)


def _abcd_j(j: float, z0: float = 50.0) -> np.ndarray:
    """Admittance inverter J (Siemens).  ABCD = [[0, −j/J], [−jJ, 0]]
    so AD − BC = 1 (reciprocal, lossless)."""
    jv = float(j)
    return np.array([[0.0, -1j / jv], [-1j * jv, 0.0]], dtype=np.complex128)


def _theta(f_hz: float, len_m: float, eps_e: float, alpha: float) -> complex:
    beta = 2.0 * math.pi * f_hz * math.sqrt(eps_e) / C0
    return (beta - 1j * alpha) * float(len_m)


def _section_abcd(sec: dict, f_hz: float) -> np.ndarray:
    kind = sec["kind"]
    if kind == "line":
        th = _theta(f_hz, sec["len_m"], sec["eps_eff"], sec.get("alpha", 0.0))
        return _abcd_line(sec["z0"], th)
    if kind == "open_stub":
        th = _theta(f_hz, sec["len_m"], sec["eps_eff"], sec.get("alpha", 0.0))
        # Yin = j tanθ / Z  (lossy: tanh of complex θ)
        yin = (1j / sec["z0"]) * np.tan(th)
        return _abcd_shunt_y(yin)
    if kind == "short_stub":
        th = _theta(f_hz, sec["len_m"], sec["eps_eff"], sec.get("alpha", 0.0))
        yin = (-1j / sec["z0"]) / np.tan(th)          # −j cotθ / Z
        return _abcd_shunt_y(yin)
    if kind == "gap":
        w = 2.0 * math.pi * f_hz
        return _abcd_series_z(1.0 / (1j * w * max(sec["c_f"], 1e-18)))
    if kind == "inverter":
        return _abcd_j(sec["j"], sec.get("z0", 50.0))
    # --- lumped two-terminal elements (ideal L/C/R) ---
    w = 2.0 * math.pi * f_hz
    if kind == "series_l":                       # Z = jωL (+ optional ESR)
        return _abcd_series_z(1j * w * sec["l_h"] + sec.get("r_ohm", 0.0))
    if kind == "series_c":                       # Z = 1/(jωC)
        return _abcd_series_z(1.0 / (1j * w * max(sec["c_f"], 1e-18)))
    if kind == "series_r":                       # Z = R
        return _abcd_series_z(sec["r_ohm"])
    if kind == "shunt_l":                        # Y = 1/(jωL)
        return _abcd_shunt_y(1.0 / (1j * w * max(sec["l_h"], 1e-18)))
    if kind == "shunt_c":                        # Y = jωC
        return _abcd_shunt_y(1j * w * sec["c_f"])
    if kind == "shunt_r":                        # Y = 1/R
        return _abcd_shunt_y(1.0 / max(sec["r_ohm"], 1e-12))
    if kind == "series_rl":                      # Z = R + jωL (lossy L)
        return _abcd_series_z(sec["r_ohm"] + 1j * w * sec["l_h"])
    if kind == "shunt_rc":                       # Y = 1/R + jωC (R‖C to gnd)
        return _abcd_shunt_y(1.0 / max(sec["r_ohm"], 1e-12)
                             + 1j * w * sec["c_f"])
    if kind == "shunt_lc":                       # Y = jωC + 1/(jωL)  (L‖C gnd)
        return _abcd_shunt_y(1j * w * sec["c_f"]
                             + 1.0 / (1j * w * max(sec["l_h"], 1e-18)))
    if kind == "series_lc":                      # Z = jωL + 1/(jωC)
        z = 1j * w * sec["l_h"] + 1.0 / (1j * w * max(sec["c_f"], 1e-18))
        return _abcd_series_z(z)
    if kind == "series_plc":                     # Z = jωL/(1−ω²LC)  (L‖C arm)
        den = 1.0 - w * w * sec["l_h"] * sec["c_f"]
        if abs(den) < 1e-12:
            den = 1e-12
        return _abcd_series_z(1j * w * sec["l_h"] / den)
    if kind == "shunt_slc":                      # Y = 1/(jωL + 1/jωC) (to gnd)
        z = 1j * w * sec["l_h"] + 1.0 / (1j * w * max(sec["c_f"], 1e-18))
        if abs(z) < 1e-15:
            z = 1e-15
        return _abcd_shunt_y(1.0 / z)
    raise ValueError(f"unknown section kind {kind!r}")


def abcd_cascade(sections: list[dict], f_hz: float) -> np.ndarray:
    m = np.eye(2, dtype=np.complex128)
    for sec in sections:
        m = m @ _section_abcd(sec, f_hz)
    return m


def abcd_to_s(abcd: np.ndarray, z0: float = 50.0) -> tuple[complex, complex]:
    a, b, c, d = abcd[0, 0], abcd[0, 1], abcd[1, 0], abcd[1, 1]
    den = a + b / z0 + c * z0 + d
    s11 = (a + b / z0 - c * z0 - d) / den
    s21 = 2.0 / den
    return complex(s11), complex(s21)


def db20(x: complex | float) -> float:
    return 20.0 * math.log10(max(abs(x), 1e-12))


# --- section helpers ---------------------------------------------------------


def _line_sec(eps_r: float, h_m: float, z0: float, len_m: float,
              tan_delta: float, f_ref: float) -> dict:
    w = microstrip_width(eps_r, h_m, z0)
    ee = eps_eff(eps_r, w, h_m)
    alpha = microstrip_alpha(f_ref, eps_r, w, h_m, tan_delta, z0)
    return {"kind": "line", "z0": float(z0), "w_m": w, "len_m": float(len_m),
            "eps_eff": ee, "alpha": alpha}


def _stub_sec(kind: str, eps_r: float, h_m: float, z0: float, len_m: float,
              tan_delta: float, f_ref: float) -> dict:
    w = microstrip_width(eps_r, h_m, z0)
    ee = eps_eff(eps_r, w, h_m)
    alpha = microstrip_alpha(f_ref, eps_r, w, h_m, tan_delta, z0)
    return {"kind": kind, "z0": float(z0), "w_m": w, "len_m": float(len_m),
            "eps_eff": ee, "alpha": alpha}


def _lam_g(f_hz: float, eps_e: float) -> float:
    return C0 / (f_hz * math.sqrt(eps_e))


def _fill(kind: str, proto: str, n: int, f_c: float, f_lo: float, f_hi: float,
          eps_r: float, h_m: float, tan_delta: float, ripple_db: float,
          g: list[float], sections: list[dict], extra: dict | None = None,
          topo: str | None = None) -> dict:
    w50 = microstrip_width(eps_r, h_m, 50.0)
    qu = q_unloaded(f_c, eps_r, h_m, tan_delta)
    fbw = (f_hi - f_lo) / f_c if f_hi > f_lo else 1.0
    g_sum = float(sum(g[1:-1]))
    il_est = 4.343 * g_sum / max(fbw * qu, 1e-9)
    d = {
        "kind": kind, "proto": proto, "n": int(n),
        "topo": topo or TOPOS[kind][0],
        "f_c": float(f_c), "f_lo": float(f_lo), "f_hi": float(f_hi),
        "eps_r": float(eps_r), "h_m": float(h_m),
        "tan_delta": float(tan_delta), "ripple_db": float(ripple_db),
        "z0": 50.0, "g": [float(x) for x in g],
        "w50_m": float(w50), "q_u": float(qu),
        "il_est_db": float(il_est),
        "sections": sections,
    }
    if extra:
        d.update(extra)
    return d


# --- designs -----------------------------------------------------------------


def design_lpf(f_c: float, n: int = 5, proto: str = "butterworth",
               eps_r: float = 4.4, h_m: float = 1.6e-3,
               tan_delta: float = 0.02, ripple_db: float = 0.1,
               z_high: float = Z_HIGH, z_low: float = Z_LOW) -> dict:
    """Stepped-impedance microstrip low-pass (Pozar 8.6)."""
    g = prototype_g(proto, n, ripple_db)
    sections = []
    # leading 50 Ω feed
    w50 = microstrip_width(eps_r, h_m, 50.0)
    feed = max(2.0 * h_m, 3e-3)
    sections.append(_line_sec(eps_r, h_m, 50.0, feed, tan_delta, f_c))
    for k in range(1, n + 1):
        if k % 2 == 1:          # series L → high-Z
            z, role = z_high, "L"
            w = microstrip_width(eps_r, h_m, z)
            ee = eps_eff(eps_r, w, h_m)
            ell = _lam_g(f_c, ee) / (2.0 * math.pi) * (g[k] * 50.0 / z)
        else:                   # shunt C → low-Z
            z, role = z_low, "C"
            w = microstrip_width(eps_r, h_m, z)
            ee = eps_eff(eps_r, w, h_m)
            ell = _lam_g(f_c, ee) / (2.0 * math.pi) * (g[k] * z / 50.0)
        sec = _line_sec(eps_r, h_m, z, ell, tan_delta, f_c)
        sec["role"] = role
        sections.append(sec)
    sections.append(_line_sec(eps_r, h_m, 50.0, feed, tan_delta, f_c))
    return _fill("lpf", proto, n, f_c, 0.0, f_c, eps_r, h_m, tan_delta,
                 ripple_db, g, sections,
                 {"z_high": z_high, "z_low": z_low, "w50_m": w50})


def _gap_from_c(c_f: float, w: float, h_m: float, eps_r: float) -> float:
    """Invert a series-gap capacitance to a gap width (m).

    C ≈ ε₀ ε_eff w · (1/π) ln(coth(π s / 4h))  (Hammerstad end-gap).
    """
    ee = eps_eff(eps_r, w, h_m)
    target = float(c_f) / max(EPS0 * ee * w / math.pi, 1e-30)
    lo, hi = MIN_TRACE_M, 4.0 * h_m
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        pred = math.log(1.0 / max(math.tanh(math.pi * mid / (4.0 * h_m)), 1e-12))
        if pred > target:
            lo = mid
        else:
            hi = mid
    return max(MIN_TRACE_M, min(2e-3, 0.5 * (lo + hi)))


def design_hpf(f_c: float, n: int = 5, proto: str = "butterworth",
               eps_r: float = 4.4, h_m: float = 1.6e-3,
               tan_delta: float = 0.02, ripple_db: float = 0.1) -> dict:
    """Complementary high-pass: series gap-C and shunt shorted-stub L.

    LPF series L → series C = 1/(ω_c g_k Z₀); LPF shunt C → shunt L =
    Z₀/(ω_c g_k) realised as a shorted stub Zin = j Z_s tanθ.
    """
    g = prototype_g(proto, n, ripple_db)
    wc = 2.0 * math.pi * f_c
    w50 = microstrip_width(eps_r, h_m, 50.0)
    ee50 = eps_eff(eps_r, w50, h_m)
    feed = max(2.0 * h_m, 3e-3)
    sections = [_line_sec(eps_r, h_m, 50.0, feed, tan_delta, f_c)]
    for k in range(1, n + 1):
        if k % 2 == 1:
            c_f = 1.0 / (wc * g[k] * 50.0)
            gap = _gap_from_c(c_f, w50, h_m, eps_r)
            sections.append({"kind": "gap", "c_f": float(c_f),
                             "w_m": float(w50), "gap_m": float(gap)})
        else:
            l_h = 50.0 / (wc * g[k])
            z_s = Z_HIGH
            theta = math.atan(wc * l_h / z_s)
            ell = theta * _lam_g(f_c, ee50) / (2.0 * math.pi)
            sections.append(_stub_sec("short_stub", eps_r, h_m, z_s, ell,
                                      tan_delta, f_c))
    sections.append(_line_sec(eps_r, h_m, 50.0, feed, tan_delta, f_c))
    return _fill("hpf", proto, n, f_c, f_c, 4.0 * f_c, eps_r, h_m, tan_delta,
                 ripple_db, g, sections)


def design_bpf(f_lo: float, f_hi: float, n: int = 3,
               proto: str = "butterworth",
               eps_r: float = 4.4, h_m: float = 1.6e-3,
               tan_delta: float = 0.02, ripple_db: float = 0.1) -> dict:
    """Half-wave resonator BPF via J-inverters (Pozar 8.7), hairpin layout."""
    if not (f_lo < f_hi):
        raise ValueError("f_lo must be < f_hi")
    f0 = 0.5 * (f_lo + f_hi)
    delta = (f_hi - f_lo) / f0
    g = prototype_g(proto, n, ripple_db)
    y0 = 1.0 / 50.0
    j = [0.0] * (n + 1)
    j[0] = y0 * math.sqrt(math.pi * delta / (2.0 * g[0] * g[1]))
    for i in range(1, n):
        j[i] = y0 * math.pi * delta / (2.0 * math.sqrt(g[i] * g[i + 1]))
    j[n] = y0 * math.sqrt(math.pi * delta / (2.0 * g[n] * g[n + 1]))
    w50 = microstrip_width(eps_r, h_m, 50.0)
    ee50 = eps_eff(eps_r, w50, h_m)
    lam2 = _lam_g(f0, ee50) / 2.0
    feed = max(2.0 * h_m, 3e-3)
    sections = [_line_sec(eps_r, h_m, 50.0, feed, tan_delta, f0)]
    gaps = []
    for i in range(n + 1):
        sections.append({"kind": "inverter", "j": float(j[i]), "z0": 50.0})
        # edge-coupled gap from k ≈ (Z0e−Z0o)/(Z0e+Z0o) ≈ J/Y0
        k_c = min(0.85, max(0.02, j[i] / y0))
        s = max(MIN_TRACE_M, (2.0 * h_m / math.pi) * math.acosh(1.0 / k_c))
        gaps.append(float(s))
        if i < n:
            sec = _line_sec(eps_r, h_m, 50.0, lam2, tan_delta, f0)
            sec["role"] = "resonator"
            sections.append(sec)
    sections.append(_line_sec(eps_r, h_m, 50.0, feed, tan_delta, f0))
    extra = {"fbw": float(delta), "j_y0": [float(x / y0) for x in j],
             "gaps_m": gaps, "lam2_m": float(lam2)}
    return _fill("bpf", proto, n, f0, f_lo, f_hi, eps_r, h_m, tan_delta,
                 ripple_db, g, sections, extra)


def design_bsf(f_lo: float, f_hi: float, n: int = 3,
               proto: str = "butterworth",
               eps_r: float = 4.4, h_m: float = 1.6e-3,
               tan_delta: float = 0.02, ripple_db: float = 0.1) -> dict:
    """Open-stub bandstop on a 50 Ω through-line (Pozar 8.8.3)."""
    if not (f_lo < f_hi):
        raise ValueError("f_lo must be < f_hi")
    f0 = 0.5 * (f_lo + f_hi)
    delta = (f_hi - f_lo) / f0
    g = prototype_g(proto, n, ripple_db)
    w50 = microstrip_width(eps_r, h_m, 50.0)
    ee50 = eps_eff(eps_r, w50, h_m)
    lam4 = _lam_g(f0, ee50) / 4.0
    feed = max(2.0 * h_m, 3e-3)
    sections = [_line_sec(eps_r, h_m, 50.0, feed, tan_delta, f0)]
    for k in range(1, n + 1):
        y_over = (4.0 * g[k] / math.pi) * delta
        z_stub = 50.0 / max(y_over, 0.05)
        z_stub = min(max(z_stub, Z_LOW), 150.0)
        sections.append(_stub_sec("open_stub", eps_r, h_m, z_stub, lam4,
                                  tan_delta, f0))
        if k < n:
            sections.append(_line_sec(eps_r, h_m, 50.0, lam4, tan_delta, f0))
    sections.append(_line_sec(eps_r, h_m, 50.0, feed, tan_delta, f0))
    return _fill("bsf", proto, n, f0, f_lo, f_hi, eps_r, h_m, tan_delta,
                 ripple_db, g, sections, {"fbw": float(delta)})


# --- lumped / commensurate topologies ------------------------------------------


def _add_comp(comps: list[dict], prefix: str, value: float, role: str) -> None:
    """Append a BOM entry {ref,type,value,unit,role}; refs count per type."""
    k = sum(1 for c in comps if c["type"] == prefix) + 1
    comps.append({"ref": f"{prefix}{k}", "type": prefix,
                  "value": float(value),
                  "unit": {"L": "H", "C": "F", "R": "Ω"}[prefix],
                  "role": role})


def design_lc(kind: str, f_c: float | None = None,
              f_lo: float | None = None, f_hi: float | None = None,
              n: int = 5, proto: str = "butterworth", ripple_db: float = 0.1,
              z0: float = 50.0, eps_r: float = 4.4, h_m: float = 1.6e-3,
              tan_delta: float = 0.02) -> dict:
    """Lumped LC ladder from the g-prototype (Pozar 8.3/8.4).

    Series-first ladder between a Z₀ source and load:

    * **LPF** — series L_k = g_k Z₀/ω_c (odd k), shunt C_k = g_k/(Z₀ ω_c).
    * **HPF** — LP→HP dual: series C_k = 1/(g_k Z₀ ω_c), shunt
      L_k = Z₀/(g_k ω_c).
    * **BPF/BSF** — LP→BP / LP→BS reactance transforms (Pozar 8.4,
      Table 8.6) with ω₀ = √(ω_lo ω_hi) (geometric mean) and
      Δ = (f_hi − f_lo)/f₀.  Every branch resonates at f₀:

      =====  ============================  ==============================
      arm    BP (series LC / shunt L‖C)    BS (series L‖C / shunt LC)
      =====  ============================  ==============================
      series L = g_k Z₀/(Δω₀)             L = Δ g_k Z₀/ω₀
             C = Δ/(g_k Z₀ ω₀)            C = 1/(Δ g_k Z₀ ω₀)
      shunt  L = Δ Z₀/(g_k ω₀)            L = Z₀/(Δ g_k ω₀)
             C = g_k/(Δ Z₀ ω₀)            C = Δ g_k/(Z₀ ω₀)
      =====  ============================  ==============================
    """
    kind = kind.lower()
    if kind not in KINDS:
        raise ValueError(f"unknown filter kind {kind!r}; expected {KINDS}")
    g = prototype_g(proto, n, ripple_db)
    z0 = float(z0)
    comps: list[dict] = []
    sections: list[dict] = []
    extra: dict = {}
    if kind in ("lpf", "hpf"):
        if f_c is None:
            raise ValueError(f"{kind} requires f_c")
        wc = 2.0 * math.pi * float(f_c)
        for k in range(1, n + 1):
            if kind == "lpf":
                if k % 2 == 1:
                    l_h = g[k] * z0 / wc
                    sections.append({"kind": "series_l", "l_h": l_h})
                    _add_comp(comps, "L", l_h, "series")
                else:
                    c_f = g[k] / (z0 * wc)
                    sections.append({"kind": "shunt_c", "c_f": c_f})
                    _add_comp(comps, "C", c_f, "shunt")
            else:
                if k % 2 == 1:
                    c_f = 1.0 / (g[k] * z0 * wc)
                    sections.append({"kind": "series_c", "c_f": c_f})
                    _add_comp(comps, "C", c_f, "series")
                else:
                    l_h = z0 / (g[k] * wc)
                    sections.append({"kind": "shunt_l", "l_h": l_h})
                    _add_comp(comps, "L", l_h, "shunt")
        fc = float(f_c)
        flo, fhi = (0.0, fc) if kind == "lpf" else (fc, 4.0 * fc)
    else:
        if f_lo is None or f_hi is None or not (f_lo < f_hi):
            raise ValueError(f"{kind} requires f_lo < f_hi")
        f0 = math.sqrt(float(f_lo) * float(f_hi))
        delta = (float(f_hi) - float(f_lo)) / f0
        w0 = 2.0 * math.pi * f0
        for k in range(1, n + 1):
            if kind == "bpf":
                if k % 2 == 1:
                    l_h = g[k] * z0 / (delta * w0)
                    c_f = delta / (g[k] * z0 * w0)
                    sections.append({"kind": "series_lc",
                                     "l_h": l_h, "c_f": c_f})
                else:
                    l_h = delta * z0 / (g[k] * w0)
                    c_f = g[k] / (delta * z0 * w0)
                    sections.append({"kind": "shunt_lc",
                                     "l_h": l_h, "c_f": c_f})
            else:                                   # bsf
                if k % 2 == 1:
                    l_h = delta * g[k] * z0 / w0
                    c_f = 1.0 / (delta * g[k] * z0 * w0)
                    sections.append({"kind": "series_plc",
                                     "l_h": l_h, "c_f": c_f})
                else:
                    l_h = z0 / (delta * g[k] * w0)
                    c_f = g[k] * delta / (z0 * w0)
                    sections.append({"kind": "shunt_slc",
                                     "l_h": l_h, "c_f": c_f})
            _add_comp(comps, "L", sections[-1]["l_h"],
                      "series" if k % 2 == 1 else "shunt")
            _add_comp(comps, "C", sections[-1]["c_f"],
                      "series" if k % 2 == 1 else "shunt")
        fc, flo, fhi = f0, float(f_lo), float(f_hi)
        extra = {"fbw": float(delta)}
    d = _fill(kind, proto, n, fc, flo, fhi, eps_r, h_m, tan_delta,
              ripple_db, g, sections,
              {"il_est_db": 0.0, "z0": z0, **extra}, topo="lc")
    d["components"] = comps
    return d


def design_dc_lc(f_lo: float, f_hi: float, n: int = 3,
                 proto: str = "butterworth", ripple_db: float = 0.1,
                 z0: float = 50.0, eps_r: float = 4.4, h_m: float = 1.6e-3,
                 tan_delta: float = 0.02) -> dict:
    """Capacitively-coupled shunt-LC band-pass (nodal "dc_lc" ladder).

    n identical shunt parallel-LC resonators at f₀ = √(f_lo·f_hi) with
    L_r = Z₀/ω₀, C_r = 1/(ω₀² L_r) (slope parameter b = ω₀ C_r = Y₀),
    coupled by series capacitors approximating J-inverters (Pozar 8.7):

        J_{0,1}   = √(Y₀ b Δ/(g₀ g₁)),   J_{k,k+1} = Δ b/√(g_k g_{k+1}),
        J_{n,n+1} = √(Y₀ b Δ/(g_n g_{n+1})),   C_c = J/ω₀.

    A series capacitor between two resonator nodes inverts with
    J ≈ ω₀ C_c and contributes −C_c shunt arms, absorbed into the
    resonators:

        C_k = C_r − C_{k−1,k} − C_{k,k+1}     (must stay positive).

    Approximation: J = ω₀ C_c is first-order in the coupling reactance
    — accurate for narrow/moderate fractional bandwidths (Δ ≲ 0.2).
    End coupling to Z₀ is the J_{0,1} / J_{n,n+1} capacitor.
    """
    if not (f_lo < f_hi):
        raise ValueError("f_lo must be < f_hi")
    g = prototype_g(proto, n, ripple_db)
    z0 = float(z0)
    y0 = 1.0 / z0
    f0 = math.sqrt(float(f_lo) * float(f_hi))
    delta = (float(f_hi) - float(f_lo)) / f0
    w0 = 2.0 * math.pi * f0
    l_r = z0 / w0
    c_r = 1.0 / (w0 * w0 * l_r)
    b = w0 * c_r                                    # = Y₀
    j = [0.0] * (n + 1)
    j[0] = math.sqrt(y0 * b * delta / (g[0] * g[1]))
    for k in range(1, n):
        j[k] = delta * b / math.sqrt(g[k] * g[k + 1])
    j[n] = math.sqrt(y0 * b * delta / (g[n] * g[n + 1]))
    cc = [x / w0 for x in j]                        # series coupling caps
    comps: list[dict] = []
    sections: list[dict] = []
    for k in range(1, n + 1):
        c_k = c_r - cc[k - 1] - cc[k]
        if c_k <= 0.0:
            raise ValueError(
                f"coupling caps exceed the resonator capacitance at node "
                f"{k} (C_r={c_r:.3g} F); reduce bandwidth or order")
        sections.append({"kind": "series_c", "c_f": float(cc[k - 1])})
        _add_comp(comps, "C", cc[k - 1], "series")
        sections.append({"kind": "shunt_lc", "l_h": l_r, "c_f": float(c_k)})
        _add_comp(comps, "L", l_r, "shunt")
        _add_comp(comps, "C", c_k, "shunt")
    sections.append({"kind": "series_c", "c_f": float(cc[n])})
    _add_comp(comps, "C", cc[n], "series")
    d = _fill("bpf", proto, n, f0, float(f_lo), float(f_hi), eps_r, h_m,
              tan_delta, ripple_db, g, sections,
              {"il_est_db": 0.0, "z0": z0, "fbw": float(delta),
               "j_y0": [float(x / y0) for x in j]}, topo="dc_lc")
    d["components"] = comps
    return d


def design_qw_tl(kind: str, f_c: float | None = None,
                 f_lo: float | None = None, f_hi: float | None = None,
                 n: int = 5, proto: str = "butterworth",
                 ripple_db: float = 0.1, z0: float = 50.0,
                 eps_r: float = 4.4, h_m: float = 1.6e-3,
                 tan_delta: float = 0.02, z_stub_bpf: float = 30.0,
                 z_min: float = Z_LOW, z_max: float = 150.0) -> dict:
    """Commensurate quarter-wave TEM-line filter (Richards + Kuroda).

    Richards' transform Ω = tan(βℓ) maps the lumped prototype onto
    commensurate stubs: a series L is a shorted stub of Z = g_k, a
    shunt C an open stub of Z = 1/g_k (normalized, λ/8 at f_c so the
    Ω = 1 cutoff sits at βℓ = π/4).  Kuroda's identities convert the
    series stubs to shunt form against matched unit elements (a matched
    UE adds phase only):

        UE(Z_u) · series shorted stub(Z_s)
              ≡ shunt open stub(Z_u(Z_u+Z_s)/Z_s) · UE(Z_u+Z_s),
        UE(Z_u) · shunt open stub(Z_o)  ≡  shunt open stub(Z_o) · UE(Z_u),

    so the LPF becomes all *open* stubs separated by λ/8 UEs (λ/4 at
    the commensurate frequency 2 f_c).

    The BPF uses shunt *shorted* λ/4 stubs at f₀ = √(f_lo f_hi) —
    parallel resonators with slope b = π Y_stub/4 — separated by λ/4
    UEs acting as J-inverters (J = 1/Z_ue).  All stubs share
    Z = `z_stub_bpf` and the UE impedances carry the g-weighting:
    Z_ue,0,1 = √(g₀g₁/(Y₀ b Δ))·Z₀… i.e. 1/J with J from the usual
    inverter ladder (see `design_dc_lc`).

    All impedances are clamped to [`z_min`, `z_max`] Ω for microstrip
    realizability — an intentional, flagged (`z_clamped`) approximation
    that detunes the exact prototype response (worst for narrow BPF
    bandwidths, where interior unit elements want Z > 150 Ω).
    """
    kind = kind.lower()
    g = prototype_g(proto, n, ripple_db)
    z0 = float(z0)

    def _clamp(z: float) -> float:
        return min(max(z, z_min), z_max)

    if kind == "lpf":
        if f_c is None:
            raise ValueError("lpf qw_tl requires f_c")
        # normalized ladder: s = series shorted stub (Z=g), p = shunt
        # open stub (Z=1/g); bracket with matched UEs
        net: list[list] = [["ue", 1.0]]
        for k in range(1, n + 1):
            net.append(["s", g[k]] if k % 2 == 1 else ["p", 1.0 / g[k]])
        net.append(["ue", 1.0])
        guard = 0
        while True:
            i = next((i for i, e in enumerate(net) if e[0] == "s"), None)
            if i is None:
                break
            guard += 1
            if guard > 200:
                raise RuntimeError("kuroda elimination did not converge")
            if net[i - 1][0] == "ue":
                zu, zs = net[i - 1][1], net[i][1]
                net[i - 1:i + 1] = [["p", zu * (zu + zs) / zs],
                                    ["ue", zu + zs]]
            elif i + 1 < len(net) and net[i + 1][0] == "ue":
                zu, zs = net[i + 1][1], net[i][1]
                net[i:i + 2] = [["ue", zu + zs],
                                ["p", zu * (zu + zs) / zs]]
            else:
                # no adjacent UE: bubble the nearest left UE rightward —
                # UEs commute with shunt stubs (2nd identity)
                j = i - 1
                while net[j][0] != "ue":
                    j -= 1
                for k in range(j, i - 1):
                    net[k], net[k + 1] = net[k + 1], net[k]
        f_c = float(f_c)
        sections = []
        zs_net = []
        for t, zn in net:
            z = _clamp(zn * z0)
            zs_net.append(z)
            ee = eps_eff(eps_r, microstrip_width(eps_r, h_m, z), h_m)
            ell = _lam_g(f_c, ee) / 8.0
            if t == "p":
                sections.append(_stub_sec("open_stub", eps_r, h_m, z, ell,
                                          tan_delta, f_c))
            else:
                sections.append(_line_sec(eps_r, h_m, z, ell, tan_delta, f_c))
        extra = {
            "z_sections_ohm": zs_net,
            "z_clamped": any(abs(a - bnm * z0) > 1e-9
                             for a, (t, bnm) in zip(zs_net, net)),
            "f_commensurate": 2.0 * f_c,
        }
        return _fill("lpf", proto, n, f_c, 0.0, f_c, eps_r, h_m, tan_delta,
                     ripple_db, g, sections, extra, topo="qw_tl")
    if kind == "bpf":
        if f_lo is None or f_hi is None or not (f_lo < f_hi):
            raise ValueError("bpf qw_tl requires f_lo < f_hi")
        f0 = math.sqrt(float(f_lo) * float(f_hi))
        delta = (float(f_hi) - float(f_lo)) / f0
        y0 = 1.0 / z0
        b = math.pi / (4.0 * float(z_stub_bpf))      # shorted λ/4 slope
        j = [0.0] * (n + 1)
        j[0] = math.sqrt(y0 * b * delta / (g[0] * g[1]))
        for k in range(1, n):
            j[k] = delta * b / math.sqrt(g[k] * g[k + 1])
        j[n] = math.sqrt(y0 * b * delta / (g[n] * g[n + 1]))
        z_ue = [_clamp(1.0 / x) for x in j]
        w_st = microstrip_width(eps_r, h_m, z_stub_bpf)
        lam4_st = _lam_g(f0, eps_eff(eps_r, w_st, h_m)) / 4.0
        sections = []
        zs_net = []
        for k in range(n + 1):
            zl = z_ue[k]
            ee = eps_eff(eps_r, microstrip_width(eps_r, h_m, zl), h_m)
            sections.append(_line_sec(eps_r, h_m, zl,
                                      _lam_g(f0, ee) / 4.0, tan_delta, f0))
            zs_net.append(zl)
            if k < n:
                sections.append(_stub_sec("short_stub", eps_r, h_m,
                                          float(z_stub_bpf), lam4_st,
                                          tan_delta, f0))
                zs_net.append(float(z_stub_bpf))
        extra = {
            "fbw": float(delta), "z_stub_ohm": float(z_stub_bpf),
            "z_ue_ohm": z_ue, "z_sections_ohm": zs_net,
            "z_clamped": any(abs(z_ue[k] - 1.0 / j[k]) > 1e-9
                             for k in range(n + 1)),
            "j_y0": [float(x / y0) for x in j],
        }
        return _fill("bpf", proto, n, f0, float(f_lo), float(f_hi), eps_r,
                     h_m, tan_delta, ripple_db, g, sections, extra,
                     topo="qw_tl")
    raise ValueError(f"qw_tl topology not defined for kind {kind!r}")


def design_c_shunt(f_lo: float, f_hi: float, n: int = 3,
                   proto: str = "butterworth", ripple_db: float = 0.1,
                   theta0_deg: float = 60.0, z_stub: float = 50.0,
                   z0: float = 50.0, eps_r: float = 4.4, h_m: float = 1.6e-3,
                   tan_delta: float = 0.02) -> dict:
    """Combline-style capacitively-loaded shunt-resonator BPF.

    Each resonator is a shorted microstrip stub of electrical length
    θ₀ < 90° at f₀ = √(f_lo f_hi) (default θ₀ = 60°), resonated by a
    shunt loading capacitor (combline principle, Hong & Lancaster
    ch. 9):

        C_load = cot θ₀/(ω₀ Z_s),
        B(ω) = ω C − cot(θ₀ ω/ω₀)/Z_s,
        b = (ω₀/2) B′(ω₀) = ω₀ C/2 + θ₀ csc²θ₀/(2 Z_s).

    Series gap capacitors approximate the J-inverters (same J ladder as
    `design_dc_lc`, J ≈ ω₀ C_c) and are inverted to physical gaps with
    `_gap_from_c`; the loading caps absorb the inverter's negative
    shunt arms: C_k = C_load − C_{k−1,k} − C_{k,k+1}.  First-order in
    the coupling — accurate for Δ ≲ 0.2.
    """
    if not (f_lo < f_hi):
        raise ValueError("f_lo must be < f_hi")
    g = prototype_g(proto, n, ripple_db)
    z0 = float(z0)
    y0 = 1.0 / z0
    f0 = math.sqrt(float(f_lo) * float(f_hi))
    delta = (float(f_hi) - float(f_lo)) / f0
    w0 = 2.0 * math.pi * f0
    th0 = math.radians(float(theta0_deg))
    c_load = (1.0 / math.tan(th0)) / (w0 * float(z_stub))
    b = 0.5 * (w0 * c_load + th0 / (math.sin(th0) ** 2 * float(z_stub)))
    j = [0.0] * (n + 1)
    j[0] = math.sqrt(y0 * b * delta / (g[0] * g[1]))
    for k in range(1, n):
        j[k] = delta * b / math.sqrt(g[k] * g[k + 1])
    j[n] = math.sqrt(y0 * b * delta / (g[n] * g[n + 1]))
    cc = [x / w0 for x in j]
    w50 = microstrip_width(eps_r, h_m, 50.0)
    ell_st = th0 * _lam_g(f0, eps_eff(eps_r, microstrip_width(
        eps_r, h_m, z_stub), h_m)) / (2.0 * math.pi)
    feed = max(2.0 * h_m, 3e-3)
    comps: list[dict] = []
    sections = [_line_sec(eps_r, h_m, 50.0, feed, tan_delta, f0)]
    for k in range(1, n + 1):
        c_k = c_load - cc[k - 1] - cc[k]
        if c_k <= 0.0:
            raise ValueError(
                f"coupling caps exceed the loading capacitance at node "
                f"{k}; reduce bandwidth, theta0, or z_stub")
        sections.append({"kind": "gap", "c_f": float(cc[k - 1]),
                         "w_m": float(w50),
                         "gap_m": float(_gap_from_c(cc[k - 1], w50, h_m,
                                                    eps_r))})
        _add_comp(comps, "C", cc[k - 1], "series")
        sections.append(_stub_sec("short_stub", eps_r, h_m, float(z_stub),
                                  ell_st, tan_delta, f0))
        sections.append({"kind": "shunt_c", "c_f": float(c_k)})
        _add_comp(comps, "C", c_k, "shunt")
    sections.append({"kind": "gap", "c_f": float(cc[n]), "w_m": float(w50),
                     "gap_m": float(_gap_from_c(cc[n], w50, h_m, eps_r))})
    _add_comp(comps, "C", cc[n], "series")
    sections.append(_line_sec(eps_r, h_m, 50.0, feed, tan_delta, f0))
    d = _fill("bpf", proto, n, f0, float(f_lo), float(f_hi), eps_r, h_m,
              tan_delta, ripple_db, g, sections,
              {"fbw": float(delta), "theta0_deg": float(theta0_deg),
               "z_stub_ohm": float(z_stub),
               "j_y0": [float(x / y0) for x in j]}, topo="c_shunt")
    d["components"] = comps
    return d


def design_rc(kind: str = "lpf", f_c: float = 1e3, n: int = 3,
              z0: float = 50.0, scale: float = 10.0, eps_r: float = 4.4,
              h_m: float = 1.6e-3, tan_delta: float = 0.02) -> dict:
    """Passive RC ladder (audio/LF), Butterworth-staged 1st-order sections.

    n cascaded RC stages with per-stage corner

        f_s = f_c / √(2^{1/n} − 1)

    so the cascade sits −3.01 dB at f_c and rolls off −20n dB/decade.
    Stage impedances scale ×`scale` per stage (R_k = Z₀·scale^{k−1},
    C_k = 1/(2π f_s R_k)) to limit inter-stage loading.  LPF stage =
    series R + shunt C; HPF stage = series C + shunt R.

    Honest approximation: a *passive* RC ladder has only real poles and
    cannot reproduce true Butterworth complex pole pairs; stage loading
    shifts the corners further.  The taper bounds inter-stage loading
    to ~10 % between stages 1…n−1, but the Z₀ terminations themselves
    load the ladder: measured at n=3, ×10 in a 50 Ω environment the
    LPF corner lands ≈ 25 % low (and DC insertion is large — the series
    R's divide into the 50 Ω load), while the HPF corner runs high
    because the load parallels the final shunt R (use scale ≈ 1–2 for
    HPF, or active stages for real corners).  The ABCD sweep shows the
    real loaded response, Z₀ source and load included.
    """
    kind = kind.lower()
    if kind not in ("lpf", "hpf"):
        raise ValueError("design_rc supports lpf/hpf only")
    g = prototype_g("butterworth", n)
    f_s = float(f_c) / math.sqrt(2.0 ** (1.0 / int(n)) - 1.0)
    comps: list[dict] = []
    sections: list[dict] = []
    for k in range(1, n + 1):
        r_k = float(z0) * float(scale) ** (k - 1)
        c_k = 1.0 / (2.0 * math.pi * f_s * r_k)
        if kind == "lpf":
            sections.append({"kind": "series_r", "r_ohm": r_k})
            _add_comp(comps, "R", r_k, "series")
            sections.append({"kind": "shunt_c", "c_f": c_k})
            _add_comp(comps, "C", c_k, "shunt")
        else:
            sections.append({"kind": "series_c", "c_f": c_k})
            _add_comp(comps, "C", c_k, "series")
            sections.append({"kind": "shunt_r", "r_ohm": r_k})
            _add_comp(comps, "R", r_k, "shunt")
    flo, fhi = ((0.0, float(f_c)) if kind == "lpf"
                else (float(f_c), 4.0 * float(f_c)))
    d = _fill(kind, "butterworth", n, float(f_c), flo, fhi, eps_r, h_m,
              tan_delta, 0.0, g, sections,
              {"il_est_db": 0.0, "z0": float(z0),
               "stage_corner_hz": f_s, "stage_scale": float(scale)},
              topo="rc")
    d["components"] = comps
    return d


def design_crc(f_c: float = 1e3, n: int = 2, z0: float = 50.0,
               scale: float = 10.0, eps_r: float = 4.4, h_m: float = 1.6e-3,
               tan_delta: float = 0.02) -> dict:
    """CRC π ladder (supply/audio smoothing LPF): C–(R–C)×n.

    A shunt C at every node with a series R between nodes: n series
    resistors and n+1 capacitors.  Same staged-corner rule as
    `design_rc` with n+1 poles (f_s = f_c/√(2^{1/(n+1)} − 1)),
    R_k = Z₀·scale^{k−1}, C_k = 1/(2π f_s R_k) with each C scaled
    alongside its following R.  The series R damps the π sections (no
    peaking).  Passive real-pole approximation — see `design_rc`.
    """
    n = int(n)
    g = prototype_g("butterworth", n + 1)
    f_s = float(f_c) / math.sqrt(2.0 ** (1.0 / (n + 1)) - 1.0)
    comps: list[dict] = []
    sections: list[dict] = []
    for k in range(1, n + 2):
        r_k = float(z0) * float(scale) ** (k - 1)
        c_k = 1.0 / (2.0 * math.pi * f_s * r_k)
        sections.append({"kind": "shunt_c", "c_f": c_k})
        _add_comp(comps, "C", c_k, "shunt")
        if k <= n:
            sections.append({"kind": "series_r", "r_ohm": r_k})
            _add_comp(comps, "R", r_k, "series")
    d = _fill("lpf", "butterworth", n + 1, float(f_c), 0.0, float(f_c),
              eps_r, h_m, tan_delta, 0.0, g, sections,
              {"il_est_db": 0.0, "z0": float(z0),
               "stage_corner_hz": f_s, "stage_scale": float(scale)},
              topo="crc")
    d["components"] = comps
    return d


def design_rl(kind: str = "lpf", f_c: float = 1e3, n: int = 3,
              z0: float = 50.0, scale: float = 10.0, eps_r: float = 4.4,
              h_m: float = 1.6e-3, tan_delta: float = 0.02) -> dict:
    """Passive RL ladder (audio/LF).  Per-stage corner f_c = R/(2πL).

    LPF stage = series L + shunt R; HPF stage = series R + shunt L,
    with staged corners f_s = f_c/√(2^{1/n} − 1) and L_k = R_k/(2π f_s),
    R_k = Z₀·scale^{k−1}.  The resistors are real loss elements — the
    ABCD sweep shows the true insertion loss.  Same passive real-pole
    staging approximation as `design_rc`.
    """
    kind = kind.lower()
    if kind not in ("lpf", "hpf"):
        raise ValueError("design_rl supports lpf/hpf only")
    g = prototype_g("butterworth", n)
    f_s = float(f_c) / math.sqrt(2.0 ** (1.0 / int(n)) - 1.0)
    comps: list[dict] = []
    sections: list[dict] = []
    for k in range(1, n + 1):
        r_k = float(z0) * float(scale) ** (k - 1)
        l_k = r_k / (2.0 * math.pi * f_s)
        if kind == "lpf":
            sections.append({"kind": "series_l", "l_h": l_k})
            _add_comp(comps, "L", l_k, "series")
            sections.append({"kind": "shunt_r", "r_ohm": r_k})
            _add_comp(comps, "R", r_k, "shunt")
        else:
            sections.append({"kind": "series_r", "r_ohm": r_k})
            _add_comp(comps, "R", r_k, "series")
            sections.append({"kind": "shunt_l", "l_h": l_k})
            _add_comp(comps, "L", l_k, "shunt")
    flo, fhi = ((0.0, float(f_c)) if kind == "lpf"
                else (float(f_c), 4.0 * float(f_c)))
    d = _fill(kind, "butterworth", n, float(f_c), flo, fhi, eps_r, h_m,
              tan_delta, 0.0, g, sections,
              {"il_est_db": 0.0, "z0": float(z0),
               "stage_corner_hz": f_s, "stage_scale": float(scale)},
              topo="rl")
    d["components"] = comps
    return d


def design_filter(kind: str, f_c: float | None = None,
                  f_lo: float | None = None, f_hi: float | None = None,
                  n: int = 5, proto: str = "butterworth",
                  eps_r: float = 4.4, h_m: float = 1.6e-3,
                  tan_delta: float = 0.02, ripple_db: float = 0.1,
                  topo: str | None = None, z0: float = 50.0) -> dict:
    """Dispatch on (kind, topo).  `topo` defaults to TOPOS[kind][0] —
    the distributed microstrip realisation — so existing calls are
    unchanged.  `z0` is the system impedance (lumped/RC/RL ladders).
    """
    kind = kind.lower()
    if kind not in KINDS:
        raise ValueError(f"unknown filter kind {kind!r}; expected {KINDS}")
    topo = TOPOS[kind][0] if topo is None else topo.lower()
    if topo not in TOPOS[kind]:
        raise ValueError(
            f"topology {topo!r} not available for {kind}; "
            f"expected one of {TOPOS[kind]}")
    if kind in ("lpf", "hpf"):
        if f_c is None:
            raise ValueError(f"{kind} requires f_c")
        if topo == "lc":
            return design_lc(kind, f_c=f_c, n=n, proto=proto,
                             ripple_db=ripple_db, z0=z0, eps_r=eps_r,
                             h_m=h_m, tan_delta=tan_delta)
        if topo == "qw_tl":
            return design_qw_tl("lpf", f_c=f_c, n=n, proto=proto,
                                ripple_db=ripple_db, z0=z0, eps_r=eps_r,
                                h_m=h_m, tan_delta=tan_delta)
        if topo == "rc":
            return design_rc(kind, f_c=f_c, n=n, z0=z0, eps_r=eps_r,
                             h_m=h_m, tan_delta=tan_delta)
        if topo == "crc":
            return design_crc(f_c=f_c, n=n, z0=z0, eps_r=eps_r,
                              h_m=h_m, tan_delta=tan_delta)
        if topo == "rl":
            return design_rl(kind, f_c=f_c, n=n, z0=z0, eps_r=eps_r,
                             h_m=h_m, tan_delta=tan_delta)
        fn = design_lpf if kind == "lpf" else design_hpf
        return fn(f_c, n=n, proto=proto, eps_r=eps_r, h_m=h_m,
                  tan_delta=tan_delta, ripple_db=ripple_db)
    if f_lo is None or f_hi is None:
        if f_c is None:
            raise ValueError(f"{kind} requires f_lo/f_hi or f_c + 10 % FBW")
        f_lo, f_hi = 0.95 * f_c, 1.05 * f_c
    if topo == "lc":
        return design_lc(kind, f_lo=f_lo, f_hi=f_hi, n=n, proto=proto,
                         ripple_db=ripple_db, z0=z0, eps_r=eps_r, h_m=h_m,
                         tan_delta=tan_delta)
    if topo == "dc_lc":
        return design_dc_lc(f_lo, f_hi, n=n, proto=proto,
                            ripple_db=ripple_db, z0=z0, eps_r=eps_r,
                            h_m=h_m, tan_delta=tan_delta)
    if topo == "qw_tl":
        return design_qw_tl("bpf", f_lo=f_lo, f_hi=f_hi, n=n, proto=proto,
                            ripple_db=ripple_db, z0=z0, eps_r=eps_r,
                            h_m=h_m, tan_delta=tan_delta)
    if topo == "c_shunt":
        return design_c_shunt(f_lo, f_hi, n=n, proto=proto,
                              ripple_db=ripple_db, z0=z0, eps_r=eps_r,
                              h_m=h_m, tan_delta=tan_delta)
    fn = design_bpf if kind == "bpf" else design_bsf
    nn = n if n <= 7 else 5
    return fn(f_lo, f_hi, n=nn, proto=proto, eps_r=eps_r, h_m=h_m,
              tan_delta=tan_delta, ripple_db=ripple_db)


# --- sweep / metrics ---------------------------------------------------------


def sweep(design: dict, f_lo: float | None = None, f_hi: float | None = None,
          n_points: int = 81) -> dict:
    """S11/S21 vs frequency.  Default span is 0.1 f_c … 3 f_c."""
    fc = float(design["f_c"])
    if f_lo is None:
        f_lo = 0.1 * fc if design["kind"] != "hpf" else 0.05 * fc
    if f_hi is None:
        f_hi = 3.0 * fc
    freqs = np.linspace(float(f_lo), float(f_hi), int(n_points))
    s11, s21 = [], []
    for f in freqs:
        a = abcd_cascade(design["sections"], float(f))
        r, t = abcd_to_s(a, design.get("z0", 50.0))
        s11.append(complex(r))
        s21.append(complex(t))
    return {
        "f_hz": freqs.tolist(),
        "s11": s11,
        "s21": s21,
        "s11_db": [db20(x) for x in s11],
        "s21_db": [db20(x) for x in s21],
    }


def _in_pass(kind: str, f: float, d: dict) -> bool:
    if kind == "lpf":
        return f <= d["f_c"]
    if kind == "hpf":
        return f >= d["f_c"]
    if kind == "bpf":
        return d["f_lo"] <= f <= d["f_hi"]
    return f <= d["f_lo"] or f >= d["f_hi"]          # bsf pass = outside notch


def _in_stop(kind: str, f: float, d: dict) -> bool:
    # Near-skirt stop (digest: 40 dBc at 10 % from the edge).  Cut off
    # before the first distributed harmonic (≈ 2 f_c) so a λ/2 spurious
    # passband is not scored as "the" stopband.
    if kind == "lpf":
        return 1.1 * d["f_c"] <= f <= 2.0 * d["f_c"]
    if kind == "hpf":
        return 0.05 * d["f_c"] <= f <= 0.9 * d["f_c"]
    if kind == "bpf":
        f0 = d["f_c"]
        return ((0.5 * f0 <= f <= 0.9 * d["f_lo"])
                or (1.1 * d["f_hi"] <= f <= 1.5 * f0))
    return d["f_lo"] <= f <= d["f_hi"]


def metrics(design: dict, sw: dict | None = None) -> dict:
    sw = sw or sweep(design)
    kind = design["kind"]
    pb_s21, pb_s11, sb_s21 = [], [], []
    for f, a11, a21 in zip(sw["f_hz"], sw["s11_db"], sw["s21_db"]):
        if _in_pass(kind, f, design):
            pb_s21.append(a21)
            pb_s11.append(a11)
        if _in_stop(kind, f, design):
            sb_s21.append(a21)
    il = -min(pb_s21) if pb_s21 else 99.0            # IL = −S21 (dB)
    # worst (least negative) passband S21 magnitude as "max IL"
    il_max = -min(pb_s21) if pb_s21 else 99.0
    il_min = -max(pb_s21) if pb_s21 else 99.0
    rl = -max(pb_s11) if pb_s11 else 0.0             # RL = −S11
    rej = -max(sb_s21) if sb_s21 else 0.0            # rejection = −S21 stop
    return {
        "il_db": float(il_min),
        "il_max_db": float(il_max),
        "rl_db": float(rl),
        "rejection_db": float(rej),
        "q_u": float(design["q_u"]),
        "il_est_db": float(design["il_est_db"]),
        "n_pass": len(pb_s21),
        "n_stop": len(sb_s21),
    }


def design_params(design: dict) -> dict:
    """everythingRF digest checklist, JSON-safe.  Lumped topologies add a
    BOM-centric checklist (tolerances, inductor SRF/Q, dielectrics)."""
    d = design
    out = {
        "kind": d["kind"],
        "proto": d["proto"],
        "n": d["n"],
        "topology": d.get("topo", TOPOS[d["kind"]][0]),
        "f_c_mhz": d["f_c"] / 1e6,
        "f_lo_mhz": d["f_lo"] / 1e6,
        "f_hi_mhz": d["f_hi"] / 1e6,
        "g": d["g"],
        "z0_ohm": d.get("z0", 50.0),
        "w50_mm": d["w50_m"] * 1e3,
        "q_u": d["q_u"],
        "il_est_db": d["il_est_db"],
        "return_loss_target_db": RL_TARGET_DB,
        "rejection_target_dbc": REJ_TARGET_DBC,
        "rejection_offset": "10% from band edge",
        "rolloff_db_per_decade": 20.0 * d["n"] if d["proto"] == "butterworth"
        else None,
        "min_trace_mm": MIN_TRACE_M * 1e3,
        "launch_note": (
            "microstrip F.Cu over solid B.Cu (stripline-like return). "
            "Open CPW launches leak and raise the rejection floor — "
            "channelize (conductive cover) to recover ~90 dB floors"
        ),
        "etch_tol_um": 25.0,
        "solder_mask_note": (
            "solder mask on resonators lowers f slightly — leave the "
            "hairpin/stubs unmasked or retune"
        ),
        "source": SOURCE,
    }
    if d.get("topo") in LUMPED_TOPOS:
        out.update({
            "bom_count": len(d.get("components") or []),
            "tolerance_note": (
                "prototype g-values assume exact LC — use C0G/NP0 ceramics "
                "±1 % (no X7R in the signal path) and inductors ±2 %"
            ),
            "inductor_note": (
                "inductor SRF > 3× f_hi and Q > 40 at f_c; shielded SMD or "
                "air-core — DCR/SRF, not the microstrip IL_est, set the "
                "real insertion loss (il_est_db = 0 for lumped)"
            ),
            "mounting_note": (
                "0805 pad cascade, shunt elements via'd to a solid ground; "
                "keep shunt drops short — trace L adds to the shunt arm"
            ),
        })
        if d["topo"] in ("rc", "crc", "rl"):
            out["staging_note"] = (
                "passive RC/RL ladder: staged real-pole corners approximate "
                "a Butterworth −3 dB point but cannot form complex pole "
                "pairs — skirts are softer than an active/LC Butterworth of "
                "the same order; the sweep shows the true loaded response"
            )
        elif d["topo"] == "dc_lc":
            out["staging_note"] = (
                "capacitive J-inverters are first-order in the coupling "
                "(J ≈ ω₀C_c) — accurate for Δ ≲ 0.2"
            )
    elif d.get("topo") == "qw_tl":
        out["commensurate_note"] = (
            "Richards/Kuroda commensurate lines; impedances clamped to "
            f"[{Z_LOW:.0f}, 150] Ω for microstrip — "
            + ("clamp ACTIVE, response detuned (see z_clamped)"
               if d.get("z_clamped") else "no clamping needed"))
    elif d.get("topo") == "c_shunt":
        out["commensurate_note"] = (
            "combline-style loaded stubs; gap-capacitor J-inverters are "
            "first-order in the coupling (J ≈ ω₀C_c)")
    return out


# --- layout (mm) -------------------------------------------------------------


def _rect(x, y, w, h, layer="F.Cu") -> dict:
    return {"x": float(x), "y": float(y), "w": float(w), "h": float(h),
            "layer": layer}


def layout_mm(design: dict) -> dict:
    """Copper rectangles + pads in mm, origin at the filter centre."""
    if design.get("topo") in LUMPED_TOPOS:
        return _layout_lumped(design)
    kind = design["kind"]
    if kind == "bpf":
        return _layout_hairpin(design)
    return _layout_cascade(design)


def _layout_cascade(design: dict) -> dict:
    """Horizontal cascade of line/stub/gap sections (LPF/HPF/BSF)."""
    rects, vias, x = [], [], 0.0
    extra_pads = []
    n_shunt = 0
    y_stub_sign = 1.0
    max_y = 0.0
    for sec in design["sections"]:
        if sec["kind"] == "line":
            w = sec["w_m"] * 1e3
            ell = sec["len_m"] * 1e3
            rects.append(_rect(x + ell / 2.0, 0.0, ell, w))
            x += ell
        elif sec["kind"] in ("open_stub", "short_stub"):
            w = sec["w_m"] * 1e3
            ell = sec["len_m"] * 1e3
            cy = y_stub_sign * ell / 2.0
            rects.append(_rect(x, cy, w, ell))
            if sec["kind"] == "short_stub":
                vias.append({"x": x, "y": y_stub_sign * ell, "d": max(w, 0.6)})
            max_y = max(max_y, ell)
            y_stub_sign *= -1.0
        elif sec["kind"].startswith("shunt_"):
            # lumped shunt element (e.g. combline loading C): 0805 drop
            # to a via'd ground point
            n_shunt += 1
            extra_pads.append({"name": f"S{n_shunt}a", "x": x, "y": 0.0,
                               "w": 1.0, "h": 1.3})
            extra_pads.append({"name": f"S{n_shunt}b", "x": x, "y": 2.0,
                               "w": 1.0, "h": 1.3})
            rects.append(_rect(x, 3.0, 0.5, 2.0))
            vias.append({"x": x, "y": 3.8, "d": 0.8})
        elif sec["kind"] == "gap":
            g = max(sec.get("gap_m", 0.2e-3) * 1e3, 0.15)
            x += g
        elif sec["kind"] == "inverter":
            x += max(design.get("gaps_m", [0.4e-3])[0] * 1e3, 0.2)
    # centre on origin
    shift = x / 2.0
    for r in rects:
        r["x"] -= shift
    for v in vias:
        v["x"] -= shift
    for p in extra_pads:
        p["x"] -= shift
    w50 = design["w50_m"] * 1e3
    pads = [
        {"name": "1", "x": -shift, "y": 0.0, "w": max(w50, 1.0), "h": max(w50, 1.0)},
        {"name": "2", "x": x - shift, "y": 0.0, "w": max(w50, 1.0), "h": max(w50, 1.0)},
        *extra_pads,
    ]
    return _bbox({"rects": rects, "pads": pads, "vias": vias,
                  "kind": design["kind"]})


def _layout_lumped(design: dict) -> dict:
    """0805 SMD pad cascade for lumped designs.

    Series components sit inline along x as pad pairs (1.0 × 1.3 mm
    pads, 2.0 mm pitch = 0805); shunt components drop to an F.Cu GND
    strip below (via'd to the B.Cu plane of the wrapping board).
    Driven by the design's `components` BOM so pads map 1:1 to refs.
    """
    comps = design.get("components") or []
    pitch, gap, gnd_y = 2.0, 1.5, 3.6
    pad_w, pad_h = 1.0, 1.3
    lead = 2.0
    items = []
    cx = lead
    for c in comps:
        items.append((c, cx))
        cx += (pitch + gap) if c["role"] == "series" else gap
    end = max(cx - gap + lead, 2.0 * lead)
    rects = [_rect(end / 2.0, 0.0, end, 0.5),          # series track
             _rect(end / 2.0, gnd_y, end, 1.0)]        # GND strip
    pads, vias = [], []
    n_via = max(1, int(end // 4.0))
    for i in range(n_via + 1):
        vias.append({"x": end * i / n_via, "y": gnd_y, "d": 0.8})
    for c, x0 in items:
        if c["role"] == "series":
            pads.append({"name": f"{c['ref']}.1", "x": x0, "y": 0.0,
                         "w": pad_w, "h": pad_h})
            pads.append({"name": f"{c['ref']}.2", "x": x0 + pitch, "y": 0.0,
                         "w": pad_w, "h": pad_h})
        else:
            pads.append({"name": f"{c['ref']}.1", "x": x0, "y": 0.0,
                         "w": pad_w, "h": pad_h})
            pads.append({"name": f"{c['ref']}.2", "x": x0, "y": 2.0,
                         "w": pad_w, "h": pad_h})
            rects.append(_rect(x0, (2.0 + gnd_y) / 2.0, 0.5, gnd_y - 2.0))
    pads.append({"name": "1", "x": 0.0, "y": 0.0, "w": 1.5, "h": 1.5})
    pads.append({"name": "2", "x": end, "y": 0.0, "w": 1.5, "h": 1.5})
    shift = end / 2.0
    for r in rects:
        r["x"] -= shift
    for p in pads:
        p["x"] -= shift
    for v in vias:
        v["x"] -= shift
    return _bbox({"rects": rects, "pads": pads, "vias": vias,
                  "kind": design["kind"], "topo": design.get("topo")})


def _layout_hairpin(design: dict) -> dict:
    """Folded λ/2 U-resonators (hairpin), gaps from J-inverters."""
    n = design["n"]
    w = design["w50_m"] * 1e3
    lam2 = design.get("lam2_m", C0 / (design["f_c"] * 2.0)) * 1e3
    arm = lam2 / 2.0 * 0.9
    bar = max(3.0 * w, 2.0)
    gaps = [s * 1e3 for s in design.get("gaps_m", [0.4] * (n + 1))]
    feed = 4.0
    rects = []
    x = 0.0
    # port-1 feed
    rects.append(_rect(x + feed / 2.0, 0.0, feed, w))
    x += feed + gaps[0]
    for i in range(n):
        # U opening downward; adjacent U's share the coupling gap
        flip = i % 2                                  # stagger for coupling
        y0 = 0.0 if not flip else 0.0
        # left arm, bar, right arm
        rects.append(_rect(x + w / 2.0, y0 + arm / 2.0, w, arm))
        rects.append(_rect(x + (w + bar) / 2.0, y0 + arm, bar + w, w))
        rects.append(_rect(x + bar + w / 2.0, y0 + arm / 2.0, w, arm))
        x += w + bar + (gaps[i + 1] if i + 1 < len(gaps) else w)
    rects.append(_rect(x + feed / 2.0, 0.0, feed, w))
    x += feed
    shift = x / 2.0
    for r in rects:
        r["x"] -= shift
    pads = [
        {"name": "1", "x": -shift, "y": 0.0, "w": max(w, 1.0), "h": max(w, 1.0)},
        {"name": "2", "x": x - shift, "y": 0.0, "w": max(w, 1.0), "h": max(w, 1.0)},
    ]
    return _bbox({"rects": rects, "pads": pads, "vias": [],
                  "kind": "bpf"})


def _bbox(layout: dict) -> dict:
    xs, ys = [], []
    for r in layout["rects"]:
        xs += [r["x"] - r["w"] / 2, r["x"] + r["w"] / 2]
        ys += [r["y"] - r["h"] / 2, r["y"] + r["h"] / 2]
    for p in layout["pads"]:
        xs += [p["x"] - p["w"] / 2, p["x"] + p["w"] / 2]
        ys += [p["y"] - p["h"] / 2, p["y"] + p["h"] / 2]
    layout["bbox"] = {
        "xmin": min(xs), "xmax": max(xs),
        "ymin": min(ys), "ymax": max(ys),
    }
    return layout


def preview_from_layout(layout: dict) -> dict:
    """Same prims schema as kicad_gen.preview_from_sexpr."""
    prims = []
    for r in layout["rects"]:
        prims.append({
            "kind": "rect",
            "a": [r["x"] - r["w"] / 2, r["y"] - r["h"] / 2],
            "b": [r["x"] + r["w"] / 2, r["y"] + r["h"] / 2],
            "layer": r.get("layer", "F.Cu"),
        })
    for p in layout["pads"]:
        prims.append({
            "kind": "pad", "name": p["name"],
            "c": [p["x"], p["y"]], "size": [p["w"], p["h"]],
            "layer": "F.Cu",
        })
    for v in layout.get("vias") or []:
        prims.append({
            "kind": "circle", "c": [v["x"], v["y"]],
            "r": v["d"] / 2.0, "w": 0.15, "layer": "F.Cu",
        })
    return {"prims": prims, "bbox": layout["bbox"]}


def design_public(design: dict) -> dict:
    """JSON-safe design (sections kept, no numpy)."""
    out = {k: v for k, v in design.items() if k != "sections"}
    secs = []
    for s in design["sections"]:
        secs.append({k: (float(v) if isinstance(v, (int, float, np.floating))
                         else v)
                     for k, v in s.items()})
    out["sections"] = secs
    return out


# --- evolution ---------------------------------------------------------------


def _params_of(design: dict) -> np.ndarray:
    """Log-scale knobs: length and width (or gap) of every physical section."""
    vals = []
    for s in design["sections"]:
        if s["kind"] in ("line", "open_stub", "short_stub"):
            vals.append(math.log(max(s["len_m"], 1e-6)))
            vals.append(math.log(max(s["w_m"], MIN_TRACE_M)))
        elif s["kind"] == "inverter":
            vals.append(math.log(max(abs(s["j"]), 1e-6)))
        elif s["kind"] == "gap":
            vals.append(math.log(max(s["c_f"], 1e-18)))
    return np.asarray(vals, dtype=float)


def _apply_params(design: dict, vec: np.ndarray) -> dict:
    d = design_public(design)
    d["sections"] = [dict(s) for s in design["sections"]]
    it = iter(vec)
    eps_r, h_m, td = d["eps_r"], d["h_m"], d["tan_delta"]
    f_ref = d["f_c"]
    for s in d["sections"]:
        if s["kind"] in ("line", "open_stub", "short_stub"):
            s["len_m"] = math.exp(next(it))
            s["w_m"] = max(MIN_TRACE_M, math.exp(next(it)))
            s["z0"] = microstrip_z0(eps_r, s["w_m"], h_m)
            s["eps_eff"] = eps_eff(eps_r, s["w_m"], h_m)
            s["alpha"] = microstrip_alpha(f_ref, eps_r, s["w_m"], h_m, td,
                                          s["z0"])
        elif s["kind"] == "inverter":
            s["j"] = math.exp(next(it))
        elif s["kind"] == "gap":
            s["c_f"] = math.exp(next(it))
    return d


def _energy(design: dict) -> tuple[float, dict]:
    sw = sweep(design, n_points=41)
    m = metrics(design, sw)
    lay = layout_mm(design)
    b = lay["bbox"]
    span = max(b["xmax"] - b["xmin"], b["ymax"] - b["ymin"], 1e-3)
    e_il = max(0.0, m["il_max_db"] / 3.0)
    e_rl = max(0.0, (RL_TARGET_DB - m["rl_db"]) / RL_TARGET_DB)
    e_rej = max(0.0, (REJ_TARGET_DBC - m["rejection_db"]) / REJ_TARGET_DBC)
    min_w = min((s.get("w_m", 1.0) for s in design["sections"]
                 if "w_m" in s), default=1.0)
    e_dfm = max(0.0, (MIN_TRACE_M - min_w) / MIN_TRACE_M)
    e_size = span / 80.0                                  # ~80 mm reference
    terms = {"E_il": e_il, "E_rl": e_rl, "E_rej": e_rej,
             "E_dfm": e_dfm, "E_size": e_size}
    e = e_il + e_rl + e_rej + 0.25 * e_dfm + 0.15 * e_size
    return e, {**terms, **m}


def filter_sa(kind: str = "lpf", f_c: float = 2.45e9,
              f_lo: float | None = None, f_hi: float | None = None,
              n: int = 5, proto: str = "butterworth",
              eps_r: float = 4.4, h_m: float = 1.6e-3,
              tan_delta: float = 0.02, ripple_db: float = 0.1,
              hadamard_order: int = 32,
              T_start: float = 1.0, T_end: float = 0.02,
              cooling: float = 0.995, max_steps: int = 400,
              callback=None, stop_flag=None, live_params=None,
              rng=None) -> tuple[dict, dict]:
    """Hadamard-seeded SA over section lengths/widths of a PCB filter."""
    rng = rng or np.random.default_rng()
    base = design_filter(kind, f_c=f_c, f_lo=f_lo, f_hi=f_hi, n=n,
                         proto=proto, eps_r=eps_r, h_m=h_m,
                         tan_delta=tan_delta, ripple_db=ripple_db)
    x0 = _params_of(base)
    npar = len(x0)
    ho = int(hadamard_order)
    if ho < 4 or ho & (ho - 1):
        raise ValueError("hadamard_order must be a power of 2, ≥ 4")
    H = sylvester(ho)
    row = H[int(rng.integers(0, ho))].astype(float)
    # seed: ±8 % around the analytic design
    scale = 0.08 * row[:npar]
    if len(scale) < npar:
        scale = np.resize(scale, npar)
    x = x0 + scale

    def eval_x(vec):
        d = _apply_params(base, vec)
        e, terms = _energy(d)
        return e, terms, d

    e_cur, terms, des = eval_x(x)
    best_x, best_e, best_terms, best_des = x.copy(), e_cur, terms, des
    t = T_start
    steps = accepts = 0
    t0 = time.monotonic()

    while steps < max_steps and t > T_end:
        trial = x.copy()
        if rng.random() < 0.7:
            i = int(rng.integers(0, npar))
            trial[i] += rng.normal(0.0, 0.05)
        else:
            i, j = rng.integers(0, npar, size=2)
            trial[i], trial[j] = trial[j], trial[i]
        e_new, t_terms, t_des = eval_x(trial)
        if e_new < e_cur or rng.random() < math.exp(
                -(e_new - e_cur) / max(t, 1e-10)):
            x, e_cur, terms, des = trial, e_new, t_terms, t_des
            accepts += 1
            if e_new < best_e:
                best_x, best_e, best_terms, best_des = (
                    trial.copy(), e_new, t_terms, t_des)
        steps += 1
        t *= cooling
        if steps % 20 == 0:
            if live_params is not None:
                cl = live_params.get("cooling")
                if cl is not None:
                    cooling = float(cl)
            if callback is not None:
                callback({
                    "step": steps, "T": t, "E": e_cur, "best_E": best_e,
                    "accepts": accepts,
                    "il_db": best_terms.get("il_db"),
                    "rl_db": best_terms.get("rl_db"),
                    "rejection_db": best_terms.get("rejection_db"),
                    "layout": layout_mm(best_des),
                })
            if stop_flag is not None and stop_flag.is_set():
                break

    elapsed = time.monotonic() - t0
    best = {
        "design": design_public(best_des),
        "terms": {k: float(v) for k, v in best_terms.items()},
        "layout": layout_mm(best_des),
        "seed_row": {"order": ho},
    }
    info = {
        "steps": steps, "accepts": accepts, "best_E": best_e,
        "elapsed_s": elapsed, "best_design": best,
    }
    return best, info


# --- self-check --------------------------------------------------------------

if __name__ == "__main__":
    # Butterworth n=3: g1=1, g2=2, g3=1, g4=1
    g = prototype_g("butterworth", 3)
    assert abs(g[1] - 1.0) < 1e-12 and abs(g[2] - 2.0) < 1e-12
    assert abs(g[3] - 1.0) < 1e-12 and abs(g[4] - 1.0) < 1e-12
    print("PASS  Butterworth n=3 g = [1, 1, 2, 1, 1]")

    # Chebyshev n=3, 0.1 dB: g1≈1.0315, g2≈1.1474, g3≈1.0315
    gc = prototype_g("chebyshev", 3, 0.1)
    assert 1.02 < gc[1] < 1.04 and 1.13 < gc[2] < 1.16
    print(f"PASS  Chebyshev n=3 0.1 dB g1={gc[1]:.4f} g2={gc[2]:.4f}")

    # FR4 50 Ω width
    w50 = microstrip_width(4.4, 1.6e-3, 50.0) * 1e3
    assert 3.0 <= w50 <= 3.1, w50
    z_back = microstrip_z0(4.4, w50 * 1e-3, 1.6e-3)
    assert abs(z_back - 50.0) / 50.0 < 0.05, z_back
    print(f"PASS  microstrip 50 Ω: W={w50:.3f} mm, Z0 back={z_back:.1f} Ω")

    # LPF @ 1 GHz: pass at 0.5 GHz, reject at 2 GHz
    lpf = design_lpf(1e9, n=5, proto="butterworth")
    a = abcd_cascade(lpf["sections"], 0.5e9)
    _, t_lo = abcd_to_s(a)
    a = abcd_cascade(lpf["sections"], 2.0e9)
    _, t_hi = abcd_to_s(a)
    assert db20(t_lo) > -3.0, db20(t_lo)
    assert db20(t_hi) < -10.0, db20(t_hi)
    print(f"PASS  LPF n=5 @1 GHz: S21(0.5)={db20(t_lo):.1f} dB, "
          f"S21(2)={db20(t_hi):.1f} dB")

    # HPF: reject below, pass above
    hpf = design_hpf(1e9, n=5)
    _, t_lo = abcd_to_s(abcd_cascade(hpf["sections"], 0.4e9))
    _, t_hi = abcd_to_s(abcd_cascade(hpf["sections"], 2.0e9))
    assert db20(t_hi) > db20(t_lo), (db20(t_lo), db20(t_hi))
    print(f"PASS  HPF n=5 @1 GHz: S21(0.4)={db20(t_lo):.1f} dB, "
          f"S21(2)={db20(t_hi):.1f} dB")

    # BPF around 2.45 GHz
    bpf = design_bpf(2.3e9, 2.6e9, n=3)
    _, t0 = abcd_to_s(abcd_cascade(bpf["sections"], 2.45e9))
    _, tsb = abcd_to_s(abcd_cascade(bpf["sections"], 1.5e9))
    assert db20(t0) > db20(tsb), (db20(t0), db20(tsb))
    print(f"PASS  hairpin BPF 2.3–2.6 GHz: S21(2.45)={db20(t0):.1f} dB, "
          f"S21(1.5)={db20(tsb):.1f} dB")

    # BSF: notch at f0
    bsf = design_bsf(2.4e9, 2.5e9, n=3)
    _, t0 = abcd_to_s(abcd_cascade(bsf["sections"], 2.45e9))
    _, tsb = abcd_to_s(abcd_cascade(bsf["sections"], 1.5e9))
    assert db20(t0) < db20(tsb), (db20(t0), db20(tsb))
    print(f"PASS  stub BSF 2.4–2.5 GHz: S21(2.45)={db20(t0):.1f} dB, "
          f"S21(1.5)={db20(tsb):.1f} dB")

    # layout + preview
    for d in (lpf, hpf, bpf, bsf):
        lay = layout_mm(d)
        prev = preview_from_layout(lay)
        assert prev["prims"] and prev["bbox"]["xmax"] > prev["bbox"]["xmin"]
    print("PASS  layout/preview for LPF/HPF/BPF/BSF")

    # Q_u finite, IL estimate positive
    assert lpf["q_u"] > 10.0 and lpf["il_est_db"] > 0.0
    print(f"PASS  Q_u={lpf['q_u']:.0f}, IL_est={lpf['il_est_db']:.2f} dB")

    # SA: energy should be finite and a valid design comes back
    best, info = filter_sa("lpf", f_c=1e9, n=3, max_steps=40,
                           hadamard_order=16, rng=np.random.default_rng(2))
    assert info["steps"] == 40 and np.isfinite(info["best_E"])
    assert best["design"]["kind"] == "lpf"
    print(f"PASS  filter_sa LPF n=3 (40 steps): best_E={info['best_E']:.3f}, "
          f"IL={best['terms']['il_db']:.2f} dB, "
          f"RL={best['terms']['rl_db']:.1f} dB [{info['elapsed_s']:.2f} s]")

    # --- lumped / commensurate topologies --------------------------------

    def _s21(d, f):
        a = abcd_cascade(d["sections"], f)
        return db20(abcd_to_s(a, d.get("z0", 50.0))[1])

    # exact Butterworth n=3 values: z0 = 1 Ω, ωc = 1 rad/s →
    # series L1 = g1 Z0/ωc = 1 H, shunt C2 = g2/(Z0 ωc) = 2 F, L3 = 1 H
    d1 = design_lc("lpf", f_c=1.0 / (2.0 * math.pi), n=3, z0=1.0)
    assert d1["topo"] == "lc" and len(d1["components"]) == 3
    s = d1["sections"]
    assert s[0]["kind"] == "series_l" and abs(s[0]["l_h"] - 1.0) < 1e-12
    assert s[1]["kind"] == "shunt_c" and abs(s[1]["c_f"] - 2.0) < 1e-12
    assert s[2]["kind"] == "series_l" and abs(s[2]["l_h"] - 1.0) < 1e-12
    print("PASS  lc LPF n=3 exact: L1=1 H, C2=2 F, L3=1 H (z0=1 Ω, ωc=1)")

    # lc behavioral: n=5 Butterworth skirts ±30 dB at 2×/0.5× corner
    lc_lpf = design_lc("lpf", f_c=1e9, n=5)
    assert _s21(lc_lpf, 0.5e9) > -1.0 and _s21(lc_lpf, 2.0e9) < -20.0
    lc_hpf = design_lc("hpf", f_c=1e9, n=5)
    assert _s21(lc_hpf, 2.0e9) > -1.0 and _s21(lc_hpf, 0.5e9) < -20.0
    print(f"PASS  lc LPF/HPF n=5 @1 GHz: "
          f"{_s21(lc_lpf, 0.5e9):.2f}/{_s21(lc_lpf, 2e9):.1f} dB, "
          f"{_s21(lc_hpf, 2e9):.2f}/{_s21(lc_hpf, 0.5e9):.1f} dB")

    # lc bpf/bsf around f0 = √(f_lo f_hi)
    f0 = math.sqrt(2.3e9 * 2.6e9)
    lc_bpf = design_lc("bpf", f_lo=2.3e9, f_hi=2.6e9, n=3)
    assert _s21(lc_bpf, f0) > -1.0 and _s21(lc_bpf, 1.5e9) < -20.0
    f0b = math.sqrt(2.4e9 * 2.5e9)
    lc_bsf = design_lc("bsf", f_lo=2.4e9, f_hi=2.5e9, n=3)
    assert _s21(lc_bsf, f0b) < -15.0 and _s21(lc_bsf, 1.5e9) > -1.0
    print(f"PASS  lc BPF/BSF n=3: BPF S21(f0)={_s21(lc_bpf, f0):.2f} dB, "
          f"BSF notch {_s21(lc_bsf, f0b):.1f} dB")

    # dc_lc / c_shunt: passband centered at f0, stopband rejection
    dl = design_dc_lc(2.3e9, 2.6e9, n=3)
    assert _s21(dl, f0) > -2.0 and _s21(dl, 1.5e9) < -10.0
    cs = design_c_shunt(2.3e9, 2.6e9, n=3)
    assert _s21(cs, f0) > -4.0 and _s21(cs, 1.5e9) < -10.0
    print(f"PASS  dc_lc/c_shunt BPF n=3: S21(f0)={_s21(dl, f0):.2f}/"
          f"{_s21(cs, f0):.2f} dB, S21(1.5G)={_s21(dl, 1.5e9):.1f}/"
          f"{_s21(cs, 1.5e9):.1f} dB")

    # qw_tl: Kuroda LPF passes/rejects; commensurate BPF peaks at f0
    qw = design_qw_tl("lpf", f_c=1e9, n=5)
    assert _s21(qw, 0.5e9) > -2.0 and _s21(qw, 1.5e9) < -15.0
    qb = design_qw_tl("bpf", f_lo=2.3e9, f_hi=2.6e9, n=3)
    assert _s21(qb, f0) > _s21(qb, 0.8 * f0) + 6.0
    assert _s21(qb, f0) > _s21(qb, 1.25 * f0) + 6.0
    print(f"PASS  qw_tl: LPF {_s21(qw, 0.5e9):.2f}/{_s21(qw, 1.5e9):.1f} dB "
          f"(z_clamped={qw['z_clamped']}), BPF f0 {_s21(qb, f0):.2f} dB "
          f"(z_clamped={qb['z_clamped']})")

    # RC ladder: −3 dB point within ±35 % of target (loading pulls it
    # low — documented); HPF smoke (corner runs high in 50 Ω, see
    # design_rc docstring)
    rc = design_rc("lpf", f_c=1e3, n=3)
    sw_rc = sweep(rc, f_lo=10.0, f_hi=2e5, n_points=801)
    s_rc = np.asarray(sw_rc["s21_db"])
    f_rc = np.asarray(sw_rc["f_hz"])
    i3 = int(np.argmin(np.abs(s_rc - (s_rc[0] - 3.01))))
    assert abs(f_rc[i3] / 1e3 - 1.0) < 0.35, f_rc[i3]
    rch = design_rc("hpf", f_c=1e3, n=3)
    assert _s21(rch, 1e5) > _s21(rch, 1e3) > _s21(rch, 10.0)
    print(f"PASS  rc LPF n=3 @1 kHz: −3 dB at {f_rc[i3] / 1e3:.3f} kHz "
          f"(±35 % tol; DC level {s_rc[0]:.1f} dB), HPF monotone")

    # crc / rl smoke: build, sweep finite, low-pass behavior
    crc = design_crc(f_c=1e3, n=2)
    assert _s21(crc, 1e4) < _s21(crc, 100.0)
    rl = design_rl("lpf", f_c=1e3, n=3)
    assert _s21(rl, 1e4) < _s21(rl, 100.0)
    rlh = design_rl("hpf", f_c=1e3, n=3)
    assert _s21(rlh, 1e5) > _s21(rlh, 10.0)
    assert all(np.isfinite(sweep(d)["s21_db"]).all()
               for d in (crc, rl, rlh))
    print(f"PASS  crc/rl smoke: CRC 100 Hz→10 kHz {_s21(crc, 100.0):.1f}→"
          f"{_s21(crc, 1e4):.1f} dB, RL {_s21(rl, 100.0):.1f}→"
          f"{_s21(rl, 1e4):.1f} dB")

    # lumped layout + preview + JSON safety
    lay = layout_mm(lc_lpf)
    prev = preview_from_layout(lay)
    assert prev["prims"] and lay["vias"] and len(lay["pads"]) >= 2 * 5 + 2
    import json
    json.dumps(design_public(lc_lpf))
    lay_cs = layout_mm(cs)          # distributed + lumped loading caps
    assert preview_from_layout(lay_cs)["prims"]
    print(f"PASS  lumped layout/preview: {len(lay['pads'])} pads, "
          f"{len(lay['vias'])} vias; design_public JSON-safe")

    # topo validation + defaults keep existing behavior
    assert design_filter("lpf", f_c=1e9)["topo"] == "stepped"
    assert design_filter("hpf", f_c=1e9)["topo"] == "stub"
    assert design_filter("bpf", f_lo=2.3e9, f_hi=2.6e9)["topo"] == "hairpin"
    assert design_filter("bsf", f_lo=2.4e9, f_hi=2.5e9)["topo"] == "stub"
    try:
        design_filter("lpf", f_c=1e9, topo="bogus")
        raise AssertionError("bad topo accepted")
    except ValueError as e:
        assert "expected one of" in str(e)
    print("PASS  topo validation: defaults stepped/stub/hairpin/stub, "
          "bad topo raises")

    print("rf_filter self-check: all checks passed")
