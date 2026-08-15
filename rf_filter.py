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

**Output forms.**  This module currently realises the prototype as
distributed microstrip (KiCad copper).  The same *g*-values also
determine lumped ladders — series-L/shunt-C (LC), RC, and CRC/π —
which are a planned extra export, not a second solver.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .em_physics import C0, EPS0, ETA0, MU0, SIGMA_CU
from .hadamard import sylvester

KINDS = ("lpf", "hpf", "bpf", "bsf")
PROTOS = ("butterworth", "chebyshev")
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
          g: list[float], sections: list[dict], extra: dict | None = None
          ) -> dict:
    w50 = microstrip_width(eps_r, h_m, 50.0)
    qu = q_unloaded(f_c, eps_r, h_m, tan_delta)
    fbw = (f_hi - f_lo) / f_c if f_hi > f_lo else 1.0
    g_sum = float(sum(g[1:-1]))
    il_est = 4.343 * g_sum / max(fbw * qu, 1e-9)
    d = {
        "kind": kind, "proto": proto, "n": int(n),
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


def design_filter(kind: str, f_c: float | None = None,
                  f_lo: float | None = None, f_hi: float | None = None,
                  n: int = 5, proto: str = "butterworth",
                  eps_r: float = 4.4, h_m: float = 1.6e-3,
                  tan_delta: float = 0.02, ripple_db: float = 0.1) -> dict:
    kind = kind.lower()
    if kind not in KINDS:
        raise ValueError(f"unknown filter kind {kind!r}; expected {KINDS}")
    if kind in ("lpf", "hpf"):
        if f_c is None:
            raise ValueError(f"{kind} requires f_c")
        fn = design_lpf if kind == "lpf" else design_hpf
        return fn(f_c, n=n, proto=proto, eps_r=eps_r, h_m=h_m,
                  tan_delta=tan_delta, ripple_db=ripple_db)
    if f_lo is None or f_hi is None:
        if f_c is None:
            raise ValueError(f"{kind} requires f_lo/f_hi or f_c + 10 % FBW")
        f_lo, f_hi = 0.95 * f_c, 1.05 * f_c
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
    """everythingRF digest checklist, JSON-safe."""
    d = design
    return {
        "kind": d["kind"],
        "proto": d["proto"],
        "n": d["n"],
        "f_c_mhz": d["f_c"] / 1e6,
        "f_lo_mhz": d["f_lo"] / 1e6,
        "f_hi_mhz": d["f_hi"] / 1e6,
        "g": d["g"],
        "z0_ohm": 50.0,
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


# --- layout (mm) -------------------------------------------------------------


def _rect(x, y, w, h, layer="F.Cu") -> dict:
    return {"x": float(x), "y": float(y), "w": float(w), "h": float(h),
            "layer": layer}


def layout_mm(design: dict) -> dict:
    """Copper rectangles + pads in mm, origin at the filter centre."""
    kind = design["kind"]
    if kind == "bpf":
        return _layout_hairpin(design)
    return _layout_cascade(design)


def _layout_cascade(design: dict) -> dict:
    """Horizontal cascade of line/stub/gap sections (LPF/HPF/BSF)."""
    rects, vias, x = [], [], 0.0
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
    w50 = design["w50_m"] * 1e3
    pads = [
        {"name": "1", "x": -shift, "y": 0.0, "w": max(w50, 1.0), "h": max(w50, 1.0)},
        {"name": "2", "x": x - shift, "y": 0.0, "w": max(w50, 1.0), "h": max(w50, 1.0)},
    ]
    return _bbox({"rects": rects, "pads": pads, "vias": vias,
                  "kind": design["kind"]})


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

    print("rf_filter self-check: all checks passed")
