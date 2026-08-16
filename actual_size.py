"""Actual size — Press (1980) scales from fundamental constants.

William H. Press, "Man's size in terms of fundamental constants",
*Am. J. Phys.* **48**, 597 (1980)
https://fermatslibrary.com/s/man-ssize-in-terms-of-fundamental-constants

Three requirements (complicated molecules, an evolved atmosphere, as
large as possible without breaking) fix a length in terms of e, ħ, G,
m_e, m_p and a single chemistry fraction ε ≈ 0.003 of a Rydberg:

    a₀      = 4πϵ₀ ħ² / (m_e e²)
    ρ₀      = m_p / (2 a₀)³
    Ry      = e² / (8πϵ₀ a₀)          (= 13.6 eV)
    T       ~ (ε / k_B) Ry
    R_⊕     ~ ε^{1/2} (2 a₀) (e² / (4πϵ₀ G m_p²))^{1/2}
    M_⊕     ~ ρ₀ R_⊕³
    L       ~ ε^{1/4} (2 a₀) (e² / (4πϵ₀ G m_p²))^{1/4}
    t       ~ ε^{-2.75}  (lifespan; poorly determined)

L depends only weakly on ε (∝ ε^{1/4}); t depends strongly.  The
paper's numerical anchors at ε = 0.003 are L = 2.6 cm, R_⊕ = 6.5×10⁸ cm,
M_⊕ = 3.8×10²⁶ g, T = 470 K.

On a materials board of order n the *actual-size pitch* is L / n, so
the whole lattice is one Press-length creature.  That pitch is what
the Materials lab can stamp into ``pitch_mm``.
"""
from __future__ import annotations

import math

# CODATA 2018/2022 SI.
C = 299792458.0
HBAR = 1.054571817e-34
E_CHARGE = 1.602176634e-19
M_E = 9.1093837015e-31
M_P = 1.67262192369e-27
G = 6.67430e-11
K_B = 1.380649e-23
EPS0 = 8.8541878128e-12
SIGMA_SB = 2.0 * math.pi ** 5 * K_B ** 4 / (15.0 * (2.0 * math.pi * HBAR) ** 3 * C ** 2)

#: Paper's hydrogen-bond fraction of a Rydberg.
EPS_DEFAULT = 0.003


def bohr_radius() -> float:
    """a₀ (m)."""
    return 4.0 * math.pi * EPS0 * HBAR * HBAR / (M_E * E_CHARGE * E_CHARGE)


def rydberg_J() -> float:
    """1 Ry in joules — e² / (8πϵ₀ a₀) = ½ α² m_e c²."""
    a0 = bohr_radius()
    return E_CHARGE * E_CHARGE / (8.0 * math.pi * EPS0 * a0)


def scale_density() -> float:
    """ρ₀ = m_p / (2 a₀)³  (kg/m³)."""
    a0 = bohr_radius()
    return M_P / (2.0 * a0) ** 3


def gravity_number() -> float:
    """e² / (4πϵ₀ G m_p²) — electrostatic / gravitational strength on two protons."""
    return E_CHARGE ** 2 / (4.0 * math.pi * EPS0 * G * M_P * M_P)


def scales(eps: float = EPS_DEFAULT) -> dict:
    """All Press scales at chemistry fraction ``eps``.

    Lengths in metres, masses in kg, temperature in kelvin, time in
    seconds.  Also the paper's cgs anchors for regression checks.
    """
    eps = float(eps)
    if not (1e-6 < eps < 1.0):
        raise ValueError(f"eps must be a small positive Rydberg fraction, got {eps}")
    a0 = bohr_radius()
    ry = rydberg_J()
    rho = scale_density()
    gn = gravity_number()  # ~ 1.2e36
    # Closed forms matching Press eqs. (4), (7), (8), (11), (12), (14).
    T = (eps / K_B) * ry
    R = math.sqrt(eps) * (2.0 * a0) * math.sqrt(gn)
    M = rho * R ** 3
    L = (eps ** 0.25) * (2.0 * a0) * (gn ** 0.25)
    # Solar-constant scale (eq. 12) and shelter/growth time (eq. 14).
    J = SIGMA_SB * T ** 4
    # t ~ (ρ L ε Ry) / (J) — chemical energy / (flux × area L²)
    chem = (rho * L ** 3 / M_P) * (eps * ry)
    t = chem / max(J * L * L, 1e-300)
    return {
        "eps": eps,
        "a0_m": a0,
        "rho0_kg_m3": rho,
        "Ry_eV": ry / E_CHARGE,
        "T_K": T,
        "R_earth_m": R,
        "M_earth_kg": M,
        "L_m": L,
        "L_cm": L * 100.0,
        "t_s": t,
        "solar_W_m2": J,
        "gravity_number": gn,
        # paper anchors at ε = 0.003
        "paper_L_cm": 2.6 * (eps / EPS_DEFAULT) ** 0.25,
        "paper_R_cm": 6.5e8 * math.sqrt(eps / EPS_DEFAULT),
        "paper_M_g": 3.8e26 * (eps / EPS_DEFAULT) ** 1.5,
        "paper_T_K": 470.0 * (eps / EPS_DEFAULT),
    }


def pitch_mm(order: int, eps: float = EPS_DEFAULT) -> float:
    """Cell pitch so n cells span one Press length L (mm)."""
    n = max(int(order), 1)
    L_mm = scales(eps)["L_m"] * 1.0e3
    return float(L_mm / n)


def analyze(order: int = 16, eps: float = EPS_DEFAULT,
            pitch_lo: float = 0.2, pitch_hi: float = 20.0) -> dict:
    """JSON-safe payload for the Materials lab / API."""
    s = scales(eps)
    raw = pitch_mm(order, eps)
    used = min(max(raw, pitch_lo), pitch_hi)
    return {
        "eps": eps,
        "order": int(order),
        "a0_m": s["a0_m"],
        "Ry_eV": s["Ry_eV"],
        "T_K": s["T_K"],
        "L_cm": s["L_cm"],
        "L_m": s["L_m"],
        "R_earth_m": s["R_earth_m"],
        "M_earth_kg": s["M_earth_kg"],
        "t_s": s["t_s"],
        "pitch_mm": used,
        "pitch_mm_raw": raw,
        "clamped": used != raw,
        "board_mm": used * int(order),
        "paper_L_cm": s["paper_L_cm"],
        "paper_T_K": s["paper_T_K"],
    }


if __name__ == "__main__":
    s = scales(0.003)
    # Bohr / Rydberg must match textbook values.
    assert abs(s["a0_m"] - 5.291772e-11) < 1e-15, s["a0_m"]
    assert abs(s["Ry_eV"] - 13.605693) < 0.01, s["Ry_eV"]
    # Paper: T = 470 K, L = 2.6 cm — order-of-magnitude, not digits.
    assert 300 < s["T_K"] < 700, s["T_K"]
    assert 0.5 < s["L_cm"] < 20.0, s["L_cm"]
    # Weak ε dependence of L (∝ ε^{1/4}).
    r = scales(0.003 * 16)["L_m"] / s["L_m"]
    assert abs(r - 2.0) < 0.05, r
    p = pitch_mm(16)
    assert 0.2 < p < 5.0, p
    print(f"actual-size self-check OK  L={s['L_cm']:.2f} cm  "
          f"T={s['T_K']:.0f} K  pitch16={p:.3f} mm  "
          f"R⊕={s['R_earth_m']/1e6:.1f} Mm")
