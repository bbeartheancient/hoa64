"""Spherical-crown diffraction as a Hadamard search prior.

Formulas from Liu et al., "Spherical crown diffraction model by
occlusion utilizing for a curved holographic display", *Optics Express*
30 (2022) 465321 — ``~/spherical-crown.pdf``.

Rayleigh–Sommerfeld on concentric spheres (eqs. 1–7), optical-path
select function OPSF (eq. 8), crown cutoff d_m(φ) (eqs. 10–13), and
the 2-D FFT approximation (eqs. 17–19, 23):

    d        = √(r² + R² − 2 r R cosΔφ cosΔθ)
    cos α    = (R − r cosΔφ cosΔθ) / d          (OIP)
             = −(r − R cosΔφ cosΔθ) / d         (IOP)
    h(φ, θ)  = exp(j k d) / (j k d) · cos α
    d_m(φ)   = √(R² + r² + 2 R r cos(φ + φ_m))
    d_m-max  = √(R² + r² + 2 R r cos φ_m)
    h_appr   = h  if d ≥ d_m-max else 0
    U_d      = IFFT[ FFT(U_o) · FFT(h_appr) ]

Sampling (eq. 29): N_θ > 4π min(R,r)/λ , N_φ > 2 φ_m min(R,r)/λ.

``crown_sa`` / ``crown_ils`` minimise Gram F of H plus
``lam_c · F(Re U_d)`` so the crown channel is pulled toward an
orthogonal (Hadamard) reconstruction.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .hadamard import random_seed, verify


# Paper numerical section: r = 5 cm, R = 50 cm, φ_m = π/3.
# We keep the same ratio in dimensionless units.
R_DEFAULT = 10.0
R_INNER_DEFAULT = 1.0
PHI_M_DEFAULT = math.pi / 3.0


def path_length(r, R, dphi, dtheta):
    """Eq. (2): chord between concentric-sphere samples."""
    return np.sqrt(np.maximum(
        r * r + R * R - 2.0 * r * R * np.cos(dphi) * np.cos(dtheta),
        1e-18,
    ))


def cos_alpha(r, R, d, dphi, dtheta, mode: str = "oip"):
    """Eq. (4): obliquity.  ``mode`` is ``oip`` (object inside) or ``iop``."""
    c = np.cos(dphi) * np.cos(dtheta)
    if mode == "iop":
        return -(r - R * c) / d
    return (R - r * c) / d


def d_m(phi, r=R_INNER_DEFAULT, R=R_DEFAULT, phi_m=PHI_M_DEFAULT):
    """Eq. (11): crown path cutoff at latitude φ."""
    return np.sqrt(R * R + r * r + 2.0 * R * r * np.cos(phi + phi_m))


def d_m_max(r=R_INNER_DEFAULT, R=R_DEFAULT, phi_m=PHI_M_DEFAULT):
    """Eq. (23): max of d_m, attained at φ = 0."""
    return float(np.sqrt(R * R + r * r + 2.0 * R * r * np.cos(phi_m)))


def sample_min(r=R_INNER_DEFAULT, R=R_DEFAULT, lam=None, phi_m=PHI_M_DEFAULT):
    """Eq. (29): (N_θ, N_φ) Nyquist lower bounds."""
    m = min(float(R), float(r))
    if lam is None:
        lam = wavelength_for(n=64, r=r, R=R)
    return (4.0 * math.pi * m / lam, 2.0 * float(phi_m) * m / lam)


def wavelength_for(n: int, r=R_INNER_DEFAULT, R=R_DEFAULT):
    """λ just above the N_θ Nyquist bound for an n-sample azimuth ring."""
    m = min(float(R), float(r))
    return 4.0 * math.pi * m / max(int(n), 1) * 1.1


def psf(n: int, r=R_INNER_DEFAULT, R=R_DEFAULT, phi_m=PHI_M_DEFAULT,
        lam=None, mode: str = "oip", approx: bool = True) -> np.ndarray:
    """n×n complex PSF h_appr(φ, θ) (eqs. 5, 9, 17).

    Rows are latitude φ ∈ [−φ_m, φ_m], columns azimuth θ ∈ [−π, π).
    ``approx=True`` uses the constant d_m-max mask (eq. 17).
    """
    n = int(n)
    if lam is None:
        lam = wavelength_for(n, r=r, R=R)
    k = 2.0 * math.pi / float(lam)
    phi = np.linspace(-phi_m, phi_m, n)
    theta = np.linspace(-math.pi, math.pi, n, endpoint=False)
    dphi, dth = np.meshgrid(phi, theta, indexing="ij")
    d = path_length(r, R, dphi, dth)
    ca = cos_alpha(r, R, d, dphi, dth, mode=mode)
    h = np.exp(1j * k * d) / (1j * k * d) * ca
    cut = d_m_max(r=r, R=R, phi_m=phi_m) if approx else d_m(
        dphi, r=r, R=R, phi_m=phi_m)
    h = np.where(d >= cut, h, 0.0)
    return h.astype(np.complex128)


def propagate(U_o: np.ndarray, h: np.ndarray | None = None, **psf_kw) -> np.ndarray:
    """Eq. (19): U_d = IFFT[ FFT(U_o) · FFT(h) ]."""
    U = np.asarray(U_o, dtype=np.complex128)
    if h is None:
        h = psf(U.shape[0], **psf_kw)
    return np.fft.ifft2(np.fft.fft2(U) * np.fft.fft2(h))


def gram_energy(H: np.ndarray) -> float:
    A = np.asarray(H, dtype=np.float64)
    n = A.shape[0]
    G = A @ A.T
    off = G - n * np.eye(n)
    return float(np.sum(off * off) / 2.0)


def crown_energy(H: np.ndarray, h: np.ndarray | None = None, **psf_kw) -> float:
    """Gram energy of Re(U_d) — 0 when the crown reconstruction is orthogonal."""
    U = propagate(np.asarray(H, dtype=np.float64), h=h, **psf_kw)
    return gram_energy(np.real(U))


def analyze(H: np.ndarray, **psf_kw) -> dict:
    H = np.asarray(H)
    n = int(H.shape[0])
    hh = psf(n, **psf_kw)
    U = propagate(H.astype(np.float64), h=hh)
    return {
        "order": n,
        "F": gram_energy(H),
        "E_c": crown_energy(H, h=hh),
        "d_m_max": d_m_max(
            r=psf_kw.get("r", R_INNER_DEFAULT),
            R=psf_kw.get("R", R_DEFAULT),
            phi_m=psf_kw.get("phi_m", PHI_M_DEFAULT),
        ),
        "psf_power": float(np.mean(np.abs(hh) ** 2)),
        "Ud_power": float(np.mean(np.abs(U) ** 2)),
    }


def _swap_pair(H, rng) -> bool:
    n = H.shape[0]
    interior = H[1:, 1:]
    pos = np.argwhere(interior == 1)
    neg = np.argwhere(interior == -1)
    if len(pos) == 0 or len(neg) == 0:
        return False
    pi = int(rng.integers(0, len(pos)))
    ni = int(rng.integers(0, len(neg)))
    rp, cp = pos[pi]
    rn, cn = neg[ni]
    H[rp + 1, cp + 1] = -1
    H[rn + 1, cn + 1] = 1
    return True


def crown_sa(order, T_start=20.0, T_end=0.01, cooling=0.9995,
             max_steps=20000, lam_c=1.0, rng=None, callback=None,
             stop_flag=None, start=None, **psf_kw):
    """SA: Gram F(H) + lam_c · F(Re U_d) through the crown PSF."""
    rng = rng or np.random.default_rng()
    if start is not None:
        H = np.array(start, dtype=np.int8, copy=True)
        if H.shape != (order, order):
            raise ValueError(f"start shape {H.shape} != ({order}, {order})")
    else:
        H = random_seed(order, rng).astype(np.int8)
    hh = psf(order, **psf_kw)

    def tot(M):
        return gram_energy(M) + float(lam_c) * crown_energy(M, h=hh)

    E_cur = tot(H)
    best_H = H.copy()
    best_E = E_cur
    T = T_start
    steps = accepts = 0
    t0 = time.monotonic()

    while steps < max_steps and T > T_end:
        H_save = H.copy()
        if not _swap_pair(H, rng):
            H = H_save
            steps += 1
            if stop_flag is not None and steps % 500 == 0 and stop_flag.is_set():
                break
            continue
        E_new = tot(H)
        delta = E_new - E_cur
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            E_cur = E_new
            accepts += 1
            if E_cur < best_E:
                best_H = H.copy()
                best_E = E_cur
        else:
            H = H_save
        steps += 1
        T *= cooling
        if steps % 500 == 0:
            if callback is not None:
                callback({
                    "step": steps, "T": T, "E": E_cur, "best_E": best_E,
                    "accepts": accepts, "H": best_H,
                    "E_c": crown_energy(best_H, h=hh),
                })
            if stop_flag is not None and stop_flag.is_set():
                break
    return best_H, dict(
        steps=steps, accepts=accepts, best_E=best_E,
        E_c=crown_energy(best_H, h=hh),
        elapsed_s=time.monotonic() - t0,
        hadamard=(best_E < 1e-6),
    )


def crown_ils(order, T_start=20.0, cooling=0.9995, sa_steps=15000,
              restarts=5, time_budget=None, lam_c=1.0, rng=None,
              stop_flag=None, progress_callback=None, **psf_kw):
    """ILS wrapper around ``crown_sa`` (same contract as ``tile_ils``)."""
    rng = rng or np.random.default_rng()
    best_H = None
    best_f = None
    it = 0
    t0 = time.monotonic()
    while True:
        if stop_flag is not None and stop_flag.is_set():
            break
        if time_budget and time.monotonic() - t0 > time_budget:
            break
        if time_budget is None and it >= restarts:
            break
        H, _st = crown_sa(
            order, T_start=T_start, cooling=cooling, max_steps=sa_steps,
            lam_c=lam_c, rng=rng, callback=progress_callback,
            stop_flag=stop_flag, **psf_kw,
        )
        f = gram_energy(H)
        if best_H is None or f < best_f:
            best_H = H.copy()
            best_f = f
        if progress_callback is not None:
            progress_callback({
                "iter": it, "f": f, "best_f": best_f,
                "elapsed_s": time.monotonic() - t0,
            })
        it += 1
        T_start = min(30.0, T_start * 1.3)
    return best_H, best_f, (best_H is not None and verify(best_H))


if __name__ == "__main__":
    from .hadamard import sylvester
    dm = d_m_max()
    assert dm > 0
    # eq (23) at φ_m=π/3, r=1, R=10
    expect = math.sqrt(100 + 1 + 20 * math.cos(math.pi / 3))
    assert abs(dm - expect) < 1e-12, (dm, expect)
    h = psf(16)
    assert h.shape == (16, 16) and np.isfinite(h).all()
    H8 = sylvester(8)
    U = propagate(H8)
    assert U.shape == (8, 8)
    nt, np_ = sample_min(lam=wavelength_for(64))
    assert nt > 0 and np_ > 0
    Hc, info = crown_sa(4, max_steps=800, rng=np.random.default_rng(1))
    assert info["accepts"] > 0
    print(f"crown self-check OK  d_m-max={dm:.4f}  sa4 E={info['best_E']:.2f}")
