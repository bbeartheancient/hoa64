"""Brillouin-zone folding as a Hadamard / materials prior.

Guan, Zhang, Xiao, Wang, Yu, Liao, "High-Q intrinsic and nonlinear
chirality in planar silicon metasurfaces enabled by Brillouin zone
folding", *Opt. Express* **34**, 31398 (2026)
https://doi.org/10.1364/OE.609157

A gap perturbation doubles the real-space period (p = 2 p₀) and halves
the first Brillouin zone.  Guided-mode dispersion then satisfies

    f(k_x) = f(k_x + π/p₀) ≈ f₀(k_x + π/p₀)

so the X-point mode at k = π/p₀ folds onto Γ (k = 0) and becomes a
leaky quasi-guided mode.  On an n×n ±1 lattice (n even) the discrete
2-D DFT plays the role of the photonic band: the four high-symmetry
corners are

    Γ = F[0, 0],   X = F[0, n/2],   Y = F[n/2, 0],   M = F[n/2, n/2].

Folding the zone is the 2×2 block sum (the supercell DFT).  Coherent
addition of those four corners is the X→Γ transfer the paper uses to
inherit the parent guided-mode Q.

Circular dichroism (eq. 1 of the paper) is reported on the 2-layer
weave by reading F.Cu / B.Cu neighbour transitions as the four
T_{ij} channels.

``bzf_sa`` / ``bzf_ils`` minimise Gram F plus
``lam_b · (1 − fold_coherence)``.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .hadamard import random_seed, verify


def dft2(H: np.ndarray) -> np.ndarray:
    """Unitary-ish 2-D DFT (unnormalized, matching numpy.fft)."""
    return np.fft.fft2(np.asarray(H, dtype=np.float64))


def high_symmetry(F: np.ndarray) -> dict:
    """Γ, X, Y, M samples of an even-order 2-D DFT."""
    F = np.asarray(F)
    n = int(F.shape[0])
    if n < 2 or n % 2:
        z = complex(F[0, 0]) if F.size else 0j
        return {"gamma": z, "X": 0j, "Y": 0j, "M": 0j, "n": n}
    h = n // 2
    return {
        "gamma": complex(F[0, 0]),
        "X": complex(F[0, h]),
        "Y": complex(F[h, 0]),
        "M": complex(F[h, h]),
        "n": n,
    }


def fold_zone(F: np.ndarray) -> np.ndarray:
    """Halve the BZ: 2×2 block sum, X/M fold onto the reduced Γ.

    For period doubling p = 2 p₀ the reduced-zone transform is
    F_Γ' = F_Γ + F_X + F_Y + F_M on the four corners, and likewise
    for every reduced-zone k.
    """
    F = np.asarray(F)
    n = int(F.shape[0])
    if n < 2 or n % 2:
        return F.copy()
    h = n // 2
    return F[:h, :h] + F[:h, h:] + F[h:, :h] + F[h:, h:]


def fold_coherence(H: np.ndarray) -> float:
    """|Γ+X+Y+M|² / (4 Σ |corner|²) ∈ [0, 1].

    1 when the four high-symmetry amplitudes add in phase (clean X→Γ
    fold); 1/4 when they are random-phased and equal.
    """
    hs = high_symmetry(dft2(H))
    corners = np.array([hs["gamma"], hs["X"], hs["Y"], hs["M"]],
                       dtype=np.complex128)
    denom = 4.0 * float(np.sum(np.abs(corners) ** 2))
    if denom <= 0.0:
        return 0.0
    return float(np.abs(np.sum(corners)) ** 2 / denom)


def x_to_gamma(H: np.ndarray) -> float:
    """Fraction of (X,Y) power relative to (Γ, X, Y, M)."""
    hs = high_symmetry(dft2(H))
    corners = np.array([hs["gamma"], hs["X"], hs["Y"], hs["M"]])
    tot = float(np.sum(np.abs(corners) ** 2))
    if tot <= 0.0:
        return 0.0
    return float((np.abs(hs["X"]) ** 2 + np.abs(hs["Y"]) ** 2) / tot)


def circular_dichroism(H: np.ndarray) -> float:
    """Paper eq. (1) on the 2-layer weave.

    Map L ↔ H=+1 (F.Cu) and R ↔ H=−1 (B.Cu).  T_{ij} is the fraction
    of horizontal bonds from an i-cell to a j-neighbour.  CD ∈ [−1, 1].
    """
    A = np.asarray(H, dtype=np.int8)
    L = A == 1
    R = A == -1
    Ln = np.roll(L, -1, axis=1)
    Rn = np.roll(R, -1, axis=1)
    n2 = float(A.size)
    Tll = float(np.count_nonzero(L & Ln)) / n2
    Trr = float(np.count_nonzero(R & Rn)) / n2
    Trl = float(np.count_nonzero(L & Rn)) / n2
    Tlr = float(np.count_nonzero(R & Ln)) / n2
    num = (Tll + Trl) - (Trr + Tlr)
    den = (Tll + Trl) + (Trr + Tlr)
    if den == 0.0:
        return 0.0
    return float(num / den)


def q_factor(dx: float, q0: float = 2200.0, dx0: float = 0.02) -> float:
    """Paper: Q drops as the gap perturbation dx grows (Fig. 2a).

    ``dx`` is a dimensionless seam fraction (0 = unperturbed GM).
    The 2200 figure is their optimized QGM at dx = 20 nm / p = 850 nm.
    """
    dx = max(float(dx), 0.0)
    return float(q0 * dx0 / (dx + dx0))


def seam_dx(H: np.ndarray) -> float:
    """Mean |ΔH| on even/odd column seams — the discrete gap dx."""
    A = np.asarray(H, dtype=np.int8)
    n = A.shape[1]
    if n < 2:
        return 0.0
    even = A[:, 0::2]
    odd = A[:, 1::2]
    m = min(even.shape[1], odd.shape[1])
    if m == 0:
        return 0.0
    return float(np.mean(np.abs(even[:, :m].astype(np.int16)
                                - odd[:, :m].astype(np.int16))) / 2.0)


def period_double(H: np.ndarray) -> np.ndarray:
    """Configuration ii: double the x-period by repeating each column.

    The Brillouin zone halves; the new X-point is the old Γ of the
    half-period cell.
    """
    A = np.asarray(H, dtype=np.int8)
    return np.repeat(A, 2, axis=1)


def gram_energy(H: np.ndarray) -> float:
    A = np.asarray(H, dtype=np.float64)
    n = A.shape[0]
    G = A @ A.T
    off = G - n * np.eye(n)
    return float(np.sum(off * off) / 2.0)


def bzf_energy(H: np.ndarray) -> float:
    """1 − fold_coherence — 0 on a perfectly phase-aligned fold."""
    return 1.0 - fold_coherence(H)


def analyze(H: np.ndarray) -> dict:
    H = np.asarray(H)
    n = int(H.shape[0])
    F = dft2(H)
    hs = high_symmetry(F)
    dx = seam_dx(H)
    return {
        "order": n,
        "F": gram_energy(H),
        "fold_coherence": fold_coherence(H),
        "x_to_gamma": x_to_gamma(H),
        "E_b": bzf_energy(H),
        "dx": dx,
        "Q": q_factor(dx),
        "CD": circular_dichroism(H),
        "gamma": abs(hs["gamma"]),
        "X": abs(hs["X"]),
        "Y": abs(hs["Y"]),
        "M": abs(hs["M"]),
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


def bzf_sa(order, T_start=20.0, T_end=0.01, cooling=0.9995,
           max_steps=20000, lam_b=1.0, rng=None, callback=None,
           stop_flag=None, start=None):
    """SA: Gram F + lam_b · (1 − fold_coherence)."""
    rng = rng or np.random.default_rng()
    if start is not None:
        H = np.array(start, dtype=np.int8, copy=True)
        if H.shape != (order, order):
            raise ValueError(f"start shape {H.shape} != ({order}, {order})")
    else:
        H = random_seed(order, rng).astype(np.int8)

    def tot(M):
        return gram_energy(M) + float(lam_b) * bzf_energy(M)

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
                    "fold_coherence": fold_coherence(best_H),
                    "CD": circular_dichroism(best_H),
                })
            if stop_flag is not None and stop_flag.is_set():
                break
    return best_H, dict(
        steps=steps, accepts=accepts, best_E=best_E,
        fold_coherence=fold_coherence(best_H),
        CD=circular_dichroism(best_H),
        elapsed_s=time.monotonic() - t0,
        hadamard=(best_E < 1e-6),
    )


def bzf_ils(order, T_start=20.0, cooling=0.9995, sa_steps=15000,
            restarts=5, time_budget=None, lam_b=1.0, rng=None,
            stop_flag=None, progress_callback=None):
    """ILS wrapper around ``bzf_sa`` (same contract as ``tile_ils``)."""
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
        H, _st = bzf_sa(
            order, T_start=T_start, cooling=cooling, max_steps=sa_steps,
            lam_b=lam_b, rng=rng, callback=progress_callback,
            stop_flag=stop_flag,
        )
        f = gram_energy(H)
        if best_H is None or f < best_f:
            best_H = H.copy()
            best_f = f
        if progress_callback is not None:
            progress_callback({
                "iter": it, "f": f, "best_f": best_f,
                "fold_coherence": fold_coherence(best_H),
                "elapsed_s": time.monotonic() - t0,
            })
        it += 1
        T_start = min(30.0, T_start * 1.3)
    return best_H, best_f, (best_H is not None and verify(best_H))


if __name__ == "__main__":
    from .hadamard import sylvester
    H8 = sylvester(8)
    a = analyze(H8)
    assert 0.0 <= a["fold_coherence"] <= 1.0
    assert 0.0 <= a["Q"]
    Fd = dft2(H8)
    folded = fold_zone(Fd)
    assert folded.shape == (4, 4)
    # period-doubling a half-order cell halves the zone
    H4 = sylvester(4)
    Hd = period_double(H4)
    assert Hd.shape == (4, 8)
    Hs, info = bzf_sa(4, max_steps=800, rng=np.random.default_rng(1))
    assert info["accepts"] > 0
    print(f"brillouin self-check OK  coh(H8)={a['fold_coherence']:.3f} "
          f"CD={a['CD']:.3f} Q={a['Q']:.0f} sa4 E={info['best_E']:.2f}")
