"""Holographic bound as a Hadamard search prior.

    S = A / (4 ℓₚ²)

S is the entropy of the region of space occupying volume V.  A is the
area of the surface that bounds V.  This module keeps that formula
explicit — it is not replaced by a dimensionless stand-in.

ℓₚ on this grid is the cell edge (one broken bond = one Planck area).
That is the discrete statement of "A in Planck units", not a rescaling
of the physics: setting ℓₚ = 1 is the unit choice, not an approximation.
A free ℓₚ only multiplies S and S_* by the same 1/ℓₚ², so it cannot
move the minima.

The search residual is therefore the scale-free ratio

    E_h = (S / S_* − 1)²  =  (A / A_* − 1)²

which is independent of ℓₚ and of the n⁴ blow-up of (S − S_*)².
S itself is always reported from the explicit formula.

On a ±1 matrix V is the n×n bulk (volume n² cells).  The surface that
bounds the polarity domains inside V is the domain-wall network
A = Σ flux_map(H).  Every Hadamard tried has mean wall density ½, so
A_* = n²/2 and S_* = n²/(8 ℓₚ²).  E_h = 0 exactly on that screen.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .hadamard import random_seed, verify
from .micromag import flux_map


#: Planck length in the units of the discrete grid (one cell).
LP = 1.0


def volume(H: np.ndarray) -> float:
    """Volume V of the region — n² cells of the ±1 bulk."""
    n = int(np.asarray(H).shape[0])
    return float(n * n)


def bounding_area(H: np.ndarray) -> float:
    """Area A, in Planck units, of the surface that bounds V.

    Domain walls are that surface: ``Σ flux_map(H)``.
    """
    return float(np.sum(flux_map(H)))


# Kept as an alias — older notes called A the "wall area".
wall_area = bounding_area


def entropy(H: np.ndarray, lp: float = LP) -> float:
    """Entropy S of the volume V: S = A / (4 ℓₚ²)."""
    lp = float(lp) if lp else LP
    return bounding_area(H) / (4.0 * lp * lp)


def S_star(n: int, lp: float = LP) -> float:
    """Hadamard saturation: A_* = n²/2 ⇒ S_* = n² / (8 ℓₚ²)."""
    lp = float(lp) if lp else LP
    return (int(n) * int(n)) / (8.0 * lp * lp)


def holo_energy(H: np.ndarray, lp: float = LP) -> float:
    """(S/S_* − 1)² — 0 iff the bounding area saturates the Hadamard bound.

    S and S_* still come from S = A/(4ℓₚ²).  The ratio drops ℓₚ and n²
    so the prior stays O(1) next to Gram F.
    """
    n = int(np.asarray(H).shape[0])
    star = S_star(n, lp=lp)
    if star == 0.0:
        return 0.0
    r = entropy(H, lp=lp) / star - 1.0
    return float(r * r)


def gram_energy(H: np.ndarray) -> float:
    """Off-diagonal Gram energy F = ½ Σ_{i≠j} (H Hᵀ)_{ij}²."""
    A = np.asarray(H, dtype=np.float64)
    G = A @ A.T
    n = A.shape[0]
    off = G - n * np.eye(n)
    return float(np.sum(off * off) / 2.0)


def analyze(H: np.ndarray, lp: float = LP) -> dict:
    n = int(np.asarray(H).shape[0])
    A = bounding_area(H)
    S = A / (4.0 * float(lp) * float(lp))
    star = S_star(n, lp=lp)
    return {
        "order": n,
        "V": float(n * n),
        "A": A,
        "S": S,
        "S_star": star,
        "lp": float(lp),
        "E_h": 0.0 if star == 0.0 else (S / star - 1.0) ** 2,
        "F": gram_energy(H),
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


def holo_sa(order, T_start=20.0, T_end=0.01, cooling=0.9995,
            max_steps=20000, lam_h=1.0, lp=LP, rng=None, callback=None,
            stop_flag=None, start=None):
    """SA: Gram F + lam_h · (S/S_* − 1)².  Polarity-preserving swaps."""
    rng = rng or np.random.default_rng()
    if start is not None:
        H = np.array(start, dtype=np.int8, copy=True)
        if H.shape != (order, order):
            raise ValueError(f"start shape {H.shape} != ({order}, {order})")
    else:
        H = random_seed(order, rng).astype(np.int8)

    def tot(M):
        return gram_energy(M) + float(lam_h) * holo_energy(M, lp=lp)

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
                    "S": entropy(best_H, lp=lp),
                    "S_star": S_star(order, lp=lp),
                    "E_h": holo_energy(best_H, lp=lp),
                })
            if stop_flag is not None and stop_flag.is_set():
                break
    return best_H, dict(
        steps=steps, accepts=accepts, best_E=best_E,
        S=entropy(best_H, lp=lp), S_star=S_star(order, lp=lp),
        E_h=holo_energy(best_H, lp=lp),
        elapsed_s=time.monotonic() - t0,
        hadamard=(best_E < 1e-6),
    )


def holo_ils(order, T_start=20.0, cooling=0.9995, sa_steps=15000,
             restarts=5, time_budget=None, lam_h=1.0, lp=LP, rng=None,
             stop_flag=None, progress_callback=None):
    """ILS wrapper around ``holo_sa`` (same contract as ``tile_ils``)."""
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
        H, _st = holo_sa(
            order, T_start=T_start, cooling=cooling, max_steps=sa_steps,
            lam_h=lam_h, lp=lp, rng=rng, callback=progress_callback,
            stop_flag=stop_flag,
        )
        f = gram_energy(H)
        if best_H is None or f < best_f:
            best_H = H.copy()
            best_f = f
        if progress_callback is not None:
            progress_callback({
                "iter": it, "f": f, "best_f": best_f,
                "S": entropy(best_H, lp=lp),
                "elapsed_s": time.monotonic() - t0,
            })
        it += 1
        T_start = min(30.0, T_start * 1.3)
    return best_H, best_f, (best_H is not None and verify(best_H))


if __name__ == "__main__":
    from .hadamard import sylvester
    for n in (4, 8, 16):
        H = sylvester(n)
        a = analyze(H)
        assert abs(a["S"] - a["S_star"]) < 1e-12, a
        assert a["E_h"] == 0.0, a
    Hr = np.random.default_rng(0).choice([-1, 1], (8, 8)).astype(np.int8)
    assert holo_energy(Hr) > 0.0
    # ℓₚ cancels in the residual — explicit S changes, E_h does not
    assert abs(holo_energy(Hr, lp=1.0) - holo_energy(Hr, lp=2.0)) < 1e-12
    assert abs(entropy(Hr, lp=1.0) - 4.0 * entropy(Hr, lp=2.0)) < 1e-12
    H, info = holo_sa(4, max_steps=2000, rng=np.random.default_rng(1))
    assert info["accepts"] > 0
    print(f"holo self-check OK  S(H8)={analyze(sylvester(8))['S']:.1f} "
          f"sa4 E={info['best_E']:.2f}")
