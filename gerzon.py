"""Gerzon AB module as a Hadamard-cell navigator.

Source
------
Michael Gerzon, "Ambisonics. Part two: Studio techniques",
*Studio Sound* 17 (August 1975) pp. 24–26, 28, 40.
https://www.michaelgerzonphotos.org.uk/articles/Ambisonics%202.pdf

The AB module converts square A-format corners
(``L_B``, ``L_F``, ``R_F``, ``R_B`` at 135° / 45° / 315° / 225°)
into B-format.  Outputs in WXYZ order, Gerzon's ½ scale:

    W = ½ (L_B + L_F + R_F + R_B)
    X = ½ (−L_B + L_F + R_F − R_B)
    Y = ½ (L_B + L_F − R_F − R_B)
    Z = ½ (−L_B + L_F − R_F + R_B)

W, X, Y are the three-channel **horizontal lock**.  Z is Gerzon's
monophonic height channel (upward figure-of-eight).  For horizontal-only
A-format Z is identically zero and may be dropped; the same linear
network converts B back to A.

H₄
--
Integer kernel ``AB_H`` (rows WXYZ, columns L_B, L_F, R_F, R_B) is a
Hadamard matrix.  Swapping the L_F and L_B columns yields Sylvester
H₄, so (H₄ / 2) is involutory.  That is the "FOA is a normalized H₄"
remark in ``hadamard.py``, written in Gerzon's basis, not later SN3D.

2 × 2 cells
-----------
A matrix tile is read with row 0 = front, column 0 = left:

    H[i,   j  ] = L_F (45°)     H[i,   j+1] = R_F (315°)
    H[i+1, j  ] = L_B (135°)    H[i+1, j+1] = R_B (225°)

Then Z = ½ ((L_F + R_B) − (L_B + R_F)) is the **diagonal contrast**.
L_F (45°) and R_B (225°) are antipodes: they add in W and Z and cancel
in the odd figure-8s X, Y.  On a ±1 cell the integer ``Z_int = 2Z``
takes three values:

    |Z_int| = 0   WXY-cohesive (even parity, axis-aligned)
    |Z_int| = 2   H₂ tile (the Hadamard atom)
    |Z_int| = 4   diagonal wall — the 45°/225° pair is fully occupied

Aligned (stride-2) 2 × 2 blocks of a Sylvester matrix are all H₂.
Overlapping (stride-1) Z peaks on the Kronecker seams — the boundary
between cohesive H₂ states.

Solver
------
``gerzon_sa`` / ``gerzon_ils`` minimise the usual Gram off-diagonal
energy plus ``lam_z · mean((|Z_int| − 2)²)`` over stride-2 tiles, so
the search is pulled toward H₂ tessellation in Gerzon's basis.
Proposals include a dedicated flip of the 45°/225° pair.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .hadamard import random_seed, verify

#: Integer AB kernel, rows WXYZ, columns (L_B, L_F, R_F, R_B).
#: B = (AB_H @ A) / 2.  This *is* a Hadamard matrix.
AB_H = np.array([
    [ 1,  1,  1,  1],
    [-1,  1,  1, -1],
    [ 1,  1, -1, -1],
    [-1,  1, -1,  1],
], dtype=np.int8)

#: Speaker azimuths (degrees, Ambix: 0 = front, +90 = left) in AB column order.
AB_AZIMUTH = {"Lb": 135.0, "Lf": 45.0, "Rf": 315.0, "Rb": 225.0}

#: Antipodal pair that cancels in X,Y and adds in W,Z.
CANCEL_PAIR = ("Lf", "Rb")  # 45° and 225°


def sylvester_h4() -> np.ndarray:
    """Sylvester H₄ — ``AB_H`` with the L_F and L_B columns swapped."""
    H = AB_H.copy()
    H[:, [0, 1]] = H[:, [1, 0]]
    return H


def ab_encode(Lb, Lf, Rf, Rb):
    """A-format corners → B-format ``(W, X, Y, Z)`` at Gerzon's ½ scale.

    Arguments broadcast.  ±1 corners give values in {−2, −1, 0, 1, 2}.
    """
    Lb = np.asarray(Lb, dtype=np.float64)
    Lf = np.asarray(Lf, dtype=np.float64)
    Rf = np.asarray(Rf, dtype=np.float64)
    Rb = np.asarray(Rb, dtype=np.float64)
    W = 0.5 * (Lb + Lf + Rf + Rb)
    X = 0.5 * (-Lb + Lf + Rf - Rb)
    Y = 0.5 * (Lb + Lf - Rf - Rb)
    Z = 0.5 * (-Lb + Lf - Rf + Rb)
    return W, X, Y, Z


def ab_decode(W, X, Y, Z):
    """B-format → A-format corners.  Inverse of ``ab_encode`` (Hᵀ / 2)."""
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    # A = (AB_H.T @ B) / 2
    Lb = 0.5 * (W - X + Y - Z)
    Lf = 0.5 * (W + X + Y + Z)
    Rf = 0.5 * (W + X - Y - Z)
    Rb = 0.5 * (W - X - Y + Z)
    return Lb, Lf, Rf, Rb


def _corners(H: np.ndarray, stride: int = 1):
    """L_F, R_F, L_B, R_B views of every stride×stride-aligned 2 × 2."""
    A = np.asarray(H, dtype=np.int8)
    return (A[:-1:stride, :-1:stride],   # Lf
            A[:-1:stride, 1::stride],    # Rf
            A[1::stride, :-1:stride],    # Lb
            A[1::stride, 1::stride])     # Rb


def decode_cells(H: np.ndarray, stride: int = 1) -> dict:
    """Gerzon AB on every 2 × 2 cell.

    Returns float64 maps ``W,X,Y,Z`` of shape ``(⌈(n-1)/stride⌉, …)``
    plus integer ``Z_int = 2Z`` and counts ``n_cohesive`` / ``n_h2`` /
    ``n_wall``.
    """
    Lf, Rf, Lb, Rb = _corners(H, stride)
    W, X, Y, Z = ab_encode(Lb, Lf, Rf, Rb)
    Z_int = np.rint(2.0 * Z).astype(np.int8)
    az = np.abs(Z_int)
    return {
        "W": W, "X": X, "Y": Y, "Z": Z, "Z_int": Z_int,
        "stride": int(stride),
        "n_cells": int(Z_int.size),
        "n_cohesive": int(np.count_nonzero(az == 0)),
        "n_h2": int(np.count_nonzero(az == 2)),
        "n_wall": int(np.count_nonzero(az == 4)),
    }


def z_energy(H: np.ndarray, stride: int = 2) -> float:
    """mean((|Z_int| − 2)²) over aligned tiles — 0 iff every tile is H₂."""
    Lf, Rf, Lb, Rb = _corners(H, stride)
    Z_int = (-Lb + Lf - Rf + Rb).astype(np.int16)
    if Z_int.size == 0:
        return 0.0
    return float(np.mean((np.abs(Z_int) - 2) ** 2))


def gram_energy(H: np.ndarray) -> float:
    """Off-diagonal Gram energy F = ½ Σ_{i≠j} (H Hᵀ)_{ij}²."""
    A = np.asarray(H, dtype=np.float64)
    G = A @ A.T
    n = A.shape[0]
    off = G - n * np.eye(n)
    return float(np.sum(off * off) / 2.0)


def analyze(H: np.ndarray) -> dict:
    """Stride-2 tessellation stats + stride-1 wall field (no search)."""
    aligned = decode_cells(H, stride=2)
    overlap = decode_cells(H, stride=1)
    return {
        "order": int(np.asarray(H).shape[0]),
        "F": gram_energy(H),
        "E_z": z_energy(H, stride=2),
        "aligned": {k: aligned[k] for k in
                    ("n_cells", "n_cohesive", "n_h2", "n_wall", "stride")},
        "overlap": {k: overlap[k] for k in
                    ("n_cells", "n_cohesive", "n_h2", "n_wall", "stride")},
        "Z_wall": overlap["Z_int"],
    }


def _flip_tile(H, i, j) -> None:
    H[i, j] *= -1
    H[i, j + 1] *= -1
    H[i + 1, j] *= -1
    H[i + 1, j + 1] *= -1


def flip_cancel_pair(H, i, j) -> None:
    """Flip the 45°/225° pair (L_F, R_B) — the antipodes that cancel in X,Y."""
    H[i, j] *= -1          # Lf
    H[i + 1, j + 1] *= -1  # Rb


# Kept as an alias so older call sites / self-check stay valid.
_flip_cancel_pair = flip_cancel_pair


def gerzon_sa(order, T_start=20.0, T_end=0.01, cooling=0.9995,
              max_steps=20000, lam_z=1.0, rng=None, callback=None,
              stop_flag=None, start=None):
    """SA in Gerzon's basis: Gram F + lam_z · E_z (stride-2 H₂ prior).

    Moves: flip a whole 2 × 2, or flip only the 45°/225° pair.  Frames
    match tile SA (``step, T, E, best_E, accepts, H``) plus ``E_z``,
    ``n_h2``, ``n_wall``.  ``start`` is an optional ±1 warm start.
    """
    rng = rng or np.random.default_rng()
    if start is not None:
        H = np.array(start, dtype=np.int8, copy=True)
        if H.shape != (order, order):
            raise ValueError(f"start shape {H.shape} != ({order}, {order})")
    else:
        H = random_seed(order, rng).astype(np.int8)
    n = H.shape[0]
    if n < 2:
        raise ValueError("gerzon_sa needs order ≥ 2")

    def tot(M):
        return gram_energy(M) + float(lam_z) * z_energy(M, stride=2)

    E_cur = tot(H)
    best_H = H.copy()
    best_E = E_cur
    T = T_start
    steps = accepts = 0
    t0 = time.monotonic()

    while steps < max_steps and T > T_end:
        i = int(rng.integers(0, n - 1))
        j = int(rng.integers(0, n - 1))
        if rng.random() < 0.5:
            _flip_tile(H, i, j)
            undo = _flip_tile
        else:
            _flip_cancel_pair(H, i, j)
            undo = _flip_cancel_pair
        E_new = tot(H)
        delta = E_new - E_cur
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            E_cur = E_new
            accepts += 1
            if E_cur < best_E:
                best_H = H.copy()
                best_E = E_cur
        else:
            undo(H, i, j)
        steps += 1
        T *= cooling
        if steps % 500 == 0:
            if callback is not None:
                al = decode_cells(best_H, stride=2)
                callback({
                    "step": steps, "T": T, "E": E_cur, "best_E": best_E,
                    "accepts": accepts, "H": best_H,
                    "E_z": z_energy(best_H, stride=2),
                    "n_h2": al["n_h2"], "n_wall": al["n_wall"],
                })
            if stop_flag is not None and stop_flag.is_set():
                break
    return best_H, dict(
        steps=steps, accepts=accepts, best_E=best_E,
        E_z=z_energy(best_H, stride=2),
        elapsed_s=time.monotonic() - t0,
        hadamard=(best_E < 1e-6),
        **{k: decode_cells(best_H, stride=2)[k]
           for k in ("n_h2", "n_wall", "n_cohesive")},
    )


def gerzon_ils(order, T_start=20.0, cooling=0.9995, sa_steps=15000,
               restarts=5, time_budget=None, lam_z=1.0, rng=None,
               stop_flag=None, progress_callback=None):
    """ILS wrapper around ``gerzon_sa`` (same contract as ``tile_ils``)."""
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
        H, _st = gerzon_sa(
            order, T_start=T_start, cooling=cooling, max_steps=sa_steps,
            lam_z=lam_z, rng=rng, callback=progress_callback,
            stop_flag=stop_flag,
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
    from .hadamard import sylvester, check

    # (a) AB_H is Hadamard; Lf↔Lb swap is Sylvester H₄.
    assert check(AB_H)["is_hadamard"], "AB_H is not Hadamard"
    H4 = sylvester_h4()
    S4 = sylvester(4)
    assert np.array_equal(H4, S4), (H4, S4)
    print("PASS  AB_H is Hadamard; Lf↔Lb columns = Sylvester H₄")

    # (b) encode/decode round-trip and horizontal Z=0.
    rng = np.random.default_rng(0)
    A = rng.standard_normal(4)
    W, X, Y, Z = ab_encode(*A)
    back = np.stack(ab_decode(W, X, Y, Z))
    assert np.allclose(back, A), (back, A)
    # four equal horizontal corners → Z = 0, W = 2·sign
    W, X, Y, Z = ab_encode(1, 1, 1, 1)
    assert Z == 0.0 and W == 2.0 and X == 0.0 and Y == 0.0
    print("PASS  AB encode/decode; horizontal all-+ has Z=0")

    # (c) 45°/225° pair (Lf, Rb) are antipodes: same polarity cancels in
    #     the odd figure-8s X,Y and adds in W and Z.
    W, X, Y, Z = ab_encode(Lb=0, Lf=1, Rf=0, Rb=1)
    assert abs(X) < 1e-12 and abs(Y) < 1e-12, (W, X, Y, Z)
    assert W == 1.0 and Z == 1.0
    print("PASS  Lf/Rb (45°/225°) same polarity: X=Y=0, add in W,Z")

    # (d) trichotomy on ±1 cells.
    def zint(Lf, Rf, Lb, Rb):
        return int(round(2 * ab_encode(Lb, Lf, Rf, Rb)[3]))

    assert abs(zint(1, 1, 1, 1)) == 0          # cohesive W
    assert abs(zint(1, 1, -1, -1)) == 0        # cohesive X
    assert abs(zint(1, -1, 1, -1)) == 0        # cohesive Y
    assert abs(zint(1, 1, 1, -1)) == 2         # H2
    assert abs(zint(1, -1, -1, 1)) == 4        # wall (checkerboard)
    print("PASS  |Z_int| trichotomy: 0 cohesive / 2 H₂ / 4 wall")

    # (e) Sylvester: stride-2 tiles are all H₂; stride-1 walls sit on seams.
    for n in (4, 8, 16):
        H = sylvester(n)
        al = decode_cells(H, stride=2)
        assert al["n_h2"] == al["n_cells"] and al["n_wall"] == 0, (n, al)
        assert z_energy(H, stride=2) == 0.0
        ov = decode_cells(H, stride=1)
        if n > 4:
            assert ov["n_wall"] > 0, f"sylvester({n}) overlapping Z has no seams"
    print("PASS  Sylvester stride-2 all H₂; overlapping Z marks seams")

    # (f) Sylvester is a zero of both energies; breaking one corner
    #     of an aligned H₂ tile raises E_z (the 45°/225° pair-flip can
    #     rotate one H₂ into another and leave E_z = 0).
    H = sylvester(4)
    assert gram_energy(H) == 0.0 and z_energy(H) == 0.0
    Hf = H.copy()
    Hf[0, 0] *= -1
    assert z_energy(Hf) > 0.0
    # cold SA reports the Gerzon counters and accepts some proposals
    Hc, infoc = gerzon_sa(4, max_steps=2000, rng=np.random.default_rng(1))
    assert infoc["accepts"] > 0 and infoc["n_h2"] + infoc["n_wall"] + infoc["n_cohesive"] == 4
    print(f"(f) SA 2000 steps: accepts={infoc['accepts']}, "
          f"E={infoc['best_E']:.2f}, n_h2={infoc['n_h2']}")
    print("gerzon self-check OK")
