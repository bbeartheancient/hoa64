"""Fast HOA rotation via Wigner-D matrices (real Ambix ACN / SN3D).

Within each order n the (2n+1) coefficients transform by a real matrix built
from complex Wigner D functions. Normalization (SN3D vs N3D) cancels inside
an order, so the same matrices apply to Ambix SN3D.

Public:
  rotation_matrix_zyx     — 3×3 active rotation (yaw/pitch/roll)
  hoa_rotation_matrix     — full ((N+1)²)×((N+1)²) block-diagonal matrix
  apply_hoa_rotation      — a' = M @ a  (also (C,T) streams)
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Tuple

import numpy as np

from .basis import MAX_ORDER, N_CHANNELS, acn_index


def rotation_matrix_zyx(
    yaw: float, pitch: float, roll: float, *, degrees: bool = True
) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll); active rotation of column vectors."""
    if degrees:
        yaw, pitch, roll = map(math.radians, (yaw, pitch, roll))
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def _fact(n: int) -> float:
    if n < 0:
        return 0.0
    return float(math.factorial(n))


@lru_cache(maxsize=8192)
def wigner_d(j: int, mp: int, m: int, beta: float) -> float:
    """Wigner small-d matrix element d^j_{mp,m}(beta), real.

    Explicit finite sum (safe for j ≤ 7 used here).
    """
    if abs(mp) > j or abs(m) > j:
        return 0.0
    # Numerical stability: beta in [0, pi] preferred but general ok
    cb = math.cos(beta * 0.5)
    sb = math.sin(beta * 0.5)
    # Avoid 0**negative
    s_min = max(0, m - mp)
    s_max = min(j + m, j - mp)
    total = 0.0
    for s in range(s_min, s_max + 1):
        den = (
            _fact(j + m - s)
            * _fact(j - mp - s)
            * _fact(s)
            * _fact(s + mp - m)
        )
        if den == 0.0:
            continue
        num = _fact(j + m) * _fact(j - m) * _fact(j + mp) * _fact(j - mp)
        pref = ((-1.0) ** (mp - m + s)) * math.sqrt(num) / den
        cpow = 2 * j + m - mp - 2 * s
        spow = mp - m + 2 * s
        # handle base cases
        cterm = 1.0 if cpow == 0 else (cb ** cpow if abs(cb) > 1e-15 or cpow > 0 else 0.0)
        sterm = 1.0 if spow == 0 else (sb ** spow if abs(sb) > 1e-15 or spow > 0 else 0.0)
        if cpow < 0 and abs(cb) < 1e-15:
            cterm = 0.0
        if spow < 0 and abs(sb) < 1e-15:
            sterm = 0.0
        total += pref * cterm * sterm
    return total


def wigner_D_complex(
    j: int, alpha: float, beta: float, gamma: float
) -> np.ndarray:
    """Complex Wigner-D matrix for order j, indices m',m ∈ [-j..j].

    Ordering: row/col index = m + j  (m from -j to +j).
    D_{mp,m} = e^{-i mp α} d_{mp,m}(β) e^{-i m γ}
    """
    dim = 2 * j + 1
    D = np.zeros((dim, dim), dtype=np.complex128)
    for imp, mp in enumerate(range(-j, j + 1)):
        for im, m in enumerate(range(-j, j + 1)):
            d = wigner_d(j, mp, m, beta)
            D[imp, im] = (
                math.cos(mp * alpha)
                - 1j * math.sin(mp * alpha)
            ) * d * (
                math.cos(m * gamma) - 1j * math.sin(m * gamma)
            )
            # e^{-i θ} = cosθ - i sinθ
    return D


def _real_to_complex_matrix(j: int) -> np.ndarray:
    """Unitary map U: real ACN coeffs (m=-j..j) → complex m=-j..j.

    Convention (common in Ambisonics / real SH):
      c_0 = r_0
      c_{+m} = (-1)^m / √2 * (r_{+m} - i r_{-m})
      c_{-m} =         1 / √2 * (r_{+m} + i r_{-m})
    so r = U^H c  and c = U r  with U unitary.
    """
    dim = 2 * j + 1
    U = np.zeros((dim, dim), dtype=np.complex128)
    # index helper: m -> i = m + j
    def idx(m: int) -> int:
        return m + j

    U[idx(0), idx(0)] = 1.0 + 0.0j
    s2 = 1.0 / math.sqrt(2.0)
    for m in range(1, j + 1):
        sign = (-1.0) ** m
        # c_{+m} from r_{+m}, r_{-m}
        U[idx(m), idx(m)] = sign * s2
        U[idx(m), idx(-m)] = -1j * sign * s2
        # c_{-m} from r_{+m}, r_{-m}
        U[idx(-m), idx(m)] = s2
        U[idx(-m), idx(-m)] = 1j * s2
    return U


def real_sh_rotation_block(
    j: int, alpha: float, beta: float, gamma: float
) -> np.ndarray:
    """Real (2j+1)×(2j+1) rotation matrix for ACN order-j block (m=-j..j)."""
    if j == 0:
        return np.array([[1.0]], dtype=np.float64)
    U = _real_to_complex_matrix(j)
    D = wigner_D_complex(j, alpha, beta, gamma)
    # real coeffs: r' = U^H D U r
    R_c = U.conj().T @ D @ U
    # Should be real symmetric orthogonal (numerically tiny imag)
    return np.real(R_c)


def rotation_matrix_to_zyz(R: np.ndarray) -> Tuple[float, float, float]:
    """Extract ZYZ Euler angles (α, β, γ) from a right-handed rotation matrix.

    R = Rz(α) @ Ry(β) @ Rz(γ)  (active, column vectors).
    """
    R = np.asarray(R, dtype=np.float64)
    # β ∈ [0, π]
    # R[2,2] = cos β
    cbeta = float(np.clip(R[2, 2], -1.0, 1.0))
    beta = math.acos(cbeta)
    sb = math.sin(beta)
    if abs(sb) > 1e-10:
        # α = atan2(R[1,2]/sinβ, R[0,2]/sinβ)
        alpha = math.atan2(R[1, 2] / sb, R[0, 2] / sb)
        # γ = atan2(R[2,1]/sinβ, -R[2,0]/sinβ)
        gamma = math.atan2(R[2, 1] / sb, -R[2, 0] / sb)
    else:
        # Gimbal: β ≈ 0 or π — only α+γ or α-γ determined
        alpha = math.atan2(-R[0, 1], R[0, 0])
        gamma = 0.0
        if cbeta < 0:
            # β = π
            alpha = math.atan2(R[0, 1], -R[0, 0])
    return alpha, beta, gamma


def hoa_rotation_matrix(
    R3: np.ndarray,
    *,
    max_order: int = MAX_ORDER,
) -> np.ndarray:
    """Full Ambix ACN rotation matrix for orders 0..max_order.

    Applies the *same* geometric rotation as R3 does to Cartesian vectors.
    """
    max_order = int(max_order)
    nch = (max_order + 1) ** 2
    M = np.zeros((nch, nch), dtype=np.float64)
    alpha, beta, gamma = rotation_matrix_to_zyz(R3)

    # Order 0
    M[0, 0] = 1.0

    # Order 1: direct Cartesian for numerical exactness / convention lock.
    # ACN: [Y, Z, X] = indices 1,2,3 ; cart = [X,Y,Z]
    # v' = R3 @ v  ⇒  [X',Y',Z'] = R3 @ [X,Y,Z]
    if max_order >= 1:
        # Build 3×3 block mapping [Y,Z,X] -> [Y',Z',X']
        # [X']   [R00 R01 R02] [X]
        # [Y'] = [R10 R11 R12] [Y]
        # [Z']   [R20 R21 R22] [Z]
        # Y' = R10 X + R11 Y + R12 Z
        # Z' = R20 X + R21 Y + R22 Z
        # X' = R00 X + R01 Y + R02 Z
        # Coefficients of (Y,Z,X):
        # Y' = R11 Y + R12 Z + R10 X
        # Z' = R21 Y + R22 Z + R20 X
        # X' = R01 Y + R02 Z + R00 X
        B = np.array(
            [
                [R3[1, 1], R3[1, 2], R3[1, 0]],
                [R3[2, 1], R3[2, 2], R3[2, 0]],
                [R3[0, 1], R3[0, 2], R3[0, 0]],
            ],
            dtype=np.float64,
        )
        M[1:4, 1:4] = B

    for n in range(2, max_order + 1):
        block = real_sh_rotation_block(n, alpha, beta, gamma)
        # Verify det ~ 1; if complex conversion convention is flipped we may
        # need transpose — tests catch this against plane-wave re-encode.
        i0 = n * n  # ACN start for order n is n^2 (m=-n → n*(n+1)+(-n)=n^2)
        # Wait: n*(n+1)+(-n) = n^2 + n - n = n^2. Yes.
        dim = 2 * n + 1
        M[i0 : i0 + dim, i0 : i0 + dim] = block

    return M


def apply_hoa_rotation(
    hoa: np.ndarray,
    M: np.ndarray,
) -> np.ndarray:
    """Apply precomputed rotation matrix to (C,) or (C,T) coefficients."""
    a = np.asarray(hoa, dtype=np.float64)
    nch = M.shape[0]
    if a.ndim == 1:
        aa = np.zeros(nch, dtype=np.float64)
        n = min(nch, a.shape[0])
        aa[:n] = a[:n]
        out = M @ aa
        if a.shape[0] > nch:
            full = np.zeros_like(a)
            full[:nch] = out
            return full
        if a.shape[0] < N_CHANNELS:
            full = np.zeros(N_CHANNELS, dtype=np.float64)
            full[:nch] = out
            return full
        return out
    if a.ndim == 2:
        aa = np.zeros((nch, a.shape[1]), dtype=np.float64)
        n = min(nch, a.shape[0])
        aa[:n] = a[:n]
        out = M @ aa
        if a.shape[0] >= N_CHANNELS:
            full = np.zeros((max(a.shape[0], N_CHANNELS), a.shape[1]), dtype=np.float64)
            full[:nch] = out
            return full[: a.shape[0]]
        full = np.zeros((N_CHANNELS, a.shape[1]), dtype=np.float64)
        full[:nch] = out
        return full
    raise ValueError("hoa must be (C,) or (C,T)")


def clear_wigner_cache() -> None:
    wigner_d.cache_clear()
