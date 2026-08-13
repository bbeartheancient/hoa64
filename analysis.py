"""Spatial field analysis on HOA-7 coefficients (no learning)."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .basis import MAX_ORDER, N_CHANNELS, az_el_from_unit, unit_vector
from .decode import decode_directions, decode_grid


def field_energy(hoa: np.ndarray, max_order: int | None = None) -> float | np.ndarray:
    """Sum of squares of channels (proxy energy; SN3D-weighted optional later).

    hoa: (C,) → scalar; (C,T) → (T,)
    """
    a = np.asarray(hoa, dtype=np.float64)
    if max_order is not None:
        nch = (max_order + 1) ** 2
        a = a[..., :nch] if a.ndim == 1 else a[:nch, :]
    if a.ndim == 1:
        return float(np.dot(a, a))
    return np.sum(a * a, axis=0)


def intensity_vector(hoa: np.ndarray) -> np.ndarray:
    """Pseudo-intensity from order-1 SN3D (Ambix).

    For SN3D B-format-like (W,Y,Z,X) = (a0,a1,a2,a3):
      I ∝ W * (X, Y, Z)   in Cartesian (front, left, up).

    Returns (3,) or (3, T) as [Ix, Iy, Iz].
    """
    a = np.asarray(hoa, dtype=np.float64)
    if a.ndim == 1:
        W, Y, Z, X = a[0], a[1], a[2], a[3]
        return np.array([W * X, W * Y, W * Z], dtype=np.float64)
    W, Y, Z, X = a[0], a[1], a[2], a[3]
    return np.stack([W * X, W * Y, W * Z], axis=0)


def doa_from_intensity(
    hoa: np.ndarray,
    *,
    degrees: bool = True,
) -> Tuple[float, float] | Tuple[np.ndarray, np.ndarray]:
    """Direction of arrival from order-1 intensity vector.

    Returns (azimuth, elevation).
    """
    I = intensity_vector(hoa)
    if I.ndim == 1:
        n = np.linalg.norm(I)
        if n < 1e-15:
            return (0.0, 0.0) if degrees else (0.0, 0.0)
        u = I / n
        az, el = az_el_from_unit(u, degrees=degrees)
        return float(az), float(el)
    # (3, T)
    n = np.linalg.norm(I, axis=0, keepdims=True)
    n = np.maximum(n, 1e-15)
    u = (I / n).T  # (T, 3)
    az, el = az_el_from_unit(u, degrees=degrees)
    return az, el


def directional_power(
    hoa: np.ndarray,
    n_azi: int = 72,
    n_el: int = 36,
    *,
    max_order: int = MAX_ORDER,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Power map |a · Y(Ω)|² on a sphere grid.

    Returns azi, el, power[azi, el].
    """
    azi, el, samp = decode_grid(
        hoa, n_azi=n_azi, n_el=n_el, degrees=True, max_order=max_order
    )
    if samp.ndim == 2:
        power = samp * samp
    else:
        # (A, E, T) → average over time
        power = np.mean(samp * samp, axis=-1)
    return azi, el, power


def peak_direction(
    hoa: np.ndarray,
    n_azi: int = 96,
    n_el: int = 48,
    *,
    max_order: int = MAX_ORDER,
    degrees: bool = True,
) -> Tuple[float, float, float]:
    """Argmax of directional power (azimuth, elevation, peak_value)."""
    azi, el, power = directional_power(
        hoa, n_azi=n_azi, n_el=n_el, max_order=max_order
    )
    idx = np.unravel_index(int(np.argmax(power)), power.shape)
    az = float(azi[idx[0]])
    e = float(el[idx[1]])
    if not degrees:
        az, e = math.radians(az), math.radians(e)
    return az, e, float(power[idx])


def angular_error_deg(
    az0: float, el0: float, az1: float, el1: float
) -> float:
    """Great-circle angle between two az/el directions (degrees)."""
    u = unit_vector(az0, el0, degrees=True)
    v = unit_vector(az1, el1, degrees=True)
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return math.degrees(math.acos(c))
