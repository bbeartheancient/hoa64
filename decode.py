"""Decode HOA-7 coefficients to samples on the sphere."""

from __future__ import annotations

import numpy as np

from .basis import MAX_ORDER, sh_sn3d, sh_sn3d_batch, sphere_grid


def decode_directions(
    hoa: np.ndarray,
    azimuths: np.ndarray | float,
    elevations: np.ndarray | float,
    *,
    degrees: bool = True,
    max_order: int = MAX_ORDER,
) -> np.ndarray:
    """Sample field a · Y(Ω) at given directions.

    hoa: (C,) or (C, T)
    returns: (...) or (..., T)
    """
    a = np.asarray(hoa, dtype=np.float64)
    nch = (max_order + 1) ** 2
    # Allow shorter coefficient vectors (truncated order) or longer (ignore tail).
    if a.ndim == 1:
        aa = np.zeros(nch, dtype=np.float64)
        n = min(nch, a.shape[0])
        aa[:n] = a[:n]
        Y = sh_sn3d(azimuths, elevations, degrees=degrees, max_order=max_order)
        return np.einsum("...c,c->...", Y[..., :nch], aa)
    if a.ndim == 2:
        aa = np.zeros((nch, a.shape[1]), dtype=np.float64)
        n = min(nch, a.shape[0])
        aa[:n] = a[:n, :]
        Y = sh_sn3d(azimuths, elevations, degrees=degrees, max_order=max_order)
        return np.einsum("...c,ct->...t", Y[..., :nch], aa)
    raise ValueError("hoa must be (C,) or (C,T)")


def decode_grid(
    hoa: np.ndarray,
    n_azi: int = 72,
    n_el: int = 36,
    *,
    degrees: bool = True,
    max_order: int = MAX_ORDER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode on a product sphere grid.

    Returns azi, el, samples with samples shape (n_azi, n_el) or (n_azi, n_el, T).
    """
    azi, el, _ = sphere_grid(n_azi, n_el, degrees=degrees)
    AA, EE = np.meshgrid(azi, el, indexing="ij")
    samp = decode_directions(hoa, AA, EE, degrees=degrees, max_order=max_order)
    return azi, el, samp


def beamform(
    hoa: np.ndarray,
    azimuth: float,
    elevation: float,
    *,
    degrees: bool = True,
    max_order: int = MAX_ORDER,
) -> float | np.ndarray:
    """Look / listen toward one direction: scalar (or time series) readout."""
    return decode_directions(
        hoa, azimuth, elevation, degrees=degrees, max_order=max_order
    )
