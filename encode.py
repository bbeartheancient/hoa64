"""Encode discrete sources into HOA-7 (Ambix SN3D) coefficient vectors."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .basis import MAX_ORDER, N_CHANNELS, sh_sn3d, sh_sn3d_batch, unit_vector


def encode_points(
    azimuths: Sequence[float] | np.ndarray,
    elevations: Sequence[float] | np.ndarray,
    gains: Sequence[float] | np.ndarray | None = None,
    *,
    degrees: bool = True,
    max_order: int = MAX_ORDER,
) -> np.ndarray:
    """Encode point / plane-wave sources on the sphere into HOA coeffs.

    Each source at (az, el) with gain g contributes g * Y(az, el).
    Returns shape (n_channels,) float64. Time-domain: call per sample
    with complex gains or use encode_plane_waves for signals.
    """
    az = np.atleast_1d(np.asarray(azimuths, dtype=np.float64))
    el = np.atleast_1d(np.asarray(elevations, dtype=np.float64))
    if az.shape != el.shape:
        raise ValueError("azimuths and elevations must match")
    if gains is None:
        g = np.ones(az.shape, dtype=np.float64)
    else:
        g = np.asarray(gains, dtype=np.float64)
        g = np.broadcast_to(g, az.shape)
    Y = sh_sn3d(az, el, degrees=degrees, max_order=max_order)  # (N, C)
    # a = sum_i g_i Y_i
    return np.einsum("n,nc->c", g.reshape(-1), Y.reshape(-1, Y.shape[-1]))


def encode_plane_waves(
    azimuths: Sequence[float] | np.ndarray,
    elevations: Sequence[float] | np.ndarray,
    signals: np.ndarray,
    *,
    degrees: bool = True,
    max_order: int = MAX_ORDER,
) -> np.ndarray:
    """Encode multi-channel time signals as plane waves.

    signals: (n_sources, n_samples) or (n_sources,)
    Returns: (n_channels, n_samples) or (n_channels,)
    """
    az = np.atleast_1d(np.asarray(azimuths, dtype=np.float64))
    el = np.atleast_1d(np.asarray(elevations, dtype=np.float64))
    sig = np.asarray(signals, dtype=np.float64)
    if sig.ndim == 1:
        if sig.shape[0] == az.shape[0]:
            # (n_sources,) static frame
            return encode_points(az, el, sig, degrees=degrees, max_order=max_order)
        raise ValueError("1-D signals length must equal n_sources")
    if sig.ndim != 2 or sig.shape[0] != az.shape[0]:
        raise ValueError("signals must be (n_sources, n_samples)")
    Y = sh_sn3d(az, el, degrees=degrees, max_order=max_order)  # (S, C)
    # a[c,t] = sum_s Y[s,c] * sig[s,t]
    return np.einsum("sc,st->ct", Y, sig)


def mix(*fields: np.ndarray) -> np.ndarray:
    """Superpose HOA fields (same channel count)."""
    if not fields:
        return np.zeros(N_CHANNELS, dtype=np.float64)
    out = np.zeros_like(np.asarray(fields[0], dtype=np.float64))
    for f in fields:
        out = out + np.asarray(f, dtype=np.float64)
    return out


def encode_xyz(
    directions_xyz: np.ndarray,
    gains: np.ndarray | None = None,
    *,
    max_order: int = MAX_ORDER,
) -> np.ndarray:
    """Encode sources given as unit (or non-unit) Cartesian directions."""
    d = np.atleast_2d(np.asarray(directions_xyz, dtype=np.float64))
    Y = sh_sn3d_batch(d, max_order=max_order)
    if gains is None:
        g = np.ones(d.shape[0], dtype=np.float64)
    else:
        g = np.asarray(gains, dtype=np.float64).reshape(-1)
    return np.einsum("n,nc->c", g, Y)
