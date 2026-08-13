"""Rotate HOA fields.

Phase 2 default: Wigner-D block rotation (fast, all orders).
Phase 0 reference: dense spherical re-projection (``method="dense"``).
Order-1 shortcut: ``rotate_matrix_order1`` (exact Cartesian).
"""

from __future__ import annotations

import math

import numpy as np

from .basis import MAX_ORDER, N_CHANNELS, sh_sn3d_batch, sphere_grid, unit_vector, acn_index
from .wigner import (
    apply_hoa_rotation,
    hoa_rotation_matrix,
    rotation_matrix_zyx,
)


def _rotation_matrix_zyx(yaw: float, pitch: float, roll: float, degrees: bool) -> np.ndarray:
    return rotation_matrix_zyx(yaw, pitch, roll, degrees=degrees)


def rotate_matrix_order1(
    hoa: np.ndarray,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    *,
    degrees: bool = True,
) -> np.ndarray:
    """Exact rotation for order-1 channels (Y,Z,X); W and n≥2 unchanged."""
    a = np.array(hoa, dtype=np.float64, copy=True)
    R = _rotation_matrix_zyx(yaw, pitch, roll, degrees)
    if a.ndim == 1:
        cart = np.array([a[3], a[1], a[2]])
        Xp, Yp, Zp = R @ cart
        a[3], a[1], a[2] = Xp, Yp, Zp
        return a
    cart = np.stack([a[3], a[1], a[2]], axis=0)
    rot = R @ cart
    a[3], a[1], a[2] = rot[0], rot[1], rot[2]
    return a


def _sn3d_channel_scale(max_order: int) -> np.ndarray:
    nch = (max_order + 1) ** 2
    scale = np.empty(nch, dtype=np.float64)
    for n in range(max_order + 1):
        s = float(2 * n + 1)
        for m in range(-n, n + 1):
            scale[acn_index(n, m)] = s
    return scale


def rotate_yaw_pitch_roll(
    hoa: np.ndarray,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    *,
    degrees: bool = True,
    max_order: int = MAX_ORDER,
    method: str = "wigner",
    n_azi: int = 96,
    n_el: int = 48,
) -> np.ndarray:
    """Rotate HOA field (active): values move with R.

    Parameters
    ----------
    method :
        ``"wigner"`` (default, Phase 2) — exact band-limited SH rotation.
        ``"dense"`` — Phase 0 spherical sample / re-encode (slow reference).
    """
    if method == "wigner":
        R = _rotation_matrix_zyx(yaw, pitch, roll, degrees)
        M = hoa_rotation_matrix(R, max_order=max_order)
        return apply_hoa_rotation(hoa, M)
    if method == "dense":
        return _rotate_dense(
            hoa,
            yaw,
            pitch,
            roll,
            degrees=degrees,
            max_order=max_order,
            n_azi=n_azi,
            n_el=n_el,
        )
    raise ValueError("method must be 'wigner' or 'dense'")


def _rotate_dense(
    hoa: np.ndarray,
    yaw: float,
    pitch: float,
    roll: float,
    *,
    degrees: bool,
    max_order: int,
    n_azi: int,
    n_el: int,
) -> np.ndarray:
    """Phase 0 dense re-encode (reference)."""
    a = np.asarray(hoa, dtype=np.float64)
    if a.ndim == 2:
        out = np.zeros(((max_order + 1) ** 2, a.shape[1]), dtype=np.float64)
        for t in range(a.shape[1]):
            out[:, t] = _rotate_dense(
                a[:, t],
                yaw,
                pitch,
                roll,
                degrees=degrees,
                max_order=max_order,
                n_azi=n_azi,
                n_el=n_el,
            )
        full = np.zeros((N_CHANNELS, a.shape[1]), dtype=np.float64)
        full[: out.shape[0]] = out
        return full

    from .decode import decode_directions

    nch = (max_order + 1) ** 2
    a = a[:nch]
    R = _rotation_matrix_zyx(yaw, pitch, roll, degrees)

    azi, el, weights = sphere_grid(n_azi, n_el, degrees=True)
    AA, EE = np.meshgrid(azi, el, indexing="ij")
    dirs = unit_vector(AA, EE, degrees=True).reshape(-1, 3)
    w = weights.reshape(-1)
    f = decode_directions(
        a, AA.reshape(-1), EE.reshape(-1), degrees=True, max_order=max_order
    )
    dirs_rot = (R @ dirs.T).T
    Y_rot = sh_sn3d_batch(dirs_rot, max_order=max_order)
    a_new = np.einsum("i,i,ic->c", w, f, Y_rot)
    a_new *= _sn3d_channel_scale(max_order) / (4.0 * math.pi)
    full = np.zeros(N_CHANNELS, dtype=np.float64)
    full[: a_new.shape[0]] = a_new
    return full


def rotate_source_directions(
    azimuths: np.ndarray,
    elevations: np.ndarray,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    *,
    degrees: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate source az/el by the same R (exact plane-wave ground truth)."""
    dirs = unit_vector(azimuths, elevations, degrees=degrees)
    flat = np.atleast_2d(dirs.reshape(-1, 3))
    R = _rotation_matrix_zyx(yaw, pitch, roll, degrees)
    rot = (R @ flat.T).T
    from .basis import az_el_from_unit

    az, el = az_el_from_unit(rot, degrees=degrees)
    return az.reshape(np.shape(azimuths)), el.reshape(np.shape(elevations))


def rotate_plane_wave_field(
    hoa: np.ndarray,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    *,
    degrees: bool = True,
    max_order: int = MAX_ORDER,
    method: str = "wigner",
) -> np.ndarray:
    return rotate_yaw_pitch_roll(
        hoa,
        yaw,
        pitch,
        roll,
        degrees=degrees,
        max_order=max_order,
        method=method,
    )
