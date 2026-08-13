"""RNN integrator stub — Phase 0 interface for iterative motion in a field.

No learned weights yet: explicit Euler integration of pose + field rotation.
Validates the "calculator + loop" hypothesis before training dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .basis import N_CHANNELS
from .encode import encode_points
from .rotate import rotate_matrix_order1, rotate_yaw_pitch_roll
from .analysis import doa_from_intensity, peak_direction, field_energy


def _pad64(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    out = np.zeros(N_CHANNELS, dtype=np.float64)
    out[: min(N_CHANNELS, a.shape[0])] = a[:N_CHANNELS]
    return out


@dataclass
class SpatialState:
    """Agent + field state for iterative spatial calculation."""

    hoa: np.ndarray  # (64,) world field in listener frame after last step
    yaw: float = 0.0  # degrees, agent heading
    pitch: float = 0.0
    roll: float = 0.0
    history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.hoa = _pad64(self.hoa)

    def snapshot(self) -> dict:
        az, el = doa_from_intensity(self.hoa)
        paz, pel, pval = peak_direction(self.hoa, n_azi=72, n_el=36)
        return {
            "yaw": self.yaw,
            "pitch": self.pitch,
            "roll": self.roll,
            "energy": field_energy(self.hoa),
            "doa_intensity_az_el": (az, el),
            "doa_peak_az_el": (paz, pel),
            "peak_power": pval,
        }


def step_rotate(
    state: SpatialState,
    d_yaw: float = 0.0,
    d_pitch: float = 0.0,
    d_roll: float = 0.0,
    *,
    max_order: int = 7,
    dense: bool = True,
) -> SpatialState:
    """Integrate a pose increment: rotate the field opposite agent turn.

    If the agent yaws +θ (turns left), the world field in head frame yaws −θ.
    """
    # Agent pose update
    yaw = state.yaw + d_yaw
    pitch = state.pitch + d_pitch
    roll = state.roll + d_roll
    # Field in listener frame: rotate by -d_*
    if dense and max_order > 1:
        hoa = rotate_yaw_pitch_roll(
            state.hoa,
            yaw=-d_yaw,
            pitch=-d_pitch,
            roll=-d_roll,
            degrees=True,
            max_order=max_order,
        )
    else:
        hoa = rotate_matrix_order1(
            state.hoa, yaw=-d_yaw, pitch=-d_pitch, roll=-d_roll, degrees=True
        )
    hoa = _pad64(hoa)
    new = SpatialState(hoa=hoa, yaw=yaw, pitch=pitch, roll=roll, history=list(state.history))
    new.history.append(new.snapshot())
    return new


def world_from_sources(
    azimuths,
    elevations,
    gains=None,
    *,
    max_order: int = 7,
) -> SpatialState:
    """Build initial state from world-frame sources (listener at origin, identity pose)."""
    a = encode_points(azimuths, elevations, gains, degrees=True, max_order=max_order)
    st = SpatialState(hoa=a)
    st.history.append(st.snapshot())
    return st
