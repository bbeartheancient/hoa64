"""Synthetic test signals for Phase 1 audio pipeline."""

from __future__ import annotations

import numpy as np


def tone(
    freq_hz: float,
    duration_sec: float,
    sample_rate: int,
    *,
    amplitude: float = 0.5,
    phase: float = 0.0,
) -> np.ndarray:
    t = np.arange(int(round(duration_sec * sample_rate)), dtype=np.float64) / sample_rate
    return amplitude * np.sin(2.0 * np.pi * freq_hz * t + phase)


def noise(
    duration_sec: float,
    sample_rate: int,
    *,
    amplitude: float = 0.2,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(round(duration_sec * sample_rate))
    return amplitude * rng.standard_normal(n)


def envelope_adsr(
    n: int,
    sample_rate: int,
    *,
    attack: float = 0.01,
    release: float = 0.05,
) -> np.ndarray:
    env = np.ones(n, dtype=np.float64)
    a = min(n, int(attack * sample_rate))
    r = min(n, int(release * sample_rate))
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a)
    if r > 0:
        env[-r:] = np.linspace(1.0, 0.0, r)
    return env
