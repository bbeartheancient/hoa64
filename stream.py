"""Time-domain HOA streams: encode mono sources, frame-wise analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .analysis import (
    angular_error_deg,
    doa_from_intensity,
    field_energy,
    peak_direction,
)
from .audio_io import ensure_hoa_channels
from .basis import MAX_ORDER, N_CHANNELS, sh_sn3d
from .encode import encode_plane_waves, mix
from .stft import frame_multichannel, hann_window, stft, stft_freqs


@dataclass
class SourceSpec:
    """One plane-wave source in a synthetic or annotated scene."""

    azimuth_deg: float
    elevation_deg: float
    signal: np.ndarray  # (n_samples,)
    label: str = ""


def encode_mono_plane_wave(
    signal: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float = 0.0,
    *,
    max_order: int = MAX_ORDER,
) -> np.ndarray:
    """Mono signal from a known direction → Ambix HOA stream (C, T)."""
    sig = np.asarray(signal, dtype=np.float64).reshape(1, -1)
    return encode_plane_waves(
        [azimuth_deg],
        [elevation_deg],
        sig,
        degrees=True,
        max_order=max_order,
    )


def encode_scene(
    sources: Sequence[SourceSpec],
    *,
    max_order: int = MAX_ORDER,
) -> np.ndarray:
    """Superpose multiple plane-wave sources into one HOA stream (C, T)."""
    if not sources:
        raise ValueError("sources must be non-empty")
    lengths = [int(np.asarray(s.signal).reshape(-1).shape[0]) for s in sources]
    T = max(lengths)
    fields = []
    for s in sources:
        sig = np.asarray(s.signal, dtype=np.float64).reshape(-1)
        if sig.shape[0] < T:
            sig = np.pad(sig, (0, T - sig.shape[0]))
        elif sig.shape[0] > T:
            sig = sig[:T]
        fields.append(
            encode_mono_plane_wave(
                sig, s.azimuth_deg, s.elevation_deg, max_order=max_order
            )
        )
    out = fields[0]
    for f in fields[1:]:
        out = out + f
    return out


def hoa_rms(hoa: np.ndarray) -> np.ndarray:
    """Per-channel RMS. hoa (C,T) → (C,)."""
    a = np.asarray(hoa, dtype=np.float64)
    return np.sqrt(np.mean(a * a, axis=-1) + 1e-30)


@dataclass
class FrameAnalysis:
    t_center_sec: float
    energy: float
    doa_az_deg: float
    doa_el_deg: float
    peak_az_deg: float
    peak_el_deg: float
    peak_power: float
    order1_energy: float


def analyze_hoa_frames(
    hoa: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 40.0,
    hop_ms: float = 20.0,
    max_order: int = MAX_ORDER,
    peak_grid: bool = False,
) -> list[FrameAnalysis]:
    """Short-time spatial analysis of an HOA stream.

    Uses rectangular frames; DOA from order-1 intensity on frame-averaged
    (or energy-weighted) coefficients. Optional dense peak per frame is slower.
    """
    a = ensure_hoa_channels(hoa, max_order=max_order)
    frame_len = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    frames = frame_multichannel(a, frame_len=frame_len, hop=hop)
    # Energy-weighted mean coefficient per frame: sum_t a[c,t]*|a_w| style —
    # use simple mean of coeffs (works for quasi-stationary plane waves).
    win = hann_window(frame_len)
    win = win / (np.sum(win) + 1e-30)

    out: list[FrameAnalysis] = []
    nch_o1 = 4
    for i in range(frames.shape[0]):
        block = frames[i]  # (C, L)
        # AC-safe: do NOT average coeffs (→0 for audio). Use intensity products.
        w = win  # (L,)
        # weighted instantaneous intensity ~ W*X etc.
        W = block[0] * w
        Yc = block[1] * w
        Zc = block[2] * w
        Xc = block[3] * w
        I = np.array(
            [
                float(np.sum(W * Xc)),
                float(np.sum(W * Yc)),
                float(np.sum(W * Zc)),
            ],
            dtype=np.float64,
        )
        nrm = float(np.linalg.norm(I))
        if nrm < 1e-18:
            az, el = 0.0, 0.0
        else:
            from .basis import az_el_from_unit

            az, el = az_el_from_unit(I / nrm, degrees=True)
            az, el = float(az), float(el)

        # RMS energy of the frame
        energy = float(np.mean(np.sum(block * block, axis=0)))
        o1e = float(np.mean(np.sum(block[:nch_o1] ** 2, axis=0)))

        if peak_grid:
            # Build a pseudo-static vector: sign-stable energy-weighted mean
            # via sqrt of mean squares * sign of correlation with W
            rms = np.sqrt(np.mean(block * block, axis=1) + 1e-30)
            sign = np.sign(np.mean(block * block[0:1, :], axis=1) + 1e-30)
            pseudo = rms * sign
            paz, pel, pv = peak_direction(
                pseudo, n_azi=48, n_el=24, max_order=min(max_order, 3)
            )
        else:
            paz, pel, pv = float(az), float(el), energy
        t_center = (i * hop + 0.5 * frame_len) / float(sample_rate)
        out.append(
            FrameAnalysis(
                t_center_sec=t_center,
                energy=energy,
                doa_az_deg=float(az),
                doa_el_deg=float(el),
                peak_az_deg=float(paz),
                peak_el_deg=float(pel),
                peak_power=float(pv),
                order1_energy=o1e,
            )
        )
    return out


def analyze_hoa_stft_bands(
    hoa: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int = 1024,
    hop: int = 256,
    band_edges_hz: Sequence[float] | None = None,
) -> list[dict]:
    """Per-frequency-band intensity DOA using order-1 HOA channels only.

    Returns list of {band_hz: [lo,hi], doa_az, doa_el, energy}.
    """
    a = ensure_hoa_channels(hoa, max_order=1)
    if band_edges_hz is None:
        band_edges_hz = [0, 250, 500, 1000, 2000, 4000, 8000, sample_rate / 2]

    # STFT of W,Y,Z,X
    specs = []
    for c in range(4):
        S = stft(a[c], n_fft=n_fft, hop=hop)
        specs.append(S)
    freqs = stft_freqs(n_fft, sample_rate)
    # Time-average power-weighted intensity per bin then fold into bands
    W, Y, Z, X = specs
    # Use complex conjugate product for active intensity-like measure
    # I_x ~ Re(W * conj(X)), etc., averaged over frames
    Ix = np.mean(np.real(W * np.conj(X)), axis=1)
    Iy = np.mean(np.real(W * np.conj(Y)), axis=1)
    Iz = np.mean(np.real(W * np.conj(Z)), axis=1)
    Ew = np.mean(np.abs(W) ** 2, axis=1)

    edges = list(band_edges_hz)
    reports = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            continue
        I = np.array(
            [np.sum(Ix[mask]), np.sum(Iy[mask]), np.sum(Iz[mask])],
            dtype=np.float64,
        )
        n = np.linalg.norm(I)
        if n < 1e-15:
            az, el = 0.0, 0.0
        else:
            from .basis import az_el_from_unit

            az, el = az_el_from_unit(I / n, degrees=True)
            az, el = float(az), float(el)
        reports.append(
            {
                "band_hz": [float(lo), float(hi)],
                "doa_az_deg": az,
                "doa_el_deg": el,
                "energy": float(np.sum(Ew[mask])),
            }
        )
    return reports
