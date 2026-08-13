"""JSON spatial report — compact payload for other models / tools."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from .analysis import angular_error_deg, doa_from_intensity, field_energy, peak_direction
from .audio_io import ensure_hoa_channels, read_wav
from .basis import MAX_ORDER, N_CHANNELS, channel_names
from .stream import (
    FrameAnalysis,
    SourceSpec,
    analyze_hoa_frames,
    analyze_hoa_stft_bands,
    encode_mono_plane_wave,
    encode_scene,
    hoa_rms,
)

PathLike = Union[str, Path]

REPORT_SCHEMA_VERSION = "spatial-hoa.report.v1"


@dataclass
class SpatialReport:
    """Machine-readable spatial summary for LLM / diffusion conditioning."""

    schema: str = REPORT_SCHEMA_VERSION
    kind: str = "spatial_field"
    sample_rate: int = 0
    n_samples: int = 0
    duration_sec: float = 0.0
    max_order: int = MAX_ORDER
    n_channels: int = N_CHANNELS
    # Global broadband
    energy: float = 0.0
    doa_az_deg: float = 0.0
    doa_el_deg: float = 0.0
    peak_az_deg: float = 0.0
    peak_el_deg: float = 0.0
    peak_power: float = 0.0
    # Optional detail
    bands: list[dict] = field(default_factory=list)
    frames: list[dict] = field(default_factory=list)
    sources_hint: list[dict] = field(default_factory=list)
    channel_rms: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: PathLike) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    def one_liner(self) -> str:
        """Short string suitable for tool_result / system hints."""
        return (
            f"spatial: DOA az={self.doa_az_deg:.1f}° el={self.doa_el_deg:.1f}° "
            f"peak=({self.peak_az_deg:.1f},{self.peak_el_deg:.1f}) "
            f"E={self.energy:.4g} T={self.duration_sec:.2f}s"
        )


def _frames_to_dicts(frames: Sequence[FrameAnalysis]) -> list[dict]:
    return [
        {
            "t_sec": f.t_center_sec,
            "energy": f.energy,
            "doa_az_deg": f.doa_az_deg,
            "doa_el_deg": f.doa_el_deg,
            "peak_az_deg": f.peak_az_deg,
            "peak_el_deg": f.peak_el_deg,
            "order1_energy": f.order1_energy,
        }
        for f in frames
    ]


def report_from_hoa(
    hoa: np.ndarray,
    sample_rate: int,
    *,
    max_order: int = MAX_ORDER,
    frame_ms: float = 40.0,
    hop_ms: float = 20.0,
    include_frames: bool = True,
    include_bands: bool = True,
    include_peak_map: bool = True,
    max_frames: int = 64,
    meta: Optional[dict] = None,
    notes: Optional[list[str]] = None,
) -> SpatialReport:
    """Build a SpatialReport from HOA coefficients (C,T) or (C,)."""
    a = np.asarray(hoa, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    a = ensure_hoa_channels(a, max_order=max_order)
    n_samples = int(a.shape[1])
    duration = n_samples / float(sample_rate) if sample_rate else 0.0

    # Broadband DOA must be AC-safe (time-mean of coeffs → 0 for audio).
    # 1) global intensity from products W*X, W*Y, W*Z
    # 2) energy-weighted blend of per-frame DOAs
    from .basis import az_el_from_unit, unit_vector

    W, Yc, Zc, Xc = a[0], a[1], a[2], a[3]
    I_global = np.array(
        [
            float(np.mean(W * Xc)),
            float(np.mean(W * Yc)),
            float(np.mean(W * Zc)),
        ]
    )
    nI = float(np.linalg.norm(I_global))
    if nI > 1e-18:
        az, el = az_el_from_unit(I_global / nI, degrees=True)
        az, el = float(az), float(el)
    else:
        az, el = 0.0, 0.0

    frames = analyze_hoa_frames(
        a,
        sample_rate,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        max_order=max_order,
        peak_grid=False,
    )
    if frames:
        vecs = []
        for f in frames:
            if f.energy <= 1e-18:
                continue
            u = unit_vector(f.doa_az_deg, f.doa_el_deg, degrees=True)
            vecs.append(u * f.energy)
        if vecs:
            v = np.sum(vecs, axis=0)
            n = float(np.linalg.norm(v))
            if n > 1e-15:
                az_f, el_f = az_el_from_unit(v / n, degrees=True)
                # Prefer frame blend when global intensity is weak
                if nI < 1e-12:
                    az, el = float(az_f), float(el_f)
                else:
                    # average unit vectors of global + frames
                    ug = unit_vector(az, el, degrees=True)
                    uf = unit_vector(float(az_f), float(el_f), degrees=True)
                    um = ug + uf
                    nm = float(np.linalg.norm(um))
                    if nm > 1e-15:
                        az, el = az_el_from_unit(um / nm, degrees=True)
                        az, el = float(az), float(el)

    energy = float(np.mean(np.sum(a * a, axis=0)))

    if include_peak_map:
        # Pseudo-static field from RMS * sign(corr with W) for map peak
        rms = np.sqrt(np.mean(a * a, axis=1) + 1e-30)
        sign = np.sign(np.mean(a * a[0:1, :], axis=1) + 1e-30)
        pseudo = rms * sign
        paz, pel, pv = peak_direction(
            pseudo, n_azi=72, n_el=36, max_order=min(max_order, 3)
        )
    else:
        paz, pel, pv = az, el, energy

    bands: list[dict] = []
    if include_bands and n_samples >= 256:
        bands = analyze_hoa_stft_bands(a, sample_rate)

    frame_dicts: list[dict] = []
    if include_frames and frames:
        step = max(1, len(frames) // max_frames)
        frame_dicts = _frames_to_dicts(frames[::step][:max_frames])

    return SpatialReport(
        sample_rate=int(sample_rate),
        n_samples=n_samples,
        duration_sec=duration,
        max_order=max_order,
        n_channels=(max_order + 1) ** 2,
        energy=energy,
        doa_az_deg=float(az),
        doa_el_deg=float(el),
        peak_az_deg=float(paz),
        peak_el_deg=float(pel),
        peak_power=float(pv),
        bands=bands,
        frames=frame_dicts,
        channel_rms=[float(x) for x in hoa_rms(a)[:8]],  # first 8 for brevity
        notes=list(notes or []),
        meta=dict(meta or {}),
    )


def report_from_mono_wav(
    path: PathLike,
    azimuth_deg: float,
    elevation_deg: float = 0.0,
    *,
    max_order: int = MAX_ORDER,
    **kwargs,
) -> SpatialReport:
    """Load mono (or mixdown) WAV, encode as plane wave from known direction, report."""
    audio, sr = read_wav(path)
    if audio.shape[0] == 1:
        mono = audio[0]
    else:
        mono = np.mean(audio, axis=0)
        notes_extra = ["mixed multi-channel input down to mono before plane-wave encode"]
    hoa = encode_mono_plane_wave(mono, azimuth_deg, elevation_deg, max_order=max_order)
    notes = list(kwargs.pop("notes", []) or [])
    if audio.shape[0] > 1:
        notes.extend(notes_extra)
    notes.append(
        f"encoded mono as plane wave az={azimuth_deg}° el={elevation_deg}°"
    )
    meta = dict(kwargs.pop("meta", None) or {})
    meta.update(
        {
            "source_path": str(path),
            "encode": "mono_plane_wave",
            "encode_az_deg": azimuth_deg,
            "encode_el_deg": elevation_deg,
            "created_unix": time.time(),
        }
    )
    return report_from_hoa(
        hoa, sr, max_order=max_order, notes=notes, meta=meta, **kwargs
    )


def report_from_ambix_wav(
    path: PathLike,
    *,
    max_order: int = MAX_ORDER,
    **kwargs,
) -> SpatialReport:
    """Load multi-channel Ambix WAV (ACN order) and analyze."""
    audio, sr = read_wav(path)
    hoa = ensure_hoa_channels(audio, max_order=max_order)
    notes = list(kwargs.pop("notes", []) or [])
    notes.append(f"loaded Ambix-style multi-channel WAV with {audio.shape[0]} ch")
    meta = dict(kwargs.pop("meta", None) or {})
    meta.update(
        {
            "source_path": str(path),
            "encode": "ambix_wav",
            "input_channels": int(audio.shape[0]),
            "created_unix": time.time(),
        }
    )
    return report_from_hoa(
        hoa, sr, max_order=max_order, notes=notes, meta=meta, **kwargs
    )


def report_from_scene(
    sources: Sequence[SourceSpec],
    sample_rate: int,
    *,
    max_order: int = MAX_ORDER,
    **kwargs,
) -> SpatialReport:
    """Analyze a synthetic multi-source scene."""
    hoa = encode_scene(sources, max_order=max_order)
    hints = [
        {
            "label": s.label or f"src{i}",
            "azimuth_deg": s.azimuth_deg,
            "elevation_deg": s.elevation_deg,
            "rms": float(np.sqrt(np.mean(np.asarray(s.signal) ** 2) + 1e-30)),
        }
        for i, s in enumerate(sources)
    ]
    notes = list(kwargs.pop("notes", []) or [])
    notes.append(f"synthetic scene with {len(sources)} plane-wave source(s)")
    meta = dict(kwargs.pop("meta", None) or {})
    meta["encode"] = "synthetic_scene"
    rep = report_from_hoa(
        hoa, sample_rate, max_order=max_order, notes=notes, meta=meta, **kwargs
    )
    rep.sources_hint = hints
    return rep


def load_report(path: PathLike) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
