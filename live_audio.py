"""Live microphone / PulseAudio capture → HOA spatial report.

Uses system tools (no sounddevice):
  * ffmpeg -f pulse -i <source>
  * arecord (ALSA fallback)

Mono capture is encoded as a plane wave from a configured look direction
(default: front). Multi-channel (2–4) maps L/R to ±az pseudo-Ambix order-1.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from .audio_io import read_wav, write_wav
from .encode import encode_plane_waves
from .report import SpatialReport, report_from_hoa
from .stream import encode_mono_plane_wave

PathLike = Union[str, Path]


def list_pulse_sources() -> list[str]:
    """Return Pulse/PipeWire source names via pactl."""
    if not shutil.which("pactl"):
        return []
    try:
        out = subprocess.check_output(
            ["pactl", "list", "short", "sources"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return []
    names = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            names.append(parts[1])
    return names


def default_mic_source() -> str:
    """Prefer a non-monitor mic source."""
    sources = list_pulse_sources()
    for s in sources:
        if "monitor" not in s.lower() and ("mic" in s.lower() or "input" in s.lower() or "Line" in s):
            return s
    for s in sources:
        if "monitor" not in s.lower():
            return s
    return "default"


def capture_wav(
    path: PathLike,
    *,
    duration_sec: float = 2.0,
    sample_rate: int = 48000,
    channels: int = 1,
    source: Optional[str] = None,
    backend: str = "auto",
) -> Path:
    """Record audio to WAV. backend: auto | ffmpeg | arecord."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source = source or default_mic_source()
    backend = backend.lower()

    def _ffmpeg() -> None:
        # pulse device
        dev = source if source != "default" else "default"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "pulse", "-i", dev,
            "-t", str(duration_sec),
            "-ar", str(sample_rate),
            "-ac", str(channels),
            str(path),
        ]
        subprocess.check_call(cmd)

    def _arecord() -> None:
        cmd = [
            "arecord",
            "-d", str(int(max(1, round(duration_sec)))),
            "-f", "S16_LE",
            "-r", str(sample_rate),
            "-c", str(channels),
            str(path),
        ]
        subprocess.check_call(cmd)

    errors = []
    if backend in ("auto", "ffmpeg") and shutil.which("ffmpeg"):
        try:
            _ffmpeg()
            return path
        except Exception as e:
            errors.append(f"ffmpeg: {e}")
            if backend == "ffmpeg":
                raise
    if backend in ("auto", "arecord") and shutil.which("arecord"):
        try:
            _arecord()
            return path
        except Exception as e:
            errors.append(f"arecord: {e}")
            if backend == "arecord":
                raise
    raise RuntimeError("capture failed: " + "; ".join(errors) if errors else "no capture backend")


def stereo_to_order1_hoa(
    left: np.ndarray,
    right: np.ndarray,
    *,
    width_az_deg: float = 30.0,
) -> np.ndarray:
    """Pseudo order-1 Ambix from stereo: L at +width, R at -width, mid omni."""
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    n = min(left.shape[0], right.shape[0])
    left, right = left[:n], right[:n]
    mid = 0.5 * (left + right)
    # plane waves
    L = encode_plane_waves([width_az_deg], [0.0], left[None, :], max_order=1)
    R = encode_plane_waves([-width_az_deg], [0.0], right[None, :], max_order=1)
    M = encode_plane_waves([0.0], [0.0], mid[None, :], max_order=1)
    hoa = L + R + 0.5 * M
    # pad to 64
    out = np.zeros((64, n), dtype=np.float64)
    out[:4] = hoa[:4]
    return out


def audio_to_hoa_stream(
    audio: np.ndarray,
    *,
    az_deg: float = 0.0,
    el_deg: float = 0.0,
    max_order: int = 3,
    stereo_width_az: float = 30.0,
) -> np.ndarray:
    """(C,T) HOA from captured audio array (C_in, T) or (T,)."""
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        return encode_mono_plane_wave(a, az_deg, el_deg, max_order=max_order)
    if a.shape[0] == 1:
        return encode_mono_plane_wave(a[0], az_deg, el_deg, max_order=max_order)
    if a.shape[0] >= 4:
        # assume Ambix ACN already
        from .audio_io import ensure_hoa_channels
        return ensure_hoa_channels(a, max_order=max_order)
    if a.shape[0] == 2:
        hoa = stereo_to_order1_hoa(a[0], a[1], width_az_deg=stereo_width_az)
        if max_order < 1:
            return hoa[:1]
        return hoa
    # mixdown
    mono = np.mean(a, axis=0)
    return encode_mono_plane_wave(mono, az_deg, el_deg, max_order=max_order)


def live_report(
    *,
    duration_sec: float = 2.0,
    sample_rate: int = 48000,
    channels: int = 1,
    source: Optional[str] = None,
    az_deg: float = 0.0,
    el_deg: float = 0.0,
    max_order: int = 3,
    keep_wav: Optional[PathLike] = None,
) -> SpatialReport:
    """Capture from mic and return a SpatialReport."""
    if keep_wav:
        wav_path = Path(keep_wav)
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_path = Path(tmp.name)
        tmp.close()
    try:
        capture_wav(
            wav_path,
            duration_sec=duration_sec,
            sample_rate=sample_rate,
            channels=channels,
            source=source,
        )
        audio, sr = read_wav(wav_path)
        hoa = audio_to_hoa_stream(
            audio, az_deg=az_deg, el_deg=el_deg, max_order=max_order
        )
        rep = report_from_hoa(
            hoa,
            sr,
            max_order=max_order,
            notes=[
                f"live capture {duration_sec}s ch={audio.shape[0]} src={source or 'default'}",
                f"encode az={az_deg} el={el_deg}" if audio.shape[0] < 4 else "ambix/multi-ch path",
            ],
            meta={
                "encode": "live_mic",
                "source": source or default_mic_source(),
                "wav": str(wav_path) if keep_wav else None,
            },
        )
        return rep
    finally:
        if not keep_wav:
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass


def live_report_from_file(
    path: PathLike,
    *,
    az_deg: float = 0.0,
    el_deg: float = 0.0,
    max_order: int = 3,
) -> SpatialReport:
    """Same pipeline as live_report but from an existing WAV (offline test)."""
    audio, sr = read_wav(path)
    hoa = audio_to_hoa_stream(audio, az_deg=az_deg, el_deg=el_deg, max_order=max_order)
    return report_from_hoa(
        hoa,
        sr,
        max_order=max_order,
        notes=[f"live pipeline on file {path}"],
        meta={"encode": "live_from_file", "path": str(path)},
    )
