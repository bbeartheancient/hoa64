"""WAV read/write without scipy/soundfile (stdlib wave + numpy).

Supports mono, multi-channel (e.g. Ambix 4/16/64), and float32/int16.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from typing import Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def read_wav(path: PathLike) -> Tuple[np.ndarray, int]:
    """Read a WAV file.

    Returns
    -------
    audio : np.ndarray
        Shape (n_channels, n_samples), float64 in roughly [-1, 1].
    sample_rate : int
    """
    path = Path(path)
    with wave.open(str(path), "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sw == 2:
        mono = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sw == 4:
        # Try 32-bit int first; if file is float32 PCM some writers use WAVE_FORMAT_IEEE_FLOAT
        # which wave may still hand us as bytes — detect by scale.
        arr_i = np.frombuffer(raw, dtype="<i4")
        # Heuristic: if max abs > 2^30-ish it's int; else might be float bits misread.
        # Prefer IEEE float if values look like floats when reinterpreted.
        arr_f = np.frombuffer(raw, dtype="<f4").astype(np.float64)
        if np.max(np.abs(arr_f)) <= 8.0 and np.max(np.abs(arr_i)) > 1000:
            mono = arr_f
        else:
            mono = arr_i.astype(np.float64) / 2147483648.0
    elif sw == 3:
        # 24-bit packed little-endian
        a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        vals = (
            a[:, 0].astype(np.int32)
            | (a[:, 1].astype(np.int32) << 8)
            | (a[:, 2].astype(np.int32) << 16)
        )
        vals = np.where(vals >= 0x800000, vals - 0x1000000, vals)
        mono = vals.astype(np.float64) / 8388608.0
    elif sw == 1:
        mono = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width {sw}")

    if nch == 1:
        audio = mono.reshape(1, -1)
    else:
        audio = mono.reshape(-1, nch).T.copy()
    return audio, int(sr)


def write_wav(
    path: PathLike,
    audio: np.ndarray,
    sample_rate: int,
    *,
    subtype: str = "pcm16",
) -> None:
    """Write WAV. audio shape (n_channels, n_samples) or (n_samples,)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2:
        raise ValueError("audio must be (C,T) or (T,)")
    nch, n_samples = a.shape
    a = np.clip(a, -1.0, 1.0)

    if subtype == "pcm16":
        pcm = (a.T.reshape(-1) * 32767.0).astype("<i2")
        sw = 2
        raw = pcm.tobytes()
    elif subtype == "float32":
        pcm = a.T.reshape(-1).astype("<f4")
        sw = 4
        raw = pcm.tobytes()
    else:
        raise ValueError("subtype must be pcm16 or float32")

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(sw)
        wf.setframerate(int(sample_rate))
        # wave module doesn't set IEEE float format tag; pcm16 is portable.
        if subtype == "float32":
            # Still write bytes; many tools accept float32 wav with format 3
            # but stdlib wave always writes PCM. Prefer pcm16 for portability.
            pass
        wf.writeframes(raw)


def ensure_hoa_channels(audio: np.ndarray, max_order: int = 7) -> np.ndarray:
    """Pad or truncate multi-channel audio to (max_order+1)**2 Ambix channels."""
    nch = (max_order + 1) ** 2
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    out = np.zeros((nch, a.shape[1]), dtype=np.float64)
    n = min(nch, a.shape[0])
    out[:n] = a[:n]
    return out
