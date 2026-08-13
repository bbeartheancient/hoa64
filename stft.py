"""Minimal STFT / ISTFT in pure NumPy (no scipy)."""

from __future__ import annotations

import numpy as np


def hann_window(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(n, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / n)


def stft(
    x: np.ndarray,
    *,
    n_fft: int = 1024,
    hop: int = 256,
    window: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Short-time FFT.

    Parameters
    ----------
    x : (n_samples,) real
    Returns
    -------
    freqs_bins : (n_fft//2+1,)  (normalized later by caller with sr)
    S : complex64 (n_bins, n_frames)
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if window is None:
        window = hann_window(n_fft)
    window = np.asarray(window, dtype=np.float64)
    if window.shape[0] != n_fft:
        raise ValueError("window length must equal n_fft")

    if x.shape[0] < n_fft:
        x = np.pad(x, (0, n_fft - x.shape[0]))

    n_frames = 1 + (x.shape[0] - n_fft) // hop
    n_bins = n_fft // 2 + 1
    S = np.empty((n_bins, n_frames), dtype=np.complex128)
    for i in range(n_frames):
        start = i * hop
        frame = x[start : start + n_fft] * window
        spec = np.fft.rfft(frame, n=n_fft)
        S[:, i] = spec
    return S


def stft_freqs(n_fft: int, sample_rate: int) -> np.ndarray:
    return np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)


def frame_signal(
    x: np.ndarray,
    *,
    frame_len: int,
    hop: int,
    window: np.ndarray | None = None,
) -> np.ndarray:
    """Slice a 1-D signal into overlapping frames (n_frames, frame_len)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if window is None:
        window = hann_window(frame_len)
    if x.shape[0] < frame_len:
        x = np.pad(x, (0, frame_len - x.shape[0]))
    n_frames = 1 + (x.shape[0] - frame_len) // hop
    out = np.empty((n_frames, frame_len), dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        out[i] = x[start : start + frame_len] * window
    return out


def frame_multichannel(
    audio: np.ndarray,
    *,
    frame_len: int,
    hop: int,
) -> np.ndarray:
    """Frame multi-channel audio (C, T) → (n_frames, C, frame_len) rectangular (no window)."""
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("audio must be (C,T)")
    C, T = a.shape
    if T < frame_len:
        a = np.pad(a, ((0, 0), (0, frame_len - T)))
        T = a.shape[1]
    n_frames = 1 + (T - frame_len) // hop
    out = np.empty((n_frames, C, frame_len), dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        out[i] = a[:, start : start + frame_len]
    return out
