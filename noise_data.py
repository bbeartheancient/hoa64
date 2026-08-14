"""NOISEX-92 noise-database access and log-mel front-end for the DiT noise classifier.

NOISEX-92 provenance
--------------------
The NOISEX-92 database (Varga & Steeneken, TNO / RSG.10, 1993) is the standard
corpus of *real* recording-environment noises used to benchmark speech
enhancement and noise-robust recognition. The copy used here is the SPIB mirror
at the Federal University of Santa Catarina

    http://spib.linse.ufsc.br/noise.html
    https://spib.linse.ufsc.br/data/noise/{name}.mat

where each file is a MATLAB 5.0 ``.mat`` holding one 16-bit integer waveform,
235 s long, sampled at fs = 19.98 kHz (a Sony PCM deck at 20 kHz nominal,
re-digitized at the slightly odd 19.98 kHz). The 15 noises span stationary
(``white``, ``pink``, ``hfchannel``), military vehicle/aircraft (``f16``,
``buccaneer1/2``, ``leopard``, ``m109``, ``destroyerengine``), industrial
(``factory1/2``, ``machinegun``), speech-like (``babble``) and domestic
(``volvo``) sources. ``NOISE_CLASSES`` below fixes the label order — index in
the list ≡ class index for the classifier in ``dit_noise``.

Mel math
--------
The mel filterbank uses the HTK (Hidden Markov Model Toolkit) convention,

    mel(f) = 2595 · log10(1 + f / 700),

with triangular filters whose centers are equally spaced on the mel axis
between ``f_min`` and ``f_max`` (default: full Nyquist range) and whose
vertices sit on the FFT bin grid of an ``n_fft``-point transform. The
spectrogram is a Hann-windowed STFT magnitude-squared (power spectrum),
``S = |rfft(x · w)|²``, projected onto the filterbank.

Log-mel normalization (the classifier depends on this)
------------------------------------------------------
``mel_spectrogram`` returns log-mel in **decibels with a fixed affine
normalization**, not a per-window standardization:

    db  = 10 · log10(S_mel + 1e-12)
    out = clip((db − DB_MEAN) / DB_STD, −4, 4)      DB_MEAN = +10, DB_STD = 15

The constants are global, chosen from the observed range of full-scale
[−1, 1] audio (active frames land ≈ [−5, +40] dB → ≈ [−1, +2] normalized;
only near-silent frames hit the −4 clip). Because the same affine is applied
at train and inference time the network sees a stationary input distribution;
the spectral *tilt* (which separates white from pink, say) survives any
affine map. Inputs are expected pre-scaled to [−1, 1] (as ``load_noise``
produces).
"""
from __future__ import annotations

import pathlib
import urllib.request

import numpy as np

#: Label order ≡ class index for the DiT noise classifier.
NOISE_CLASSES = [
    "white", "pink", "babble", "factory1", "factory2",
    "buccaneer1", "buccaneer2", "f16", "destroyerengine", "destroyerops",
    "leopard", "m109", "machinegun", "volvo", "hfchannel",
]

#: NOISEX-92 sample rate (the SPIB files are all single-rate).
FS = 19980

_BASE_URL = "https://spib.linse.ufsc.br/data/noise/{}.mat"
_DEFAULT_CACHE = pathlib.Path("~/.cache/hoa64/noisex92").expanduser()

#: Fixed log-mel dB normalization — see the module docstring. The classifier
#: is trained against exactly this transform; do not rescale per window.
DB_MEAN = 10.0
DB_STD = 15.0


def _cache(cache_dir: pathlib.Path | str | None) -> pathlib.Path:
    p = pathlib.Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE
    p.mkdir(parents=True, exist_ok=True)
    return p


def download(name: str, cache_dir: pathlib.Path | str | None = None) -> pathlib.Path:
    """Fetch ``{name}.mat`` from the SPIB NOISEX-92 mirror into the cache.

    Lazy and idempotent: a file already present with a plausible size
    (> 1 MB — the real files are ~9 MB) is returned untouched. Streams to a
    ``.part`` file and renames on completion so an interrupted download never
    masquerades as a good one.
    """
    if name not in NOISE_CLASSES:
        raise ValueError(f"unknown NOISEX-92 noise {name!r}; expected one of {NOISE_CLASSES}")
    dest = _cache(cache_dir) / f"{name}.mat"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    tmp = dest.with_suffix(".mat.part")
    with urllib.request.urlopen(_BASE_URL.format(name), timeout=120) as r, tmp.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dest)
    return dest


def load_noise(name: str, cache_dir: pathlib.Path | str | None = None) -> tuple[np.ndarray, int]:
    """Load a cached NOISEX-92 file → ``(mono float32 in [-1, 1], 19980)``.

    The ``.mat`` files carry a single 16-bit integer vector under a
    noise-specific variable name; we pick the largest 1-D variable (skipping
    MATLAB's ``__header__``/``__version__``/``__globals__`` metadata) and
    scale robustly: anything with max |x| > 2 is treated as integer/full-scale
    and divided by its max abs, values already in [−1, 1] are passed through.
    """
    from scipy.io import loadmat  # lazy: scipy is only needed on this path

    path = download(name, cache_dir)
    mat = loadmat(str(path))
    best = None
    for key, val in mat.items():
        if key.startswith("__") or not isinstance(val, np.ndarray):
            continue
        v = np.asarray(val)
        if v.ndim == 2 and 1 in v.shape:
            v = v.ravel()
        if v.ndim != 1 or not np.issubdtype(v.dtype, np.number):
            continue
        if best is None or v.size > best.size:
            best = v
    if best is None:
        raise ValueError(f"no 1-D waveform variable found in {path}")
    x = best.astype(np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 2.0:
        x /= peak
    return x, FS


def _mel(f: np.ndarray | float) -> np.ndarray:
    """HTK mel: mel(f) = 2595 · log10(1 + f/700)."""
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_inv(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(fs: float, n_fft: int = 1024, n_mels: int = 64,
                   f_min: float = 20.0, f_max: float | None = None) -> np.ndarray:
    """Triangular HTK-mel filterbank, shape ``(n_mels, n_fft//2 + 1)`` float32.

    Filter vertices are FFT-bin-quantized mel-spaced frequencies; each row is
    a unit-height triangle between its left/center/right vertices.
    """
    if f_max is None:
        f_max = fs / 2.0
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, fs / 2.0, n_freqs)
    verts = _mel_inv(np.linspace(_mel(f_min), _mel(f_max), n_mels + 2))
    fb = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for i in range(n_mels):
        lo, c, hi = verts[i], verts[i + 1], verts[i + 2]
        up = (fft_freqs - lo) / max(c - lo, 1e-12)
        down = (hi - fft_freqs) / max(hi - c, 1e-12)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    return fb.astype(np.float32)


def mel_spectrogram(x: np.ndarray, fs: float, n_fft: int = 1024, hop: int = 512,
                    n_mels: int = 64) -> np.ndarray:
    """Log-mel spectrogram of ``x``, shape ``(n_mels, T)`` float32.

    Hann-windowed STFT power spectrum → mel filterbank → dB → **fixed**
    normalization ``clip((dB − DB_MEAN)/DB_STD, −4, 4)`` (see the module
    docstring — the DiT classifier is trained against exactly this transform).
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    n_frames = 1 + (x.size - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * np.hanning(n_fft).astype(np.float32)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2          # (T, n_fft//2+1)
    s_mel = power @ mel_filterbank(fs, n_fft, n_mels).T       # (T, n_mels)
    db = 10.0 * np.log10(s_mel + 1e-12)
    out = np.clip((db - DB_MEAN) / DB_STD, -4.0, 4.0)
    return out.T.astype(np.float32)                           # (n_mels, T)


def frame_windows(x: np.ndarray, fs: float, win_s: float = 0.65,
                  hop_s: float = 0.32) -> list[slice]:
    """Overlapping analysis windows over ``x`` as a list of ``slice`` objects.

    Window/hop are given in seconds (defaults 0.65 s / 0.32 s ≈ 13k/6.4k
    samples at 19.98 kHz); a final partial window is dropped.
    """
    n = len(x)
    win = int(round(win_s * fs))
    hop = int(round(hop_s * fs))
    if win <= 0 or hop <= 0:
        raise ValueError("win_s and hop_s must be positive")
    return [slice(s, s + win) for s in range(0, n - win + 1, hop)]


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """Cheap 1/f (pink) noise via FFT spectral shaping."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0.0
    x = np.fft.irfft(spec, n)
    return (x / np.max(np.abs(x))).astype(np.float32)


if __name__ == "__main__":
    # Network-free self-check on synthesized signals.
    fs = FS
    rng = np.random.default_rng(0)
    n = 2 * fs
    white = rng.standard_normal(n).astype(np.float32)
    white /= np.max(np.abs(white))
    pink = _pink(n, rng)
    t = np.arange(n) / fs
    tone = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)

    fb = mel_filterbank(fs)
    assert fb.shape == (64, 513) and np.all(fb >= 0) and np.all(fb.max(axis=1) > 0.5)
    print(f"filterbank OK: shape {fb.shape}, peak {fb.max():.2f}")

    mels = {k: mel_spectrogram(v, fs) for k, v in
            [("white", white), ("pink", pink), ("tone", tone)]}
    for k, m in mels.items():
        assert m.shape[0] == 64 and m.dtype == np.float32, (k, m.shape)
        assert np.all(np.isfinite(m)), k
        assert m.min() >= -4.0 and m.max() <= 4.0, k
        print(f"mel[{k:5s}] shape {m.shape}, range [{m.min():+.2f}, {m.max():+.2f}]")

    # White is spectrally flat, pink tilts down ~3 dB/octave → the mean of the
    # top-16 mel bands must be lower for pink than for white.
    hi = slice(-16, None)
    w_hi, p_hi = mels["white"][hi].mean(), mels["pink"][hi].mean()
    assert p_hi < w_hi, (p_hi, w_hi)
    print(f"tilt OK: high-mel mean white {w_hi:+.2f} > pink {p_hi:+.2f}")

    wins = frame_windows(white, fs)
    expected = 1 + (n - int(0.65 * fs)) // int(0.32 * fs)
    assert len(wins) == expected and wins[0] == slice(0, int(0.65 * fs))
    assert all((w.stop - w.start) == int(0.65 * fs) for w in wins)
    print(f"frame_windows OK: {len(wins)} windows of {wins[0].stop - wins[0].start} samples")

    # Optional real-file sanity check (only if already cached — no network).
    cached = _DEFAULT_CACHE / "white.mat"
    if cached.exists():
        x, fs_real = load_noise("white")
        rms = float(np.sqrt(np.mean(x ** 2)))
        assert fs_real == FS and x.ndim == 1 and x.size > 4_000_000
        assert x.max() <= 1.0 and x.min() >= -1.0 and 0.05 < rms < 1.0
        print(f"real white.mat OK: {x.size} samples, {x.size / fs_real:.1f} s, rms {rms:.3f}")
    else:
        print("SKIP: ~/.cache/hoa64/noisex92/white.mat not cached, real-load check skipped")

    print("noise_data self-check OK")
