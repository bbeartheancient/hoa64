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
(``volvo``) sources. The four RF labels (``ble``/``wifi``/``zigbee``/
``lora``) are **not** NOISEX-92 files — they are synthesized locally
(see below). ``NOISE_CLASSES`` fixes the label order — index in the
list ≡ class index for the classifier in ``dit_noise`` (append-only).

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

Synthetic RF baseband classes (``ble``/``wifi``/``zigbee``/``lora``)
--------------------------------------------------------------------
These four labels are **synthesized locally** by ``synth_waveform`` — they are
not NOISEX-92 recordings. Physics framing: the front-end Nyquist here is
≈ 10 kHz (fs = 19.98 kHz), so 2.4 GHz / sub-GHz carriers are invisible to this
pipeline. What a microphone-adjacent EM pickup or an SDR *baseband* recording
actually contains is the **envelope-equivalent** signature of the protocol:
burst cadence, duty cycle, and spectral envelope of the baseband waveform.
Each generator therefore models the real protocol timing (BLE advertising
intervals, 802.11 beacon period + SIFS/ACK, 802.15.4 frame lengths with
CSMA-CA backoff, LoRa chirp symbols) mapped into the audio band, with the
occupied-RF-bandwidth shaping standing in for the baseband spectral envelope
(1 MHz GFSK → a few kHz lowpass-shaped noise, etc.). These are physically
motivated *models*, not measurements — do not read their mel signatures as
real RF spectra. ``SYNTH_CLASSES`` names them so training code can branch.
"""
from __future__ import annotations

import pathlib
import urllib.request

import numpy as np

#: Label order ≡ class index for the DiT noise classifier.
#: ORDER PINNED — index ≡ class index; append only, never reorder.
NOISE_CLASSES = [
    "white", "pink", "babble", "factory1", "factory2",
    "buccaneer1", "buccaneer2", "f16", "destroyerengine", "destroyerops",
    "leopard", "m109", "machinegun", "volvo", "hfchannel",
    "ble", "wifi", "zigbee", "lora",
]

#: Classes synthesized locally by ``synth_waveform`` (no NOISEX-92 file).
SYNTH_CLASSES = {"ble", "wifi", "zigbee", "lora"}

#: The 15 SPIB recordings — ``download`` only accepts these.
RECORDED_CLASSES = [c for c in NOISE_CLASSES if c not in SYNTH_CLASSES]

#: Default length of a synthesized class when ``load_noise`` is asked
#: for one. 80 s @ 19.98 kHz ≈ 250 windows at the default hop — enough
#: to sit next to a downsampled NOISEX-92 class without drowning it.
SYNTH_SECONDS = 80.0

#: Fixed per-class seeds so ``load_noise("ble")`` is deterministic
#: across processes (do not use ``hash()`` — it is randomized per run).
_SYNTH_SEEDS = {"ble": 0xB1E, "wifi": 0xF1F1, "zigbee": 0x21BEE, "lora": 0x10AA}

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
    if name in SYNTH_CLASSES:
        raise ValueError(
            f"{name!r} is a synthesized RF class, not a NOISEX-92 file "
            f"(use synth_waveform / load_noise)")
    if name not in RECORDED_CLASSES:
        raise ValueError(
            f"unknown NOISEX-92 noise {name!r}; expected one of {RECORDED_CLASSES}")
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


def load_noise(name: str, cache_dir: pathlib.Path | str | None = None,
               seconds: float | None = None) -> tuple[np.ndarray, int]:
    """Load a class waveform → ``(mono float32 in [-1, 1], 19980)``.

    Recorded NOISEX-92 names go through ``download`` + scipy ``loadmat``.
    The ``.mat`` files carry a single 16-bit integer vector under a
    noise-specific variable name; we pick the largest 1-D variable (skipping
    MATLAB's ``__header__``/``__version__``/``__globals__`` metadata) and
    scale robustly: anything with max |x| > 2 is treated as integer/full-scale
    and divided by its max abs, values already in [−1, 1] are passed through.

    Synthetic RF names (``SYNTH_CLASSES``) never touch the network: they
    are generated by ``synth_waveform`` at ``seconds`` (default
    ``SYNTH_SECONDS``) with a fixed per-class seed so training is
    reproducible. ``cache_dir`` is ignored on that path.
    """
    if name in SYNTH_CLASSES:
        secs = SYNTH_SECONDS if seconds is None else float(seconds)
        rng = np.random.default_rng(_SYNTH_SEEDS[name])
        return synth_waveform(name, secs, FS, rng=rng), FS
    if name not in RECORDED_CLASSES:
        raise ValueError(
            f"unknown noise class {name!r}; expected one of {NOISE_CLASSES}")
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


# ---------------------------------------------------------------------------
# Synthetic RF baseband classes — see the module docstring for the physics
# framing. These model the *envelope-equivalent* signature (burst cadence,
# duty cycle, spectral envelope) of the protocol at audio baseband, NOT the
# RF carrier. All randomness flows through the caller's ``rng``.


def _shaped_noise(n: int, fs: float, f_lo: float, f_hi: float,
                  rng: np.random.Generator) -> np.ndarray:
    """White noise FFT-shaped to the band [f_lo, f_hi] Hz, soft (raised-cosine)
    band edges to avoid brick-wall ringing."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    roll = max(0.10 * (f_hi - f_lo), 20.0)
    mask = np.clip(np.minimum((f - (f_lo - roll)) / roll,
                              ((f_hi + roll) - f) / roll), 0.0, 1.0)
    spec *= mask
    return np.fft.irfft(spec, n)


def _raised_cosine_env(n: int, ramp_frac: float = 0.25) -> np.ndarray:
    """Unit-peak burst envelope of length ``n``, raised-cosine attack/release."""
    env = np.ones(n)
    r = min(max(1, int(round(ramp_frac * n))), n // 2)
    t = np.arange(r) / r
    up = 0.5 - 0.5 * np.cos(np.pi * t)
    env[:r] = up
    env[-r:] = up[::-1]
    return env


def _add_burst(x: np.ndarray, t_s: float, dur_s: float, f_lo: float, f_hi: float,
               fs: float, rng: np.random.Generator, ramp_frac: float = 0.25) -> None:
    """Add one band-shaped noise burst at ``t_s`` (in place).

    The noise is shaped on an 8192-sample pool (FFT frequency resolution
    ≈ 2.4 Hz at 19.98 kHz) and a random ``dur``-long window is taken, so even
    sub-millisecond bursts get proper band shaping."""
    i0 = int(round(t_s * fs))
    m = min(int(round(dur_s * fs)), len(x) - i0)
    if m <= 4:
        return
    src = _shaped_noise(max(8192, 4 * m), fs, f_lo, f_hi, rng)
    j0 = int(rng.integers(0, len(src) - m + 1))
    x[i0:i0 + m] += src[j0:j0 + m] * _raised_cosine_env(m, ramp_frac)


def _synth_ble(n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """BLE advertising surrogate — jittered advertising events (uniform
    20 ms..1 s, the spec's 20 ms..10.24 s fast range) of ~1 ms packets with a
    raised-cosine envelope; ~10 % of events are a 3-burst connection-event
    train (1.25–5 ms spacing). In-burst content is 1 MHz-GFSK-equivalent
    lowpass-shaped noise (~1–2 kHz band at this fs) with a per-event center
    offset standing in for the 40-channel frequency hop. Baseband-equivalence
    caveat: models cadence/envelope, not the 2.4 GHz carrier."""
    x = np.zeros(n)
    t = rng.uniform(0.0, 0.05)
    while t * fs < n:
        n_bursts = 3 if rng.random() < 0.10 else 1
        gap = rng.uniform(0.00125, 0.005)
        for b in range(n_bursts):
            f_lo = rng.uniform(0.5, 3.0) * 1e3   # channel-hop surrogate
            _add_burst(x, t + b * gap, rng.uniform(0.0008, 0.0012),
                       f_lo, f_lo + rng.uniform(1.0, 2.0) * 1e3, fs, rng)
        t += rng.uniform(0.020, 1.0)
    return x


def _synth_wifi(n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """802.11 surrogate — beacon frames every 102.4 ms (±2 ms jitter), each a
    ~0.5 ms wideband flat burst (≈1–9 kHz, the 20 MHz OFDM envelope smeared to
    baseband); data frames of 0.2–4 ms arrive at load-dependent
    (exponential-gap) random times, half of them followed by a 10–16 µs SIFS
    gap and a 0.1 ms ACK burst. Busier duty cycle than zigbee.
    Baseband-equivalence caveat applies (cadence/envelope, not the carrier)."""
    x = np.zeros(n)
    t = rng.uniform(0.0, 0.1024)
    while t * fs < n:                       # beacon train
        _add_burst(x, t, 0.0005, 1e3, 9e3, fs, rng, ramp_frac=0.15)
        t += 0.1024 + rng.uniform(-0.002, 0.002)
    load = rng.uniform(0.15, 0.6)           # offered-load surrogate
    t = rng.uniform(0.0, 0.02)
    while t * fs < n:                       # data + ACK traffic
        dur = rng.uniform(0.0002, 0.004)
        _add_burst(x, t, dur, 1e3, 9e3, fs, rng, ramp_frac=0.15)
        t += dur
        if rng.random() < 0.5:
            t += rng.uniform(10e-6, 16e-6)  # SIFS
            _add_burst(x, t, 0.0001, 1e3, 9e3, fs, rng, ramp_frac=0.15)
            t += 0.0001
        t += float(rng.exponential(0.004 / load))
    return x


def _synth_zigbee(n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """802.15.4 surrogate — short frames ≤ 4.2 ms (127 B PSDU at 250 kb/s)
    after a CSMA-CA-style randomized backoff (1–8 unit backoff periods of
    320 µs), then a long idle (50–500 ms) → low duty cycle. Attack is slower
    than wifi's; in-burst noise is 2 MHz-occupied-equivalent lowpass-shaped
    (~1–2 kHz). Baseband-equivalence caveat applies."""
    x = np.zeros(n)
    t = rng.uniform(0.0, 0.1)
    while t * fs < n:
        t += int(rng.integers(1, 9)) * 0.00032          # CSMA-CA backoff slots
        dur = rng.uniform(0.0005, 0.0042)
        _add_burst(x, t, dur, 200.0, rng.uniform(1e3, 2e3), fs, rng,
                   ramp_frac=0.4)                        # slower attack
        t += dur + rng.uniform(0.05, 0.5)
    return x


def _synth_lora(n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """LoRa surrogate — baseband-equivalent chirp train: each packet is an
    8–12 symbol down-chirp preamble followed by 5–20 payload up-chirps
    sweeping 0.5 → 5 kHz, all symbols of one randomly chosen spreading-factor
    duration per packet (8 / 32 / 128 ms ≈ SF7/SF9/SF12 timing at BW 125 kHz
    compressed to baseband). Renders as a diagonal mel ridge.
    Baseband-equivalence caveat applies."""
    x = np.zeros(n)
    f0, f1 = 500.0, 5000.0
    t = rng.uniform(0.0, 0.2)
    while t * fs < n:
        t_sym = float(rng.choice([0.008, 0.032, 0.128]))
        n_pre = int(rng.integers(8, 13))
        n_pay = int(rng.integers(5, 21))
        for k in range(n_pre + n_pay):
            i0 = int(round(t * fs))
            m = min(int(round(t_sym * fs)), n - i0)
            if m <= 4:
                break
            tt = np.arange(m) / fs
            fa, fb = (f0, f1) if k >= n_pre else (f1, f0)
            ph = 2 * np.pi * (fa * tt + 0.5 * (fb - fa) * tt ** 2 / t_sym)
            ph += rng.uniform(0.0, 2 * np.pi)
            x[i0:i0 + m] += np.sin(ph) * _raised_cosine_env(m, ramp_frac=0.05)
            t += t_sym
        t += rng.uniform(0.1, 0.8)          # inter-packet gap
    return x


_SYNTH = {"ble": _synth_ble, "wifi": _synth_wifi,
          "zigbee": _synth_zigbee, "lora": _synth_lora}


def synth_waveform(name: str, seconds: float, fs: float = FS,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Synthesize ``seconds`` of the baseband-equivalent RF signature ``name``.

    ``name`` must be in ``SYNTH_CLASSES``. Deterministic given ``rng``
    (a fresh ``default_rng()`` otherwise); the length is exactly
    ``int(round(seconds * fs))`` and the result is peak-normalized to 0.9 in
    [−1, 1] float32. See the module docstring for what these signals do and
    do not model (envelope/cadence at baseband, not the RF carrier).
    """
    if name not in SYNTH_CLASSES:
        raise ValueError(f"unknown synthetic class {name!r}; expected one of {sorted(SYNTH_CLASSES)}")
    if rng is None:
        rng = np.random.default_rng()
    n = int(round(seconds * fs))
    x = _SYNTH[name](n, fs, rng)
    peak = float(np.max(np.abs(x))) if n else 0.0
    if peak > 0.0:
        x = x * (0.9 / peak)
    return x.astype(np.float32)


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

    # Synthetic RF baseband classes: determinism, exact length, range, and
    # class order pinning (index ≡ class index — append-only list).
    assert NOISE_CLASSES[:15] == ["white", "pink", "babble", "factory1", "factory2",
                                  "buccaneer1", "buccaneer2", "f16", "destroyerengine",
                                  "destroyerops", "leopard", "m109", "machinegun",
                                  "volvo", "hfchannel"]
    assert NOISE_CLASSES[15:] == ["ble", "wifi", "zigbee", "lora"]
    assert SYNTH_CLASSES <= set(NOISE_CLASSES)
    secs = 4.0
    synth_mels = {}
    for name in sorted(SYNTH_CLASSES):
        a = synth_waveform(name, secs, rng=np.random.default_rng(7))
        b = synth_waveform(name, secs, rng=np.random.default_rng(7))
        assert a.shape == (int(round(secs * FS)),) and a.dtype == np.float32, (name, a.shape)
        assert np.array_equal(a, b), f"{name} not deterministic given rng"
        peak = float(np.abs(a).max())
        assert 0.5 < peak <= 1.0, (name, peak)
        synth_mels[name] = mel_spectrogram(a, FS).mean(axis=1)
        print(f"synth[{name:6s}] OK: {a.size} samples ({a.size / FS:.1f} s), peak {peak:.2f}")

    # lora (diagonal chirp ridge) and wifi (wideband flat bursts) must be
    # separable in mean-mel space.
    def _corr(u, v):
        u, v = u - u.mean(), v - v.mean()
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
    c_lw = _corr(synth_mels["lora"], synth_mels["wifi"])
    assert c_lw < 0.9, c_lw
    print(f"mel separability OK: corr(lora, wifi) = {c_lw:+.3f} < 0.9")

    # load_noise dispatches synth classes locally; download refuses them.
    x_ble, fs_ble = load_noise("ble", seconds=0.5)
    assert fs_ble == FS and x_ble.shape == (int(round(0.5 * FS)),)
    x_ble2, _ = load_noise("ble", seconds=0.5)
    assert np.array_equal(x_ble, x_ble2), "load_noise(ble) not deterministic"
    try:
        download("ble")
        raise AssertionError("download('ble') should refuse synth classes")
    except ValueError as e:
        assert "synthesized" in str(e)
    print("load_noise/download synth dispatch OK")

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
