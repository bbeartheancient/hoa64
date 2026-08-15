"""Live radio-telemetry capture → baseband waveform for the noise analyzer.

Physics framing (be honest about what this is)
----------------------------------------------
Without an SDR or monitor-mode interface the GHz carriers are invisible to
this machine: the analyzer front-end Nyquist is ≈ 10 kHz (fs = 19.98 kHz,
see `noise_data`), and the iwlwifi driver here offers no channel survey or
monitor mode. What the local radios *do* expose unprivileged is their own
activity bookkeeping:

* wifi — the managed-mode interface's /proc/net/dev counters (rx/tx packets
  and bytes), i.e. **this station's traffic as the driver counts it**, not
  full-channel on-air energy. Beacons from other BSSs, neighboring-cell
  traffic, and collisions never appear.
* ble — the HCI device's counters via the ``HCIGETDEVINFO`` ioctl (the same
  numbers ``hciconfig -a`` prints: commands, events, ACL/SCO packets, bytes).

The capture loop polls those counters at ~100 Hz, turns the measured deltas
into a duty envelope (a tick is *active* when any packet/event delta is
nonzero, with level ∝ the byte delta), and renders that envelope into audio
baseband by gating band-shaped noise (`noise_data._shaped_noise`) with the
measured on/off cadence. This is exactly the **envelope-equivalent model
family** of `noise_data.synth_waveform` — cadence/duty/spectral envelope, not
the carrier — except the timing is *measured from the real radio* instead of
scripted by a protocol model. A live capture therefore lands in the same mel
region the classifier's synth RF classes were trained on, but its burst
pattern is whatever the radio actually did. Do not read the result as an RF
spectrum; it is a baseband render of MAC/HCI telemetry.

Sources with no local radio (zigbee, lora on this machine) are reported by
`live_sources` as unavailable with an explicit reason — never faked. An
all-idle capture returns the near-silent waveform with
``stats["activity"] = 0`` and a note; traffic is never fabricated.
"""
from __future__ import annotations

import fcntl
import socket
import struct
import time
from pathlib import Path

import numpy as np

if __package__:                                 # normal package import
    from . import noise_data
else:                                           # direct `python hoa64/rf_capture.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from hoa64 import noise_data

#: Target counter-poll rate. 100 Hz gives 10 ms envelope resolution — fine
#: enough for beacon/advertising cadence, cheap enough for /proc + ioctl.
POLL_HZ = 100.0

#: Baseband render bands (Hz) per RF source — matched to the
#: ``noise_data._synth_*`` generators so a live render lands in the same
#: mel region the classifier was trained on.
_BANDS = {"wifi": (1.0e3, 9.0e3), "ble": (0.5e3, 5.0e3)}

#: HCIGETDEVINFO — _IOR(HCI_IOC_MAGIC, ...) on struct hci_dev_info; verified
#: against `hciconfig` output on this machine.
_HCIGETDEVINFO = 0x800448D3


# ---------------------------------------------------------------- detection

def _wireless_ifaces() -> list[str]:
    """Wireless interface names: the /sys/class/net/*/wireless marker, with a
    /proc/net/wireless fallback."""
    names = sorted(p.name for p in Path("/sys/class/net").iterdir()
                   if (p / "wireless").exists())
    if not names and Path("/proc/net/wireless").exists():
        for line in Path("/proc/net/wireless").read_text().splitlines()[2:]:
            if ":" in line:
                names.append(line.split(":", 1)[0].strip())
    return names


def _bt_devs() -> list[int]:
    """Local Bluetooth HCI device ids (``hci<N>`` entries in /sys/class/bluetooth)."""
    root = Path("/sys/class/bluetooth")
    if not root.exists():
        return []
    return sorted(int(p.name[3:]) for p in root.iterdir()
                  if p.name.startswith("hci") and p.name[3:].isdigit())


def live_sources() -> dict:
    """Availability of each analyzer live-capture source on this machine.

    Keys: ``mic`` (always listed; backend availability is live_audio's
    business), ``wifi``, ``ble``, ``zigbee``, ``lora``. Unavailable sources
    carry a ``reason``; available RF sources carry the iface/dev to poll and
    the baseband render band.
    """
    ifaces = _wireless_ifaces()
    bt = _bt_devs()
    return {
        "mic": {"available": True,
                "detail": "default audio capture backend (live_audio)"},
        "wifi": ({"available": True, "iface": ifaces[0],
                  "band_hz": list(_BANDS["wifi"]),
                  "detail": f"{ifaces[0]} /proc/net/dev counters — this station's "
                            f"traffic as the driver counts it, not on-air energy"}
                 if ifaces else
                 {"available": False, "iface": None,
                  "reason": "no wireless interface (/sys/class/net/*/wireless)"}),
        "ble": ({"available": True, "dev": f"hci{bt[0]}", "dev_id": bt[0],
                 "band_hz": list(_BANDS["ble"]),
                 "detail": f"hci{bt[0]} HCI counters via HCIGETDEVINFO ioctl "
                           f"(cmd/evt/ACL/SCO, bytes)"}
                if bt else
                {"available": False, "dev": None,
                 "reason": "no bluetooth HCI device (/sys/class/bluetooth)"}),
        "zigbee": {"available": False,
                   "reason": "no local 802.15.4 radio"},
        "lora": {"available": False,
                 "reason": "no local LoRa radio"},
    }


# ------------------------------------------------------------ counter reads

def _net_dev_counters(iface: str) -> dict:
    """Cumulative rx/tx packet+byte counters for ``iface`` from /proc/net/dev.

    Layout per line (after ``iface:``): rx_bytes rx_packets …(6 more)…
    tx_bytes tx_packets … — we want fields 0, 1, 8, 9.
    """
    for line in Path("/proc/net/dev").read_text().splitlines():
        if ":" not in line:
            continue
        name, rest = line.split(":", 1)
        if name.strip() != iface:
            continue
        f = rest.split()
        return {"rx_bytes": int(f[0]), "rx_packets": int(f[1]),
                "tx_bytes": int(f[8]), "tx_packets": int(f[9])}
    raise ValueError(f"interface {iface!r} not found in /proc/net/dev")


def _hci_counters(dev_id: int) -> dict:
    """Cumulative HCI counters for ``hci<dev_id>`` via HCIGETDEVINFO.

    ``struct hci_dev_info`` (92 bytes): dev_id at 0, name at 2, then the
    counters block at offset 52 — err_rx err_tx cmd_tx evt_rx acl_tx acl_rx
    sco_tx sco_rx byte_rx byte_tx (10 × u32). Same numbers as hciconfig.
    """
    # numeric fallbacks: some python builds don't expose AF_BLUETOOTH
    af_bt = getattr(socket, "AF_BLUETOOTH", 31)
    proto_hci = getattr(socket, "BTPROTO_HCI", 1)
    s = socket.socket(af_bt, socket.SOCK_RAW, proto_hci)
    try:
        buf = bytearray(92)
        struct.pack_into("<H", buf, 0, dev_id)
        fcntl.ioctl(s, _HCIGETDEVINFO, buf, True)
        keys = ("err_rx", "err_tx", "cmd_tx", "evt_rx", "acl_tx", "acl_rx",
                "sco_tx", "sco_rx", "byte_rx", "byte_tx")
        return dict(zip(keys, struct.unpack_from("<10I", buf, 52)))
    finally:
        s.close()


def _wifi_tick(c: dict) -> tuple[int, int]:
    """(packets, bytes) activity projection of a /proc/net/dev sample."""
    return c["rx_packets"] + c["tx_packets"], c["rx_bytes"] + c["tx_bytes"]


def _hci_tick(c: dict) -> tuple[int, int]:
    """(events, bytes) activity projection of an HCI counters sample."""
    return (c["acl_tx"] + c["acl_rx"] + c["sco_tx"] + c["sco_rx"]
            + c["evt_rx"] + c["cmd_tx"]), c["byte_rx"] + c["byte_tx"]


def _reader(source: str, info: dict):
    """Zero-arg closure returning the cumulative (events, bytes) projection."""
    if source == "wifi":
        iface = info["iface"]
        return lambda: _wifi_tick(_net_dev_counters(iface))
    dev_id = info["dev_id"]
    return lambda: _hci_tick(_hci_counters(dev_id))


# ---------------------------------------------------------------- rendering

def _envelope(deltas: list[tuple[int, int]]) -> np.ndarray:
    """Per-tick activity envelope in [0, 1] from (event, byte) counter deltas.

    A tick is *active* when any packet/event delta is nonzero; its level is
    0.25 + 0.75 · bytes/max_bytes so single small frames stay audible while
    bulk transfers saturate the envelope.
    """
    ev = np.asarray([d[0] for d in deltas], dtype=np.float64)
    by = np.asarray([d[1] for d in deltas], dtype=np.float64)
    bmax = float(by.max()) if by.size else 0.0
    level = by / bmax if bmax > 0.0 else np.ones_like(by)
    return np.where(ev > 0.0, 0.25 + 0.75 * level, 0.0)


def _render(env: np.ndarray, poll_hz: float, band: tuple[float, float],
            seconds: float, fs: float,
            rng: np.random.Generator) -> np.ndarray:
    """Render a tick envelope into audio baseband.

    Repetition-upsample the ~100 Hz envelope to ``fs``, smooth with a ~3 ms
    raised-cosine (Hann) lowpass to de-click the 10 ms steps, gate
    band-shaped noise with it, peak-normalize to 0.9. An all-zero envelope
    renders as exact digital silence.
    """
    n = int(round(seconds * fs))
    if n <= 0 or env.size == 0:
        return np.zeros(max(n, 0), dtype=np.float32)
    idx = np.minimum((np.arange(n) * (poll_hz / fs)).astype(np.int64),
                     env.size - 1)
    up = env[idx]
    m = max(3, int(round(0.003 * fs)) | 1)      # ~3 ms odd-length Hann
    w = np.hanning(m)
    w /= w.sum()
    sm = np.convolve(up, w, mode="same")
    if float(sm.max()) > 0.0:
        sm *= float(env.max()) / float(sm.max())
    x = sm * noise_data._shaped_noise(n, fs, band[0], band[1], rng)
    peak = float(np.max(np.abs(x)))
    if peak > 0.0:
        x = x * (0.9 / peak)
    return x.astype(np.float32)


# ------------------------------------------------------------------ capture

def capture(source: str, seconds: float, fs: float = noise_data.FS,
            rng: np.random.Generator | None = None,
            ) -> tuple[np.ndarray, int, dict]:
    """Capture ``seconds`` of live radio telemetry → ``(waveform, fs, stats)``.

    Polls the source's counters at ~POLL_HZ (paced on time.monotonic; the
    achieved rate is measured, not assumed), converts per-tick deltas to a
    duty envelope and renders it at baseband via `_render`. ``stats`` carries
    source, iface/dev, seconds, achieved poll_hz, ticks, packets (wifi) or
    events (ble), bytes, duty (fraction of active ticks), and ``activity`` —
    0 with a "no radio traffic observed" note when every delta was zero
    (the returned waveform is then near-silent; nothing is fabricated).
    Raises ValueError for unknown/unavailable sources.
    """
    srcs = live_sources()
    if source == "mic":
        raise ValueError("the mic source is captured by live_audio, not rf_capture")
    if source not in srcs:
        raise ValueError(
            f"unknown live source {source!r}; expected one of {sorted(srcs)}")
    info = srcs[source]
    if not info.get("available"):
        raise ValueError(f"live source {source!r} unavailable: "
                         f"{info.get('reason', 'not detected')}")
    seconds = float(seconds)
    if not (0.05 <= seconds <= 60.0):
        raise ValueError("seconds must be 0.05..60")

    read = _reader(source, info)
    try:
        prev_e, prev_b = read()
    except (OSError, ValueError) as e:
        raise ValueError(f"cannot read {source} counters: {e}") from e

    period = 1.0 / POLL_HZ
    t0 = time.monotonic()
    next_t = t0
    deltas: list[tuple[int, int]] = []
    while True:
        next_t += period
        delay = next_t - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)
        if time.monotonic() - t0 >= seconds:
            break
        try:
            e, b = read()
        except (OSError, ValueError):
            continue            # transient read failure — hold previous count
        deltas.append((max(0, e - prev_e), max(0, b - prev_b)))
        prev_e, prev_b = e, b
    elapsed = time.monotonic() - t0
    ticks = len(deltas)

    env = _envelope(deltas)
    rng = np.random.default_rng() if rng is None else rng
    x = _render(env, ticks / elapsed if ticks and elapsed > 0 else POLL_HZ,
                _BANDS[source], seconds, fs, rng)

    total = int(sum(d[0] for d in deltas))
    stats: dict = {
        "source": source,
        "seconds": seconds,
        "poll_hz": round(ticks / elapsed, 1) if ticks and elapsed > 0 else 0.0,
        "ticks": ticks,
        "bytes": int(sum(d[1] for d in deltas)),
        "duty": round(float(np.mean(env > 0)) if env.size else 0.0, 4),
        "activity": 1 if total > 0 else 0,
    }
    if source == "wifi":
        stats["iface"] = info["iface"]
        stats["packets"] = total
    else:
        stats["dev"] = info["dev"]
        stats["events"] = total
    if not stats["activity"]:
        stats["note"] = "no radio traffic observed"
    return x, int(fs), stats


if __name__ == "__main__":
    # Hardware-free render checks on fabricated tick deltas (deterministic
    # rng), then live_sources + a real ~0.5 s smoke capture where the radios
    # exist (no assertions on live traffic — idle is a fine result).
    fs = noise_data.FS
    seconds = 1.0
    n_ticks = int(seconds * POLL_HZ)

    r = np.random.default_rng(42)
    busy = [(0, 0)] * n_ticks
    for _ in range(30):                     # ~30 % duty: bursts of 1–4 ticks
        i = int(r.integers(0, n_ticks))
        for k in range(int(r.integers(1, 5))):
            if i + k < n_ticks:
                busy[i + k] = (int(r.integers(1, 9)), int(r.integers(40, 1500)))
    idle = [(0, 0)] * n_ticks

    for name, deltas in [("busy", busy), ("idle", idle)]:
        env = _envelope(deltas)
        assert env.shape == (n_ticks,) and np.all((0.0 <= env) & (env <= 1.0))
        x = _render(env, POLL_HZ, _BANDS["wifi"], seconds, fs,
                    np.random.default_rng(1))
        assert x.shape == (int(round(seconds * fs)),) and x.dtype == np.float32
        assert np.all(np.isfinite(x)) and float(np.abs(x).max()) <= 1.0
        print(f"render[{name}] OK: {x.size} samples, "
              f"peak {float(np.abs(x).max()):.2f}, duty {float((env > 0).mean()):.2f}")

    # a bursty capture must be separable from an idle one in log-mel space
    m_busy = noise_data.mel_spectrogram(
        _render(_envelope(busy), POLL_HZ, _BANDS["wifi"], seconds, fs,
                np.random.default_rng(2)), fs)
    m_idle = noise_data.mel_spectrogram(
        _render(_envelope(idle), POLL_HZ, _BANDS["wifi"], seconds, fs,
                np.random.default_rng(2)), fs)
    assert not np.allclose(m_busy, m_idle)
    assert m_busy.mean() > m_idle.mean(), (m_busy.mean(), m_idle.mean())
    print(f"mel separation OK: mean busy {m_busy.mean():+.2f} > idle {m_idle.mean():+.2f}")

    # unavailable/unknown sources must refuse loudly, never fabricate
    for bad in ["mic", "zigbee", "lora", "nope"]:
        try:
            capture(bad, 0.1)
            raise AssertionError(f"capture({bad!r}) should raise ValueError")
        except ValueError as e:
            print(f"capture({bad!r}) refused: {e}")

    srcs = live_sources()
    for name, s in srcs.items():
        print(f"live_sources[{name}]: {s}")

    for name in ("wifi", "ble"):
        if srcs[name]["available"]:
            x, fs2, stats = capture(name, 0.5)
            assert x.shape == (int(round(0.5 * fs)),) and x.dtype == np.float32
            assert fs2 == fs
            print(f"live {name} smoke capture OK: {stats}")
        else:
            print(f"SKIP live {name}: {srcs[name].get('reason')}")

    print("rf_capture self-check OK")
