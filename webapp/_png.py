"""Minimal dependency-free PNG encoder and ±1 matrix renderers.

The webapp ships matrix images as PNG bytes (inline base64 or binary
responses) without pulling in Pillow/matplotlib — a PNG is just zlib'd
scanlines with CRC'd chunks, so ~40 lines of stdlib suffice.  Matrices
are rendered in the terminal-visualizer spirit: bright green (#22c55e)
for +1 on near-black (#0a0f0a) for −1; float grids get a two-sided
blue→black→red ramp (used by Phase 2 heatmaps: Gram matrices, energy
landscapes, PSD maps).

Downsampling is nearest-neighbour with stride ceil(n/max_px) — cheap,
structure-preserving for Hadamard patterns, and the browser upscales
with imageSmoothing off.
"""

from __future__ import annotations

import math
import struct
import zlib

import numpy as np


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_png(rgb: np.ndarray) -> bytes:
    """Encode a uint8 (h, w, 3) array as PNG bytes (8-bit truecolor,
    no interlace, filter type 0 per scanline)."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, c = rgb.shape
    assert c == 3, "expected (h, w, 3) RGB"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw, 6))
        + _chunk(b"IEND", b"")
    )


def _downsample(A: np.ndarray, max_px: int) -> np.ndarray:
    n = int(A.shape[0])
    if n <= max_px:
        return A
    step = int(math.ceil(n / max_px))
    return A[::step, ::step]


def matrix_png(H: np.ndarray, max_px: int = 1024) -> bytes:
    """Render a ±1 matrix as a green-on-black PNG (≤ max_px per side)."""
    A = _downsample(np.asarray(H), max_px)
    pos = A > 0
    rgb = np.empty(A.shape + (3,), dtype=np.uint8)
    rgb[pos] = (0x22, 0xC5, 0x5E)   # #22c55e green
    rgb[~pos] = (0x0A, 0x0F, 0x0A)  # near-black
    return write_png(rgb)


def heatmap_png(grid: np.ndarray, max_px: int = 1024) -> bytes:
    """Render a float 2D grid as a blue→black→red two-sided ramp PNG."""
    A = _downsample(np.asarray(grid, dtype=np.float64), max_px)
    m = float(np.abs(A).max()) if A.size else 0.0
    t = A / m if m > 0 else np.zeros_like(A)  # [-1, 1]
    t = np.clip(t, -1.0, 1.0)
    rgb = np.zeros(A.shape + (3,), dtype=np.uint8)
    pos = t >= 0
    rgb[pos, 0] = (255 * t[pos]).astype(np.uint8)          # red channel up
    rgb[~pos, 2] = (255 * -t[~pos]).astype(np.uint8)       # blue channel up
    return write_png(rgb)


if __name__ == "__main__":
    from ..hadamard import sylvester

    png = matrix_png(sylvester(64))
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "bad PNG magic"
    with open("/tmp/h64.png", "wb") as f:
        f.write(png)
    print(f"wrote /tmp/h64.png ({len(png)} bytes)")
