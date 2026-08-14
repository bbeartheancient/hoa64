"""Minimal dependency-free PNG encoder/decoder and ±1 matrix renderers.

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

`decode_png` is the matching reader for the subset used by the AWS Open
Data Terrain Tiles (Mapzen Terrarium): 8-bit truecolor / truecolor+alpha
(color types 2 and 6), non-interlaced, all five scanline filters
(None/Sub/Up/Average/Paeth).  Palette, 16-bit and Adam7-interlaced PNGs
are rejected with a clear error.
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


# ---------------------------------------------------------------- decoder

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _paeth(a: int, b: int, c: int) -> int:
    """Paeth predictor: nearest of left/above/upper-left to a + b − c."""
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def _unfilter(raw: bytes, h: int, stride: int, bpp: int) -> np.ndarray:
    """Reconstruct scanlines from filtered bytes → (h, stride) uint8.

    Each scanline is prefixed by one filter-type byte.  Sub/Up are
    vectorized (per-channel cumulative sum / previous-row add); Average
    and Paeth have a sequential left-neighbour dependency and run as a
    compact per-byte loop (a 256 px tile row is ≤ 1 KiB, so this stays
    fast enough for terrain tiles).
    """
    out = np.empty((h, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint16)
    off = 0
    for y in range(h):
        f = raw[off]
        off += 1
        cur = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=off).astype(np.uint16)
        off += stride
        if f == 0:                      # None
            pass
        elif f == 1:                    # Sub: recon[i] = raw[i] + recon[i−bpp]
            cur = cur.reshape(-1, bpp).cumsum(axis=0).reshape(-1) & 0xFF
        elif f == 2:                    # Up: recon[i] = raw[i] + prev[i]
            cur = (cur + prev) & 0xFF
        elif f == 3:                    # Average
            for i in range(stride):
                a = cur[i - bpp] if i >= bpp else 0
                cur[i] = (cur[i] + ((a + int(prev[i])) >> 1)) & 0xFF
        elif f == 4:                    # Paeth
            for i in range(stride):
                a = int(cur[i - bpp]) if i >= bpp else 0
                b = int(prev[i])
                c = int(prev[i - bpp]) if i >= bpp else 0
                cur[i] = (cur[i] + _paeth(a, b, c)) & 0xFF
        else:
            raise ValueError(f"unknown PNG scanline filter type {f}")
        prev = cur
        out[y] = cur.astype(np.uint8)
    return out


def decode_png(data: bytes) -> np.ndarray:
    """Decode PNG bytes to a uint8 (h, w, 3) RGB array.

    Supports the subset the Terrarium elevation tiles use: 8-bit
    truecolor (color type 2) and truecolor+alpha (type 6, alpha
    dropped), non-interlaced, all five filter types.  Raises ValueError
    for palette/16-bit/interlaced PNGs or a malformed stream.
    """
    if not data.startswith(_PNG_MAGIC):
        raise ValueError("not a PNG (bad magic)")
    pos = len(_PNG_MAGIC)
    w = h = bit_depth = color_type = interlace = None
    idat = bytearray()
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            (w, h, bit_depth, color_type, _comp, _filt,
             interlace) = struct.unpack(">IIBBBBB", body)
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            break
    if w is None:
        raise ValueError("no IHDR chunk")
    if bit_depth != 8:
        raise ValueError(f"unsupported bit depth {bit_depth} (need 8)")
    if color_type not in (2, 6):
        raise ValueError(f"unsupported color type {color_type} "
                         "(need 2=RGB or 6=RGBA; no palette/gray)")
    if interlace != 0:
        raise ValueError("interlaced (Adam7) PNGs not supported")
    if not idat:
        raise ValueError("no IDAT chunks")
    bpp = 3 if color_type == 2 else 4
    raw = zlib.decompress(bytes(idat))
    stride = w * bpp
    if len(raw) != h * (stride + 1):
        raise ValueError(f"bad image data length {len(raw)} "
                         f"(expected {h * (stride + 1)})")
    img = _unfilter(raw, h, stride, bpp).reshape(h, w, bpp)
    return np.ascontiguousarray(img[:, :, :3])


if __name__ == "__main__":
    from ..hadamard import sylvester

    png = matrix_png(sylvester(64))
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "bad PNG magic"
    with open("/tmp/h64.png", "wb") as f:
        f.write(png)
    print(f"wrote /tmp/h64.png ({len(png)} bytes)")

    # ---- decode round-trip checks ------------------------------------
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(37, 53, 3), dtype=np.uint8)
    assert np.array_equal(decode_png(write_png(img)), img), "RGB round-trip"

    # exercise all five filter types: forward-filter each scanline with a
    # chosen filter, hand-build the PNG, and check the decoder inverts it.
    def _forward_filter(rows: np.ndarray, ftype: int, bpp: int = 3) -> bytes:
        out = bytearray()
        prev = np.zeros(rows.shape[1], dtype=np.int16)
        for row in rows.astype(np.int16):
            a = np.zeros_like(row)
            a[bpp:] = row[:-bpp]
            if ftype == 0:
                filt = row
            elif ftype == 1:
                filt = row - a
            elif ftype == 2:
                filt = row - prev
            elif ftype == 3:
                filt = row - ((a + prev) >> 1)
            else:
                filt = np.empty_like(row)
                for i in range(row.size):
                    ai = int(a[i]) if i >= bpp else 0
                    bi = int(prev[i])
                    ci = int(prev[i - bpp]) if i >= bpp else 0
                    filt[i] = row[i] - _paeth(ai, bi, ci)
            out.append(ftype)
            out += (filt & 0xFF).astype(np.uint8).tobytes()
            prev = row
        return bytes(out)

    def _png_with_filter(arr: np.ndarray, ftype: int) -> bytes:
        h, w, _ = arr.shape
        ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        raw = _forward_filter(arr.reshape(h, w * 3), ftype)
        return (_PNG_MAGIC + _chunk(b"IHDR", ihdr)
                + _chunk(b"IDAT", zlib.compress(raw, 6))
                + _chunk(b"IEND", b""))

    for ftype in range(5):
        got = decode_png(_png_with_filter(img, ftype))
        assert np.array_equal(got, img), f"filter {ftype} round-trip"
    # mixed filters, one per row, with correct running previous-row state
    mixed = bytearray()
    prev = np.zeros(img.shape[1] * 3, dtype=np.int16)
    for y, row in enumerate(img.reshape(img.shape[0], -1).astype(np.int16)):
        ftype = y % 5
        a = np.zeros_like(row)
        a[3:] = row[:-3]
        if ftype == 0:
            filt = row
        elif ftype == 1:
            filt = row - a
        elif ftype == 2:
            filt = row - prev
        elif ftype == 3:
            filt = row - ((a + prev) >> 1)
        else:
            filt = np.array(
                [row[i] - _paeth(int(a[i]), int(prev[i]),
                                 int(prev[i - 3]) if i >= 3 else 0)
                 for i in range(row.size)], dtype=np.int16)
        mixed.append(ftype)
        mixed += (filt & 0xFF).astype(np.uint8).tobytes()
        prev = row
    h, w, _ = img.shape
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png_mix = (_PNG_MAGIC + _chunk(b"IHDR", ihdr)
               + _chunk(b"IDAT", zlib.compress(bytes(mixed), 6))
               + _chunk(b"IEND", b""))
    assert np.array_equal(decode_png(png_mix), img), "mixed-filter round-trip"
    # RGBA input: alpha must be dropped
    rgba = np.dstack([img, np.full(img.shape[:2], 200, np.uint8)])
    h, w, _ = rgba.shape
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))
    png6 = (_PNG_MAGIC + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(raw, 6)) + _chunk(b"IEND", b""))
    assert np.array_equal(decode_png(png6), img), "RGBA round-trip"
    print("decode_png: RGB round-trip + all 5 filters + RGBA OK")
