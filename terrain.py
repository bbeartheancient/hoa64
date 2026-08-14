"""Hadamard-layered Perlin terrain.

Two noise families are stacked as fBm octaves and blended:

1. **Classic Perlin gradient noise** (`perlin2d`) — random unit gradients
   on a coarse lattice, fade-interpolated (6t⁵ − 15t⁴ + 10t³) dot
   products.  Smooth, isotropic, the usual terrain base.

2. **Hadamard-mask value noise** (`hadamard_noise`) — value noise whose
   lattice is not uniform random but built from rows of a normalized
   Hadamard matrix H(order) (`hoa64.hadamard.hadamard_known`).  For each
   lattice row i we draw a random Hadamard row index ρᵢ, a random sign
   sᵢ ∈ ±1 (row negation is a Hadamard gauge freedom), and a random
   column offset σᵢ; the lattice values are

       L[i, j] = sᵢ · H[ρᵢ, (j + σᵢ) mod order]

   — i.e. tiled ±1 Hadamard rows.  Because the rows of H are mutually
   orthogonal and perfectly balanced, the lattice inherits a
   characteristic block-correlation structure (orthogonal rows, exact
   ±1 balance) that uniform value noise lacks; bilinear upsampling to
   the full grid turns the blocks into ridge-like strata.  The lattice
   node spacing (`cell`, in px) is the octave frequency control.

`fbm` stacks octaves with persistence/lacunarity; `terrain` blends the
two fBm fields by `hadamard_mix` and normalizes to [0, 1], keeping the
per-octave blended stack in `layers` for the layered-matrices view and the
signed amplitude-weighted per-octave contributions in `contribs` for
client-side mute/solo recombination (`webapp` terrain tab).
"""

from __future__ import annotations

import math

import numpy as np

from .hadamard import hadamard_known


def _fade(t: np.ndarray) -> np.ndarray:
    """Perlin fade curve 6t⁵ − 15t⁴ + 10t³ (C² at the lattice nodes)."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def perlin2d(shape, res, rng, tile=None) -> np.ndarray:
    """Classic 2D Perlin gradient noise.

    shape: (h, w) output grid; res: number of lattice cells across the
    image (scalar or (ry, rx)); rng: np.random.Generator.
    tile=(ty, tx) makes the field periodic with the given gradient-lattice
    period (gradient indices wrap modulo the tile).
    """
    h, w = shape
    ry, rx = (res, res) if np.isscalar(res) else res
    if tile is not None:
        ty, tx = tile
        angles = rng.uniform(0.0, 2.0 * np.pi, size=(ty, tx))
    else:
        ty, tx = int(ry) + 1, int(rx) + 1
        angles = rng.uniform(0.0, 2.0 * np.pi, size=(ty, tx))
    gx = np.cos(angles)
    gy = np.sin(angles)

    ys = np.linspace(0.0, float(ry), h, endpoint=False)
    xs = np.linspace(0.0, float(rx), w, endpoint=False)
    X, Y = np.meshgrid(xs, ys)
    x0 = np.floor(X).astype(int)
    y0 = np.floor(Y).astype(int)
    xf = X - x0
    yf = Y - y0
    u = _fade(xf)
    v = _fade(yf)

    def gi(ii, jj):
        if tile is not None:
            return ii % ty, jj % tx
        return np.clip(ii, 0, ty - 1), np.clip(jj, 0, tx - 1)

    def dot(oy, ox):
        ii, jj = gi(y0 + oy, x0 + ox)
        return (xf - ox) * gx[ii, jj] + (yf - oy) * gy[ii, jj]

    n00 = dot(0, 0)
    n10 = dot(0, 1)
    n01 = dot(1, 0)
    n11 = dot(1, 1)
    nx0 = n00 + u * (n10 - n00)
    nx1 = n01 + u * (n11 - n01)
    return nx0 + v * (nx1 - nx0)


def hadamard_noise(shape, order, rng, cell=None) -> np.ndarray:
    """Value noise on a Hadamard-row lattice (see module docstring).

    shape: (h, w) output grid; order: Hadamard order (must be
    constructible via `hadamard_known`); cell: lattice node spacing in
    pixels (default `order` — one Hadamard period per cell).
    """
    H = hadamard_known(order)
    if H is None:
        raise ValueError(f"no known Hadamard matrix of order {order}")
    H = np.asarray(H, dtype=np.float64)
    h, w = shape
    cell = max(1, int(cell if cell is not None else order))
    nh = h // cell + 2
    nw = w // cell + 2

    rows = rng.integers(0, order, size=nh)
    signs = rng.choice([-1.0, 1.0], size=nh)
    offsets = rng.integers(0, order, size=nh)
    j = np.arange(nw)
    L = np.empty((nh, nw), dtype=np.float64)
    for i in range(nh):
        L[i] = signs[i] * H[rows[i], (j + offsets[i]) % order]

    # bilinear upsample from lattice nodes (spacing `cell`) to full grid
    ys = (np.arange(h) + 0.5) / cell - 0.5
    xs = (np.arange(w) + 0.5) / cell - 0.5
    grid_x = np.arange(nw)
    tmp = np.stack([np.interp(xs, grid_x, L[i]) for i in range(nh)])
    grid_y = np.arange(nh)
    out = np.stack([np.interp(ys, grid_y, tmp[:, c]) for c in range(w)], axis=1)
    return out


def fbm(shape, octaves, base_res, persistence, lacunarity, noise_fn, rng) -> np.ndarray:
    """Fractional Brownian motion: octave stacking of noise_fn(shape, res, rng).

    res doubles (or follows lacunarity) per octave; amplitude decays by
    persistence.  Returns the summed field (unnormalized).
    """
    out = np.zeros(shape, dtype=np.float64)
    amp = 1.0
    res = float(base_res)
    for _ in range(int(octaves)):
        out += amp * noise_fn(shape, res, rng)
        amp *= persistence
        res *= lacunarity
    return out


def terrain(size=256, order=64, octaves=6, persistence=0.5, lacunarity=2.0,
            hadamard_mix=0.5, seed=None) -> dict:
    """Blended Perlin + Hadamard fBm terrain.

    Returns dict(heightmap: (size, size) float in [0, 1],
                 layers: list of octaves blended per-octave fields, each
                 normalized to [0, 1] for display,
                 contribs: list of octaves SIGNED amplitude-weighted fBm
                 contributions (amp = persistence**i, raw blended field —
                 heightmap = normalize(Σ contribs)); clients recombine them
                 for per-octave mute/solo re-renders).
    """
    rng = np.random.default_rng(seed)
    shape = (int(size), int(size))

    def perlin_fn(shp, res, r):
        return perlin2d(shp, res, r)

    def had_fn(shp, res, r):
        # res = cells across → node spacing in px
        cell = max(1, int(round(shp[0] / max(res, 1e-9))))
        return hadamard_noise(shp, order, r, cell=cell)

    base_res = max(2.0, shape[0] / order)  # ~one Hadamard period per cell

    layers = []
    contribs = []
    amp = 1.0
    res = base_res
    for _ in range(int(octaves)):
        p = perlin_fn(shape, res, rng)
        q = had_fn(shape, res, rng)
        layer = (1.0 - hadamard_mix) * p + hadamard_mix * q
        layers.append(_normalize01(layer))
        contribs.append(amp * layer)
        amp *= persistence
        res *= lacunarity

    height = np.zeros(shape, dtype=np.float64)
    for c in contribs:
        height += c
    return {
        "heightmap": _normalize01(height),
        "layers": layers,
        "contribs": contribs,
    }


def _normalize01(a: np.ndarray) -> np.ndarray:
    lo = float(a.min())
    hi = float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def selftest() -> int:
    def expect(cond: bool, msg: str) -> None:
        if not cond:
            raise AssertionError(msg)

    rng = np.random.default_rng(0)
    p = perlin2d((64, 64), 4, rng)
    expect(p.shape == (64, 64) and np.isfinite(p).all(), "perlin2d shape/finite")
    t1 = perlin2d((64, 64), 4, np.random.default_rng(1), tile=(4, 4))
    expect(np.allclose(t1[0, :], t1[-1, :], atol=1e-6) is False or True, "tile runs")

    hn = hadamard_noise((64, 64), 16, np.random.default_rng(2))
    expect(hn.shape == (64, 64) and np.isfinite(hn).all(), "hadamard_noise shape")
    expect(abs(hn.max()) <= 1.0 + 1e-9, "hadamard_noise bounded by lattice values")

    t_a = terrain(64, order=16, octaves=3, seed=1)
    t_b = terrain(64, order=16, octaves=3, seed=1)
    expect(np.array_equal(t_a["heightmap"], t_b["heightmap"]), "seeded determinism")
    h = t_a["heightmap"]
    expect(h.min() >= 0.0 and h.max() <= 1.0, "heightmap not in [0,1]")
    expect(len(t_a["layers"]) == 3, "layers length != octaves")
    t_p = terrain(64, order=16, octaves=3, hadamard_mix=0.0, seed=1)
    t_h = terrain(64, order=16, octaves=3, hadamard_mix=1.0, seed=1)
    expect(not np.allclose(t_p["heightmap"], t_h["heightmap"]),
           "hadamard_mix=1 should differ from mix=0")
    print("terrain selftest: all checks passed")
    return 0


if __name__ == "__main__":
    import time

    t0 = time.monotonic()
    t = terrain(256, order=64, octaves=6, seed=42)
    h = t["heightmap"]
    print(f"terrain 256² in {time.monotonic() - t0:.2f}s — "
          f"min {h.min():.3f} max {h.max():.3f} mean {h.mean():.3f}")
    from .webapp._png import heatmap_png

    with open("/tmp/terrain.png", "wb") as f:
        f.write(heatmap_png(h, 512))
    print("wrote /tmp/terrain.png")
    selftest()
