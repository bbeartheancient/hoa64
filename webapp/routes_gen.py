"""Generative-lab routes — Phase 5a: terrain, orbitals, noise fields.

Thin wrappers over `hoa64.terrain` (Hadamard-layered Perlin fBm) and
`hoa64.orbitals` (hydrogenic |ψ|² sampling).  All compute is CPU NumPy,
so handlers run it via `asyncio.to_thread`; JSON grids are block-mean
downsampled to ≤ 128² and previews ride as base64 `heatmap_png`s.

    POST /api/gen/terrain      {size, order, octaves, persistence,
                                lacunarity, hadamard_mix, seed?}
                                 → heightmap + pngs + layers_f32: the SIGNED
                                amplitude-weighted per-octave fBm contribs
                                (amp = persistence**i), downsampled to the
                                same ≤128² grid as heightmap — the client
                                recombines them for octave mute/solo.
    POST /api/gen/orbital      {n, l, m, samples, seed?}
    POST /api/gen/noise-field  {size, order, seed?}
"""

from __future__ import annotations

import asyncio
import base64

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import orbitals, terrain
from ..hadamard import hadamard_known
from ._png import heatmap_png

GRID_MAX = 128  # JSON grids are downsampled to at most this per side
POINTS_MAX = 20000  # orbital point cloud cap in the response

router = APIRouter(prefix="/api/gen")


def _grid_json(a: np.ndarray, max_side: int = GRID_MAX) -> list:
    """Block-mean downsample a 2D grid to ≤ max_side per side → nested lists."""
    a = np.asarray(a, dtype=np.float64)
    h, w = a.shape
    sy = max(1, -(-h // max_side))  # ceil
    sx = max(1, -(-w // max_side))
    if sy > 1 or sx > 1:
        hh, ww = (h // sy) * sy, (w // sx) * sx
        a = a[:hh, :ww].reshape(hh // sy, sy, ww // sx, sx).mean(axis=(1, 3))
    return a.tolist()


def _b64(png: bytes) -> str:
    return base64.b64encode(png).decode("ascii")


def _check_order(order: int) -> None:
    if not 4 <= order <= 1028 or hadamard_known(order) is None:
        raise HTTPException(
            status_code=400, detail=f"no known Hadamard matrix of order {order}"
        )


class TerrainReq(BaseModel):
    size: int = 256
    order: int = 64
    octaves: int = 6
    persistence: float = 0.5
    lacunarity: float = 2.0
    hadamard_mix: float = 0.5
    seed: int | None = None


@router.post("/terrain")
async def gen_terrain(req: TerrainReq) -> dict:
    if not 8 <= req.size <= 512:
        raise HTTPException(status_code=400, detail="size must be in 8..512")
    if not 1 <= req.octaves <= 8:
        raise HTTPException(status_code=400, detail="octaves must be in 1..8")
    if not 0.0 <= req.persistence <= 1.0:
        raise HTTPException(status_code=400, detail="persistence must be in [0, 1]")
    if not 1.0 <= req.lacunarity <= 4.0:
        raise HTTPException(status_code=400, detail="lacunarity must be in [1, 4]")
    if not 0.0 <= req.hadamard_mix <= 1.0:
        raise HTTPException(status_code=400, detail="hadamard_mix must be in [0, 1]")
    _check_order(req.order)

    def compute() -> dict:
        t = terrain.terrain(
            size=req.size,
            order=req.order,
            octaves=req.octaves,
            persistence=req.persistence,
            lacunarity=req.lacunarity,
            hadamard_mix=req.hadamard_mix,
            seed=req.seed,
        )
        h = t["heightmap"]
        return {
            "heightmap": _grid_json(h),
            "png_b64": _b64(heatmap_png(h, 512)),
            "layers": [_b64(heatmap_png(layer, 256)) for layer in t["layers"]],
            # signed amp-weighted octave contribs on the SAME ≤128² grid as
            # heightmap — client recombination for mute/solo
            "layers_f32": [_grid_json(c) for c in t["contribs"]],
            "stats": {
                "min": float(h.min()),
                "max": float(h.max()),
                "mean": float(h.mean()),
            },
        }

    try:
        return await asyncio.to_thread(compute)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


class OrbitalReq(BaseModel):
    n: int
    l: int
    m: int
    samples: int = 20000
    seed: int | None = None


@router.post("/orbital")
async def gen_orbital(req: OrbitalReq) -> dict:
    if not 100 <= req.samples <= 100000:
        raise HTTPException(status_code=400, detail="samples must be in 100..100000")
    try:
        orbitals._validate(req.n, req.l, req.m)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    def compute() -> dict:
        s = orbitals.sample_orbital(req.n, req.l, req.m, n_samples=req.samples,
                                    seed=req.seed)
        pts = s["points"][:POINTS_MAX]
        w = s["weights"][:POINTS_MAX]
        e = s["extent"]
        # xz-plane projection: 256² count histogram, log stretch for the
        # (very peaked) density, then the two-sided heatmap ramp
        counts, _, _ = np.histogram2d(
            pts[:, 0], pts[:, 2], bins=256, range=[[-e, e], [-e, e]]
        )
        proj = np.log1p(counts)
        return {
            "points": pts.tolist(),
            "weights": w.tolist(),
            "proj_png_b64": _b64(heatmap_png(proj, 256)),
            "extent": float(e),
        }

    return await asyncio.to_thread(compute)


class NoiseFieldReq(BaseModel):
    size: int = 256
    order: int = 64
    seed: int | None = None


@router.post("/noise-field")
async def gen_noise_field(req: NoiseFieldReq) -> dict:
    if not 8 <= req.size <= 512:
        raise HTTPException(status_code=400, detail="size must be in 8..512")
    _check_order(req.order)

    def compute() -> dict:
        rng = np.random.default_rng(req.seed)

        def had_fn(shp, res, r):
            cell = max(1, int(round(shp[0] / max(res, 1e-9))))
            return terrain.hadamard_noise(shp, req.order, r, cell=cell)

        base_res = max(2.0, req.size / req.order)  # ~one Hadamard period/cell
        field = terrain.fbm((req.size, req.size), 3, base_res, 0.5, 2.0, had_fn, rng)
        field = np.abs(field)
        lo, hi = float(field.min()), float(field.max())
        if hi > lo:
            field = (field - lo) / (hi - lo)
        return {"png_b64": _b64(heatmap_png(field, 512)), "grid": _grid_json(field)}

    try:
        return await asyncio.to_thread(compute)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
