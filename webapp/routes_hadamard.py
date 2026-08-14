"""Hadamard API routes — construction, verification, and library access.

Thin JSON wrappers around the `hoa64.hadamard` toolchain.  All endpoints
are synchronous: the classical constructions (Sylvester, Paley, Miyamoto,
Cooper-Wallis, GCP, row-builder) and the Gram-matrix `check` are fast for
the ≤ 4000 orders served here (determinants only below n ≤ 500, where the
LU/slogdet cost is trivial).  Long-running search work is Phase 2 and will
go through `jobs.py`, not this router.

Every payload passes through `_jsafe` — `check`/`modular_check` return
NumPy scalars and FastAPI's jsonable_encoder chokes on np.integer /
np.bool_ when they hide inside plain dicts.

The library is the flat ~/open_hadamard/hadamard_{n}.csv collection the
search daemons maintain (comma-separated ±1, `np.int8`).
"""

from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from ..hadamard import (
    check,
    det_bound_log10,
    h2_stats,
    hadamard_known,
    modular_check,
    paley,
    sylvester,
)
from . import __version__
from ._png import matrix_png

LIB_DIR = Path.home() / "open_hadamard"
MAX_ORDER = 4000
DET_MAX = 500  # compute det_log10 only up to this order

router = APIRouter(prefix="/api")


def _jsafe(x: Any) -> Any:
    """Recursively convert NumPy values to plain Python for JSON."""
    if isinstance(x, dict):
        return {k: _jsafe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsafe(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsafe(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def _library_path(order: int) -> Path | None:
    p = LIB_DIR / f"hadamard_{order}.csv"
    return p if p.is_file() else None


def _load_library(order: int) -> np.ndarray | None:
    p = _library_path(order)
    if p is None:
        return None
    return np.loadtxt(p, delimiter=",", dtype=np.int8)


def _stats(H: np.ndarray) -> dict:
    n = int(H.shape[0])
    stats = check(H, det=n <= DET_MAX)
    stats["det_bound"] = det_bound_log10(n)
    return stats


def _valid_order(order: int) -> bool:
    return 1 <= order <= MAX_ORDER and (order in (1, 2) or order % 4 == 0)


# ---------------------------------------------------------------- health

@router.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


# ---------------------------------------------------------------- orders

_orders_cache: dict[str, Any] = {"t": 0.0, "data": []}


def _known_orders() -> list[int]:
    """Order ints parsed from the library filenames, cached for 30 s."""
    now = time.time()
    if _orders_cache["data"] and now - _orders_cache["t"] < 30.0:
        return _orders_cache["data"]
    orders = []
    for p in LIB_DIR.glob("hadamard_*.csv"):
        try:
            orders.append(int(p.stem.split("_", 1)[1]))
        except (ValueError, IndexError):
            continue
    orders.sort()
    _orders_cache.update(t=now, data=orders)
    return orders


@router.get("/orders")
def orders(max: int = 400) -> dict:
    known = [n for n in _known_orders() if n <= max]
    known_set = set(known)
    candidates = [1, 2] + list(range(4, max + 1, 4))
    gaps = [n for n in candidates if n not in known_set]
    return {"known": known, "gaps": gaps, "max": max}


# ---------------------------------------------------------------- construct

class ConstructReq(BaseModel):
    order: int
    method: str = "auto"
    seed: int | None = None


def _construct(order: int, method: str, seed: int | None) -> np.ndarray | None:
    if method == "auto":
        return hadamard_known(order)
    if method == "sylvester":
        return sylvester(order)
    if method == "paley":
        return paley(order)
    if method == "miyamoto":
        from ..miyamoto import miyamoto_from_cache

        return miyamoto_from_cache(order)
    if method == "cw":
        from ..cw_construction import cw_build

        return cw_build(order)
    if method == "gcp":
        from ..gcp_hadamard import construct as gcp_construct

        return gcp_construct(order, np.random.default_rng(seed))
    if method == "row_builder":
        from ..row_builder import try_construct

        return try_construct(order)
    raise ValueError(f"unknown method {method!r}")


@router.post("/construct")
def construct(req: ConstructReq) -> dict:
    if not _valid_order(req.order):
        return _jsafe(
            {"ok": False, "error": f"order must be 1, 2 or a multiple of 4, ≤ {MAX_ORDER}"}
        )
    try:
        H = _construct(req.order, req.method, req.seed)
    except ValueError as e:
        return _jsafe({"ok": False, "error": str(e)})
    if H is None:
        return _jsafe(
            {"ok": False, "error": f"method {req.method!r} cannot build order {req.order}"}
        )
    H = np.asarray(H, dtype=np.int8)
    png_b64 = base64.b64encode(matrix_png(H, 512)).decode("ascii")
    return _jsafe(
        {
            "ok": True,
            "order": int(H.shape[0]),
            "method": req.method,
            "stats": _stats(H),
            "png_b64": png_b64,
        }
    )


# ---------------------------------------------------------------- verify

class VerifyReq(BaseModel):
    matrix: list[list[int]] | None = None
    order: int | None = None
    mod: int | None = None


@router.post("/verify")
def verify_matrix(req: VerifyReq) -> dict:
    if (req.matrix is None) == (req.order is None):
        return _jsafe({"ok": False, "error": "supply exactly one of 'matrix' or 'order'"})
    if req.order is not None:
        H = _load_library(req.order)
        if H is None:
            return _jsafe({"ok": False, "error": f"order {req.order} not in library"})
    else:
        H = np.asarray(req.matrix, dtype=np.int8)
        if H.ndim != 2 or H.shape[0] != H.shape[1] or H.shape[0] < 1:
            return _jsafe({"ok": False, "error": "matrix must be a non-empty square 2-D array"})
    h2 = h2_stats(H)
    out = {
        "ok": True,
        "order": int(H.shape[0]),
        "stats": _stats(H),
        "h2": {
            "pairs": len(h2),
            "balanced": sum(1 for r in h2 if r["balanced"]),
            "all_balanced": bool(all(r["balanced"] for r in h2)),
        },
    }
    if req.mod:
        out["modular"] = modular_check(H, req.mod)
    return _jsafe(out)


# ---------------------------------------------------------------- library

@router.get("/library/{order}")
def library(order: int) -> dict:
    H = _load_library(order)
    if H is None:
        raise HTTPException(status_code=404, detail=f"order {order} not in library")
    png_b64 = base64.b64encode(matrix_png(H, 512)).decode("ascii")
    return _jsafe(
        {"ok": True, "order": order, "stats": _stats(H), "png_b64": png_b64}
    )


@router.get("/library/{order}/csv")
def library_csv(order: int) -> FileResponse:
    p = _library_path(order)
    if p is None:
        raise HTTPException(status_code=404, detail=f"order {order} not in library")
    return FileResponse(p, media_type="text/csv", filename=p.name)


# ---------------------------------------------------------------- image

@router.get("/matrix/image")
def matrix_image(order: int, size: int = 512) -> Response:
    size = max(16, min(2048, size))
    H = _load_library(order)
    if H is None:
        H = hadamard_known(order)
    if H is None:
        raise HTTPException(status_code=404, detail=f"no Hadamard matrix of order {order}")
    return Response(content=matrix_png(H, size), media_type="image/png")


# ---------------------------------------------------------------- hadamard space

LATTICE_MAX = 256  # lattice grids bigger than this are too heavy for JSON


class VizSpaceReq(BaseModel):
    order: int
    mode: str = "rows"
    kappa: float = 1.0
    geodesics: bool = True
    max_points: int = 256


@router.post("/viz/hadamard-space")
async def viz_hadamard_space(req: VizSpaceReq) -> dict:
    """Transmute a Hadamard matrix into ℍ³ display data (`hadamard_space`)."""
    if req.mode not in ("rows", "lattice"):
        raise HTTPException(status_code=400, detail="mode must be 'rows' or 'lattice'")
    if not 0.25 <= req.kappa <= 4.0:
        raise HTTPException(status_code=400, detail="kappa must be in 0.25..4.0")
    if not 8 <= req.max_points <= 512:
        raise HTTPException(status_code=400, detail="max_points must be in 8..512")
    if not _valid_order(req.order):
        raise HTTPException(
            status_code=400, detail="order must be 1, 2 or a multiple of 4"
        )
    if hadamard_known(req.order) is None:
        raise HTTPException(
            status_code=400, detail=f"no known Hadamard matrix of order {req.order}"
        )
    if req.mode == "lattice" and req.order > LATTICE_MAX:
        raise HTTPException(
            status_code=400, detail=f"lattice mode capped at order {LATTICE_MAX}"
        )

    def compute() -> dict:
        from ..hadamard_space import transmute

        d = transmute(
            req.order,
            mode=req.mode,
            kappa=req.kappa,
            geodesics=req.geodesics,
            max_points=req.max_points,
        )
        # round coordinates to 4 decimals to keep the JSON light
        out = {
            "mode": d["mode"],
            "order": d["order"],
            "colors": d["colors"],
            "stats": d["stats"],
            "geodesics": [np.round(g, 4) for g in d["geodesics"]],
        }
        if d["mode"] == "rows":
            out["points"] = np.round(d["points"], 4)
        else:
            out["verts"] = np.round(d["verts"], 4)
        return out

    return _jsafe(await asyncio.to_thread(compute))
