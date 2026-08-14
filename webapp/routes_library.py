"""Library / construction-DAG routes — Phase 5b: the map of all orders.

`GET /api/dag` exposes `game_of_hadamard.classify_orders` (BUILT /
CLAIMED / GAP tiers, method glyphs, Kronecker depths) for the Library
tab's construction map.  Classification scans the whole toolchain
(`hadamard_known` per order), so results are cached per `max` for 60 s
and computed in a worker thread.

`GET /api/detbounds` pairs each built order with its achieved
log₁₀ det vs the Hadamard bound ½n log₁₀ n (slogdet only for n ≤ 256 —
cap chosen so the worst case is a 256² LU; larger orders ride along
with det_log10 null).  Cached 5 min.
"""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, HTTPException

from ..game_of_hadamard import classify_orders
from ..hadamard import det_bound_log10, det_log10, hadamard_known
from .routes_hadamard import _jsafe

DET_COMPUTE_MAX = 256  # slogdet only up to this order
DAG_TTL = 60.0
DET_TTL = 300.0

router = APIRouter(prefix="/api")

_dag_cache: dict[int, tuple[float, dict]] = {}
_det_cache: dict = {"t": 0.0, "max": None, "data": None}


@router.get("/dag")
async def dag(max: int = 400) -> dict:
    if not 4 <= max <= 4000:
        raise HTTPException(status_code=400, detail="max must be in 4..4000")
    now = time.time()
    ent = _dag_cache.get(max)
    if ent and now - ent[0] < DAG_TTL:
        return ent[1]
    data = await asyncio.to_thread(classify_orders, max)
    data = _jsafe({"max": max, **data})
    _dag_cache[max] = (now, data)
    return data


@router.get("/detbounds")
async def detbounds(max: int = 400) -> dict:
    if not 4 <= max <= 4000:
        raise HTTPException(status_code=400, detail="max must be in 4..4000")
    now = time.time()
    if (
        _det_cache["data"] is not None
        and _det_cache["max"] == max
        and now - _det_cache["t"] < DET_TTL
    ):
        return _det_cache["data"]

    def compute() -> dict:
        built = classify_orders(max)["built"]
        entries = []
        for n in built:
            e = {
                "order": n,
                "det_log10": None,
                "det_bound_log10": det_bound_log10(n),
            }
            if n <= DET_COMPUTE_MAX:
                H = hadamard_known(n)
                if H is not None:
                    e["det_log10"] = det_log10(H)
            entries.append(e)
        return {"max": max, "entries": entries}

    data = _jsafe(await asyncio.to_thread(compute))
    _det_cache.update(t=now, max=max, data=data)
    return data
