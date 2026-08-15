"""Materials-lab API — cloth / touchpad / metamaterial from flux tiles.

Synchronous ``POST /design`` and ``POST /kicad``.  KiCad uses the same
one-shot token cache pattern as the antenna/filter exporters.
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from .. import kicad_gen, materials
from .routes_hadamard import _jsafe
from .routes_search import LIB_DIR

router = APIRouter(prefix="/api/materials")

_KICAD_CACHE: OrderedDict[str, dict[str, str]] = OrderedDict()
_KICAD_KEPT = 8


class DesignReq(BaseModel):
    kind: str = "cloth"
    order: int = 16
    start: str = "sylvester"
    pitch_mm: float = 1.0


def _build(req: DesignReq) -> dict:
    if req.kind not in materials.KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown kind {req.kind!r}; expected {materials.KINDS}",
        )
    if req.start not in materials.STARTS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown start {req.start!r}; expected {materials.STARTS}",
        )
    if not (4 <= req.order <= 256) or req.order % 4 != 0:
        raise HTTPException(status_code=400, detail="order must be 4k, 4..256")
    if not (0.2 <= req.pitch_mm <= 20.0):
        raise HTTPException(status_code=400, detail="pitch_mm must be 0.2..20")
    try:
        return materials.design(
            req.kind, req.order, start=req.start,
            pitch_mm=req.pitch_mm, lib_dir=LIB_DIR,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/design")
def design(req: DesignReq) -> dict:
    d = _build(req)
    # drop the raw lattice/caps if huge — already bounded in materials.py
    return _jsafe({
        "kind": req.kind,
        "order": d["order"],
        "start": d["start"],
        "stats": d["stats"],
        "tiles": d["tiles"],
        "preview": d["preview"],
        "key": d.get("key"),
    })


@router.post("/kicad")
def kicad(req: DesignReq) -> dict:
    d = _build(req)
    mhz = req.order  # used only as a name tag
    name = f"hoa64_{req.kind}_{req.order}"
    try:
        files = kicad_gen.kicad_files(
            "materials", float(mhz) * 1e6,
            layout=d["layout"],
            name=name,
            descr=f"hoa64 {req.kind} n={req.order} ({req.start})",
            comment=(f"{req.kind} from H.{req.order} flux tiles; "
                     f"pitch {req.pitch_mm} mm; hoa64.materials"),
            eps_r=4.4, h_m=1.6e-3,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    token = uuid.uuid4().hex[:8]
    _KICAD_CACHE[token] = dict(files)
    while len(_KICAD_CACHE) > _KICAD_KEPT:
        _KICAD_CACHE.popitem(last=False)
    return _jsafe({
        "token": token,
        "files": sorted(files),
        "preview": d["preview"],
        "stats": d["stats"],
        "tiles": d["tiles"],
        "key": d.get("key"),
    })


@router.get("/kicad/{token}/{name}")
def kicad_file(token: str, name: str) -> PlainTextResponse:
    entry = _KICAD_CACHE.get(token)
    content = entry.pop(name, None) if entry is not None else None
    if content is None:
        raise HTTPException(
            status_code=404, detail="unknown token or file already consumed",
        )
    if not entry:
        _KICAD_CACHE.pop(token, None)
    return PlainTextResponse(
        content,
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )
