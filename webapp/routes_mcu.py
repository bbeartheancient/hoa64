"""Microcontroller-lab API — LED matrix, WiFi mesh field, edge exports.

Wraps `hoa64.mcu`:

* ``POST /api/mcu/firmware``     — generate LED (esp32/teensy/
  circuitpython) or ESP-NOW mesh firmware into the one-shot token cache
  (same pattern as the materials KiCad exporter; individual files +
  README.md, plus a ``.zip`` of the whole set via stdlib zipfile).
* ``GET  /api/mcu/file/{token}/{name}`` — pop one generated file.
* ``POST /api/mcu/push``         — pack a W×H frame (`mcu.pack_frame`)
  and POST the raw GRB bytes to ``http://{host}/frame``.  ``host`` must
  be a bare IPv4/hostname[:port] — no scheme, no path — this is a
  trusted-local lab tool like `server.py`, not an open proxy.
* ``POST /api/mcu/mesh/collect`` — GET ``http://{host}/mesh`` from a
  mesh gateway, cache the JSON in `_LAST_MESH` and return it.
* ``POST /api/mcu/export``       — edge-engine templates
  (`mcu.export_engine`) into the one-shot cache.
"""

from __future__ import annotations

import io
import re
import uuid
import zipfile
from collections import OrderedDict

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from .. import mcu
from .routes_hadamard import _jsafe

router = APIRouter(prefix="/api/mcu")

_CACHE: OrderedDict[str, dict[str, str]] = OrderedDict()
_KEPT = 8

_LAST_MESH: dict | None = None

_HOST_RE = re.compile(
    r"^([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?"
    r"(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*)(:[0-9]{1,5})?$"
)


def _check_host(host: str) -> str:
    """Bare IPv4/hostname[:port] only — no scheme, path, or userinfo."""
    host = host.strip()
    if not _HOST_RE.match(host) or len(host) > 253:
        raise HTTPException(
            status_code=400,
            detail="host must be a bare IPv4/hostname[:port], e.g. 192.168.4.1",
        )
    return host


def _stash(files: dict[str, str], zipname: str = "firmware.zip") -> dict:
    """Store files in the bounded one-shot cache and add a .zip."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():
            z.writestr(name, content)
    out = dict(files)
    out[zipname] = buf.getvalue().decode("latin-1")
    token = uuid.uuid4().hex[:8]
    _CACHE[token] = out
    while len(_CACHE) > _KEPT:
        _CACHE.popitem(last=False)
    return {"token": token, "files": sorted(files) + [zipname]}


@router.get("/file/{token}/{name}")
def get_file(token: str, name: str) -> PlainTextResponse:
    entry = _CACHE.get(token)
    content = entry.pop(name, None) if entry is not None else None
    if content is None:
        raise HTTPException(
            status_code=404, detail="unknown token or file already consumed",
        )
    if not entry:
        _CACHE.pop(token, None)
    if name.endswith(".zip"):
        return PlainTextResponse(
            content.encode("latin-1"),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )
    return PlainTextResponse(
        content,
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


# ---------------------------------------------------------------- firmware

class FirmwareReq(BaseModel):
    kind: str = "led"                # led | mesh
    # led
    board: str = "esp32"
    w: int = 16
    h: int = 16
    pin: int | None = None
    serpentine: bool = True
    ssid: str | None = None
    password: str | None = None
    brightness: int = 64
    # mesh
    n_nodes: int = 4
    gateway_id: int = 0


@router.post("/firmware")
def firmware(req: FirmwareReq) -> dict:
    try:
        if req.kind == "led":
            cfg = {
                "board": req.board, "w": req.w, "h": req.h,
                "serpentine": req.serpentine, "brightness": req.brightness,
                "ssid": req.ssid, "password": req.password,
            }
            if req.pin is not None:
                cfg["pin"] = req.pin
            files = mcu.led_firmware(cfg)
            notes = ("flash the sketch/code.py, then POST frames to "
                     "http://<board>/frame — README.md has the wiring")
        elif req.kind == "mesh":
            files = mcu.mesh_firmware({
                "n_nodes": req.n_nodes, "gateway_id": req.gateway_id,
            })
            notes = ("flash one copy per node with a unique NODE_ID; the "
                     "gateway serves GET /mesh on 192.168.4.1 (ALPHA)")
        else:
            raise HTTPException(
                status_code=400, detail=f"unknown kind {req.kind!r}",
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return _jsafe({**_stash(files), "notes": notes})


# ---------------------------------------------------------------- push

class PushReq(BaseModel):
    host: str
    w: int = 16
    h: int = 16
    serpentine: bool = True
    pixels: list[list[int]]


@router.post("/push")
def push(req: PushReq) -> dict:
    host = _check_host(req.host)
    if not (1 <= req.w <= 64 and 1 <= req.h <= 64):
        raise HTTPException(status_code=400, detail="w/h must be 1..64")
    try:
        body = mcu.pack_frame(req.pixels, req.w, req.h, req.serpentine)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    import httpx  # lazy
    try:
        r = httpx.post(
            f"http://{host}/frame",
            content=body,
            headers={"content-type": "application/octet-stream"},
            timeout=5.0,
        )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"push failed: {e}") from e
    return _jsafe({"ok": r.status_code == 200, "status": r.status_code,
                   "bytes": len(body)})


# ---------------------------------------------------------------- mesh collect

class MeshReq(BaseModel):
    host: str


@router.post("/mesh/collect")
def mesh_collect(req: MeshReq) -> dict:
    global _LAST_MESH
    host = _check_host(req.host)
    import httpx  # lazy
    try:
        r = httpx.get(f"http://{host}/mesh", timeout=5.0)
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"collect failed: {e}") from e
    except ValueError as e:
        raise HTTPException(status_code=502, detail=f"bad mesh JSON: {e}") from e
    _LAST_MESH = data
    return _jsafe(data)


# ---------------------------------------------------------------- edge export

class ExportReq(BaseModel):
    engine: str
    target: str


@router.post("/export")
def export(req: ExportReq) -> dict:
    try:
        files = mcu.export_engine(req.engine, req.target)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return _jsafe(_stash(files, zipname=f"{req.engine}_{req.target}.zip"))
