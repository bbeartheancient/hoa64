"""HOA API routes — Phase 4 ambisonic studio (speakers / scene / decode).

Thin JSON wrappers around the HOA-7 calculator side of the package
(`basis`, `encode`, `decode`, `rotate`, `analysis`, `synth`, `audio_io`,
`report`).  Everything here is an instant computation (encode + Wigner-D
rotation + sphere-grid power maps for ≤ 5 s of audio at order ≤ 7), so no
JobManager involvement — synchronous endpoints in the `routes_hadamard`
style.  All payloads pass through `_jsafe`.

Power maps are returned transposed to (n_el × n_azi) in JSON — rows of
constant elevation read naturally top-to-bottom; the PNG renders the same
orientation.  `directional_power` itself returns power[azi, el].

WAV export: `scene(wav=true)` writes the rendered (C, T) Ambix bed to a
temp file and returns a one-shot `wav_token`; `GET /wav/{token}` serves
and deletes it (cache bounded to 8 entries).
"""

from __future__ import annotations

import base64
import math
import tempfile
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from ..analysis import (
    directional_power,
    doa_from_intensity,
    field_energy,
    peak_direction,
)
from ..audio_io import write_wav
from ..basis import MAX_ORDER, az_el_from_unit, sh_sn3d, sphere_grid
from ..decode import decode_directions, decode_grid
from ..encode import encode_plane_waves
from ..rotate import rotate_yaw_pitch_roll
from ..synth import noise, tone
from ._png import heatmap_png
from .routes_hadamard import _jsafe

MAX_SPEAKERS = 128
MAX_SOURCES = 16
MAX_DURATION = 5.0
MAX_GRID = (180, 90)  # (n_azi, n_el) caps

router = APIRouter(prefix="/api/hoa")

_WAV_CACHE: OrderedDict[str, Path] = OrderedDict()
_WAV_KEPT = 8


# ---------------------------------------------------------------- presets

_PHI = (1.0 + math.sqrt(5.0)) / 2.0


def _verts_to_positions(verts: list[tuple[float, float, float]]) -> list[dict]:
    V = np.asarray(verts, dtype=np.float64)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    az, el = az_el_from_unit(V)
    return [{"az": float(a), "el": float(e)} for a, e in zip(az, el)]


def _preset_positions(name: str) -> list[dict] | None:
    if name == "ring4":
        return [{"az": float(a), "el": 0.0} for a in (0, 90, 180, 270)]
    if name == "ring8":
        return [{"az": float(a), "el": 0.0} for a in range(0, 360, 45)]
    if name == "dome8":
        top = [{"az": float(a), "el": 45.0} for a in (45, 135, 225, 315)]
        low = [{"az": float(a), "el": 0.0} for a in (0, 90, 180, 270)]
        return top + low
    if name == "icosa":
        p = _PHI
        verts = (
            [(0, s1, s2 * p) for s1 in (1, -1) for s2 in (1, -1)]
            + [(s1, s2 * p, 0) for s1 in (1, -1) for s2 in (1, -1)]
            + [(s2 * p, 0, s1) for s1 in (1, -1) for s2 in (1, -1)]
        )
        return _verts_to_positions(verts)
    if name == "dodeca":
        p = _PHI
        q = 1.0 / p
        verts = (
            [(s1, s2, s3) for s1 in (1, -1) for s2 in (1, -1) for s3 in (1, -1)]
            + [(0, s1 * q, s2 * p) for s1 in (1, -1) for s2 in (1, -1)]
            + [(s1 * q, s2 * p, 0) for s1 in (1, -1) for s2 in (1, -1)]
            + [(s2 * p, 0, s1 * q) for s1 in (1, -1) for s2 in (1, -1)]
        )
        return _verts_to_positions(verts)
    if name == "grid":
        azi, el, _ = sphere_grid(12, 6, degrees=True)
        return [{"az": float(a), "el": float(e)} for a in azi for e in el]
    return None


def _decode_matrix(positions: list[dict], order: int) -> np.ndarray:
    az = np.array([p["az"] for p in positions], dtype=np.float64)
    el = np.array([p["el"] for p in positions], dtype=np.float64)
    return sh_sn3d(az, el, degrees=True, max_order=order)  # (S, C)


# ---------------------------------------------------------------- speakers

class SpeakerPos(BaseModel):
    az: float
    el: float


class SpeakersReq(BaseModel):
    positions: list[SpeakerPos] | None = None
    preset: str | None = None
    order: int = MAX_ORDER


@router.post("/speakers")
def speakers(req: SpeakersReq) -> dict:
    if not (1 <= req.order <= MAX_ORDER):
        raise HTTPException(status_code=400, detail=f"order must be 1..{MAX_ORDER}")
    if (req.positions is None) == (req.preset is None):
        raise HTTPException(status_code=400, detail="supply exactly one of 'positions' or 'preset'")
    if req.preset is not None:
        positions = _preset_positions(req.preset)
        if positions is None:
            raise HTTPException(status_code=400, detail=f"unknown preset {req.preset!r}")
    else:
        positions = [{"az": p.az, "el": p.el} for p in req.positions]
    if not (1 <= len(positions) <= MAX_SPEAKERS):
        raise HTTPException(status_code=400, detail=f"speakers capped at {MAX_SPEAKERS}")
    Y = _decode_matrix(positions, req.order)
    return _jsafe(
        {
            "positions": positions,
            "decode_matrix": Y.tolist(),
            "cond": float(np.linalg.cond(Y)),
            "n_channels": int(Y.shape[1]),
        }
    )


# ---------------------------------------------------------------- scene

class SourceReq(BaseModel):
    az: float
    el: float
    freq: float = 440.0
    gain: float = 1.0
    kind: str = "tone"  # tone|noise


class SceneReq(BaseModel):
    sources: list[SourceReq]
    rotate: dict[str, float] | None = None
    order: int = 3
    sr: int = 48000
    duration: float = 0.5
    grid: dict[str, int] = {"n_azi": 72, "n_el": 36}
    wav: bool = False


def _source_signal(src: SourceReq, duration: float, sr: int) -> np.ndarray:
    if src.kind == "tone":
        return tone(src.freq, duration, sr, amplitude=src.gain)
    if src.kind == "noise":
        return noise(duration, sr, amplitude=src.gain)
    raise HTTPException(status_code=400, detail=f"unknown source kind {src.kind!r}")


@router.post("/scene")
def scene(req: SceneReq) -> dict:
    if not (1 <= req.order <= MAX_ORDER):
        raise HTTPException(status_code=400, detail=f"order must be 1..{MAX_ORDER}")
    if not (1 <= len(req.sources) <= MAX_SOURCES):
        raise HTTPException(status_code=400, detail=f"sources capped at {MAX_SOURCES}")
    if not (0.0 < req.duration <= MAX_DURATION):
        raise HTTPException(status_code=400, detail=f"duration capped at {MAX_DURATION}s")
    if not (8000 <= req.sr <= 96000):
        raise HTTPException(status_code=400, detail="sr must be 8000..96000")
    n_azi = int(req.grid.get("n_azi", 72))
    n_el = int(req.grid.get("n_el", 36))
    if not (4 <= n_azi <= MAX_GRID[0] and 2 <= n_el <= MAX_GRID[1]):
        raise HTTPException(status_code=400, detail=f"grid capped at {MAX_GRID}")

    az = np.array([s.az for s in req.sources], dtype=np.float64)
    el = np.array([s.el for s in req.sources], dtype=np.float64)
    signals = np.stack([_source_signal(s, req.duration, req.sr) for s in req.sources])
    hoa = encode_plane_waves(az, el, signals, degrees=True, max_order=req.order)

    if req.rotate:
        hoa = rotate_yaw_pitch_roll(
            hoa,
            float(req.rotate.get("yaw", 0.0)),
            float(req.rotate.get("pitch", 0.0)),
            float(req.rotate.get("roll", 0.0)),
            degrees=True,
            max_order=req.order,
        )

    # representative frame: the loudest single time slice
    t_star = int(np.argmax(field_energy(hoa)))
    a = hoa[:, t_star]

    p_azi, p_el, power = directional_power(a, n_azi=n_azi, n_el=n_el,
                                           max_order=req.order)
    doa_az, doa_el = doa_from_intensity(a)
    pk_az, pk_el, pk_val = peak_direction(a, max_order=req.order)

    from ..report import report_from_hoa

    rep = report_from_hoa(
        hoa, req.sr, max_order=req.order,
        include_frames=False, include_bands=False, include_peak_map=False,
    )

    out: dict[str, Any] = {
        "power_map": {
            "azi": p_azi.tolist(),
            "el": p_el.tolist(),
            "power": power.T.tolist(),  # (n_el × n_azi) — rows of constant el
        },
        "power_png_b64": base64.b64encode(heatmap_png(power.T, 512)).decode("ascii"),
        "doa": {"az": float(doa_az), "el": float(doa_el)},
        "peak": {"az": pk_az, "el": pk_el, "val": pk_val},
        "energy": float(np.mean(field_energy(hoa))),
        "report": rep.to_dict(),
    }

    if req.wav:
        token = uuid.uuid4().hex
        path = Path(tempfile.mkdtemp(prefix="hoa64_wav_")) / f"hoa_scene_{token}.wav"
        write_wav(path, hoa, req.sr)
        _WAV_CACHE[token] = path
        while len(_WAV_CACHE) > _WAV_KEPT:
            _, old = _WAV_CACHE.popitem(last=False)
            old.unlink(missing_ok=True)
        out["wav_token"] = token

    return _jsafe(out)


@router.get("/wav/{token}")
def scene_wav(token: str) -> FileResponse:
    path = _WAV_CACHE.pop(token, None)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="unknown or consumed wav token")
    return FileResponse(
        path,
        media_type="audio/wav",
        filename=path.name,
        background=BackgroundTask(lambda: path.unlink(missing_ok=True)),
    )


# ---------------------------------------------------------------- decode-grid

class DecodeGridReq(BaseModel):
    hoa: list[float]
    speakers: list[SpeakerPos] | None = None
    n_azi: int | None = None
    n_el: int | None = None


@router.post("/decode-grid")
def decode_grid_ep(req: DecodeGridReq) -> dict:
    n_ch = len(req.hoa)
    order = int(math.isqrt(n_ch)) - 1
    if n_ch < 1 or (order + 1) ** 2 != n_ch or not (0 <= order <= MAX_ORDER):
        raise HTTPException(
            status_code=400, detail="hoa length must be (order+1)², order 0..7"
        )
    a = np.asarray(req.hoa, dtype=np.float64)
    if (req.speakers is None) == (req.n_azi is None and req.n_el is None):
        raise HTTPException(
            status_code=400, detail="supply exactly one of 'speakers' or grid n_azi/n_el"
        )
    if req.speakers is not None:
        if not (1 <= len(req.speakers) <= MAX_SPEAKERS):
            raise HTTPException(status_code=400, detail=f"speakers capped at {MAX_SPEAKERS}")
        az = np.array([s.az for s in req.speakers], dtype=np.float64)
        el = np.array([s.el for s in req.speakers], dtype=np.float64)
        samp = decode_directions(a, az, el, degrees=True, max_order=order)
        return _jsafe({"samples": samp.tolist(), "n_speakers": len(req.speakers)})
    n_azi = int(req.n_azi or 72)
    n_el = int(req.n_el or 36)
    if not (4 <= n_azi <= MAX_GRID[0] and 2 <= n_el <= MAX_GRID[1]):
        raise HTTPException(status_code=400, detail=f"grid capped at {MAX_GRID}")
    azi, el, samp = decode_grid(a, n_azi=n_azi, n_el=n_el, max_order=order)
    return _jsafe(
        {
            "azi": azi.tolist(),
            "el": el.tolist(),
            "samples": samp.tolist(),  # (n_azi × n_el), as decode_grid returns
            "png_b64": base64.b64encode(heatmap_png(samp, 512)).decode("ascii"),
        }
    )
