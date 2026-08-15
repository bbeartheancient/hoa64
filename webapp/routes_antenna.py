"""Antenna-lab API routes — design recommender, parts matcher, KiCad
export, FDTD field lab, and Hadamard-seeded topology evolution.

Wraps the antenna half of the package (`em_physics`, `antenna_design`,
`parts_db`, `kicad_gen`, `fdtd`, `antenna_evo`):

* ``POST /design`` / ``POST /parts`` / ``POST /kicad`` are instant
  computations — synchronous endpoints in the `routes_hoa` style, all
  payloads through `_jsafe`.  KiCad export uses a bounded in-memory
  token cache (one-shot per file) mirroring the `_WAV_CACHE` pattern.
* ``POST /fields`` runs `fdtd.fdtd_run` as a JobManager job: the solver
  callback becomes `report(job, ...)` frames whose |E| mid-plane slices
  (and the Stokes axial ratio when `pol_viz`) are heatmap PNGs, with
  `_BudgetStop` as the stop_flag and `params["live"]` as live_params
  (mid-run `src_amplitude` retune).
* ``POST /evolve`` runs `antenna_evo.antenna_sa` the same way; the
  callback pops the `geom` conduit key (points/z_in/gain of the
  current-best walk) into JSON-safe progress frames, and the final frame
  carries a heatmap of the best design's far-field `pattern` sampled on
  a θ×φ grid.  `live_params` retunes w_z/w_gain/w_size/cooling mid-run.

Progress streams over the existing ``WS /ws/job/{job_id}``.
"""

from __future__ import annotations

import base64
import math
import uuid
from collections import OrderedDict
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from .. import antenna_design, antenna_evo, fdtd, kicad_gen, parts_db, site_survey
from ..em_physics import MEDIA, build_dipole as em_build_dipole
from ._png import heatmap_png, write_png
from .jobs import JOBS, Job, report
from .routes_hadamard import _jsafe
from .routes_search import _BudgetStop

F_MIN_MHZ = 1.0
F_MAX_MHZ = 40000.0
_MEDIA_KEYS = tuple(MEDIA)
_KICAD_TYPES = ("patch", "meander_ifa", "loop")

router = APIRouter(prefix="/api/antenna")

_KICAD_CACHE: OrderedDict[str, dict[str, str]] = OrderedDict()
_KICAD_KEPT = 8


def _check_band(f_lo_mhz: float, f_hi_mhz: float) -> None:
    if not (F_MIN_MHZ <= f_lo_mhz <= F_MAX_MHZ
            and F_MIN_MHZ <= f_hi_mhz <= F_MAX_MHZ):
        raise HTTPException(
            status_code=400,
            detail=f"frequencies must be {F_MIN_MHZ:g}..{F_MAX_MHZ:g} MHz",
        )
    if not f_lo_mhz < f_hi_mhz:
        raise HTTPException(status_code=400, detail="f_lo_mhz must be < f_hi_mhz")


def _check_medium(medium: str) -> None:
    if medium not in MEDIA:
        raise HTTPException(
            status_code=400,
            detail=f"unknown medium {medium!r}; expected one of {_MEDIA_KEYS}",
        )


# ---------------------------------------------------------------- design

class DesignReq(BaseModel):
    f_lo_mhz: float
    f_hi_mhz: float
    medium: str = "air"
    site: dict[str, Any] | None = None


@router.post("/design")
def design(req: DesignReq) -> dict:
    _check_band(req.f_lo_mhz, req.f_hi_mhz)
    _check_medium(req.medium)
    site = None
    if req.site is not None:
        known = set(antenna_design.SiteConditions.__dataclass_fields__)
        extra = set(req.site) - known
        if extra:
            raise HTTPException(
                status_code=400,
                detail=f"unknown SiteConditions fields: {sorted(extra)}",
            )
        try:
            site = antenna_design.SiteConditions(**req.site)
        except TypeError as e:
            raise HTTPException(status_code=400, detail=f"bad site: {e}")
    entries = antenna_design.recommend(
        req.f_lo_mhz * 1e6, req.f_hi_mhz * 1e6,
        medium=req.medium, site=site,
    )
    out_entries = []
    for e in entries:
        e = dict(e)
        d = dict(e["design"])
        d.pop("pattern", None)  # non-JSON-safe callable
        e["design"] = d
        e["explain"] = antenna_design.explain(e)
        out_entries.append(e)
    f_c = 0.5 * (req.f_lo_mhz + req.f_hi_mhz)
    return _jsafe({
        "f_center_mhz": f_c,
        "required_bw_frac": (req.f_hi_mhz - req.f_lo_mhz) / f_c,
        "entries": out_entries,
    })


# ---------------------------------------------------------------- parts

class PartsReq(BaseModel):
    f_lo_mhz: float
    f_hi_mhz: float
    gain_dbi_min: float | None = None
    polarization: str | None = None
    max_size_mm: float | None = None
    mount: str | None = None
    type: str | None = None
    partial: bool | None = None  # None = full cover, then overlap fallback
    limit: int = 50  # return every in-range row; catalog is tens of parts


@router.post("/parts")
def parts(req: PartsReq) -> dict:
    _check_band(req.f_lo_mhz, req.f_hi_mhz)
    spec: dict[str, Any] = {
        "f_lo_hz": req.f_lo_mhz * 1e6,
        "f_hi_hz": req.f_hi_mhz * 1e6,
    }
    for key in ("gain_dbi_min", "polarization", "max_size_mm", "mount", "type"):
        val = getattr(req, key)
        if val is not None:
            spec[key] = val
    limit = max(1, min(50, req.limit))
    try:
        if req.partial is True:
            matches = parts_db.match({**spec, "partial": True}, limit=limit)
            mode = "overlap"
        elif req.partial is False:
            matches = parts_db.match({**spec, "partial": False}, limit=limit)
            mode = "full"
        else:
            matches = parts_db.match({**spec, "partial": False}, limit=limit)
            mode = "full"
            if not matches:
                # 2400–5800 has no single continuous row (dual-band SKUs
                # are one row per lobe). Fall back to overlapping parts.
                matches = parts_db.match({**spec, "partial": True}, limit=limit)
                mode = "overlap"
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _jsafe({"matches": matches, "coverage": mode})


# ---------------------------------------------------------------- kicad

class KicadReq(BaseModel):
    design_type: str
    f_mhz: float
    opts: dict[str, Any] | None = None


@router.post("/kicad")
def kicad(req: KicadReq) -> dict:
    if req.design_type not in _KICAD_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"unknown design_type {req.design_type!r}; "
                   f"expected one of {_KICAD_TYPES}",
        )
    if not (F_MIN_MHZ <= req.f_mhz <= F_MAX_MHZ):
        raise HTTPException(
            status_code=400,
            detail=f"f_mhz must be {F_MIN_MHZ:g}..{F_MAX_MHZ:g} MHz",
        )
    opts: dict[str, Any] = {}
    if req.opts:
        known = {"eps_r", "h_mm", "feed", "medium"}
        extra = set(req.opts) - known
        if extra:
            raise HTTPException(
                status_code=400, detail=f"unknown kicad opts: {sorted(extra)}"
            )
        opts = dict(req.opts)
        if "h_mm" in opts:
            opts["h_m"] = float(opts.pop("h_mm")) * 1e-3
        if "eps_r" in opts:
            opts["eps_r"] = float(opts["eps_r"])
        if "medium" in opts:
            _check_medium(opts["medium"])
    try:
        files = kicad_gen.kicad_files(req.design_type, req.f_mhz * 1e6, **opts)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token = uuid.uuid4().hex[:8]
    _KICAD_CACHE[token] = dict(files)
    while len(_KICAD_CACHE) > _KICAD_KEPT:
        _KICAD_CACHE.popitem(last=False)
    return {"token": token, "files": sorted(files)}


@router.get("/kicad/{token}")
def kicad_list(token: str) -> dict:
    entry = _KICAD_CACHE.get(token)
    if entry is None:
        raise HTTPException(status_code=404, detail="unknown or consumed kicad token")
    return {"token": token, "files": sorted(entry)}


@router.get("/kicad/{token}/{name}")
def kicad_file(token: str, name: str) -> PlainTextResponse:
    entry = _KICAD_CACHE.get(token)
    content = entry.pop(name, None) if entry is not None else None
    if content is None:
        raise HTTPException(
            status_code=404, detail="unknown token or file already consumed"
        )
    if not entry:
        _KICAD_CACHE.pop(token, None)
    return PlainTextResponse(
        content,
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


# ---------------------------------------------------------------- fields

class FieldsReq(BaseModel):
    f_mhz: float = 2450.0
    medium: str = "air"
    interface: bool = False
    n: int = 48
    max_steps: int = 400
    frame_every: int = 10
    pol_viz: bool = True
    budget_s: float = 120.0


def _run_fields(job: Job):
    p = job.params
    interface = bool(p["interface"])
    medium = p["medium"]
    if not interface and medium not in MEDIA:  # defensive; checked at POST
        raise HTTPException(status_code=400, detail=f"unknown medium {medium!r}")
    live = p.setdefault("live", {})
    stop = _BudgetStop(job)

    def cb(frame: dict) -> None:
        emax = float(frame.get("emax") or 0.0)
        norm = emax if emax > 0.0 else 1.0
        msg: dict[str, Any] = {
            "engine": "antenna_fields",
            "step": frame["step"],
            "t_s": frame["t_s"],
            "emax": emax,
            "e_rms": frame["e_rms"],
            "e_xy_png_b64": base64.b64encode(
                heatmap_png(frame["e_mid_xy"] / norm, 512)).decode("ascii"),
            "e_xz_png_b64": base64.b64encode(
                heatmap_png(frame["e_mid_xz"] / norm, 512)).decode("ascii"),
        }
        if interface:
            msg["e_rms_lo"] = frame["e_rms_lo"]
            msg["e_rms_hi"] = frame["e_rms_hi"]
        st = frame.get("stokes")
        if isinstance(st, dict) and "axial_ratio_db" in st:
            ar = np.asarray(st["axial_ratio_db"], dtype=np.float64)
            ar = np.nan_to_num(ar, nan=0.0, posinf=30.0, neginf=0.0)
            ar = np.clip(ar, 0.0, 30.0) / 30.0
            msg["ar_png_b64"] = base64.b64encode(
                heatmap_png(ar, 512)).decode("ascii")
        report(job, **msg)

    out = fdtd.fdtd_run(
        p["f_mhz"] * 1e6,
        medium=medium,
        interface=interface,
        n=int(p["n"]),
        frame_every=int(p["frame_every"]),
        max_steps=int(p["max_steps"]),
        pol_viz=bool(p["pol_viz"]),
        callback=cb,
        stop_flag=stop,
        live_params=live,
    )
    return _jsafe(out["info"])  # scalars + dx/dt/alpha numbers only


@router.post("/fields")
def fields_start(req: FieldsReq) -> dict:
    if not (1.0 <= req.f_mhz <= 6000.0):
        raise HTTPException(status_code=400, detail="f_mhz must be 1..6000 MHz")
    _check_medium(req.medium)
    if not (16 <= req.n <= 96):
        raise HTTPException(status_code=400, detail="n must be 16..96")
    if not (10 <= req.max_steps <= 5000):
        raise HTTPException(status_code=400, detail="max_steps must be 10..5000")
    if not (1 <= req.frame_every <= 1000):
        raise HTTPException(status_code=400, detail="frame_every must be 1..1000")
    if not (1.0 <= req.budget_s <= 3600.0):
        raise HTTPException(status_code=400, detail="budget_s must be 1..3600 s")
    params = {
        "f_mhz": req.f_mhz,
        "medium": req.medium,
        "interface": req.interface,
        "n": req.n,
        "max_steps": req.max_steps,
        "frame_every": req.frame_every,
        "pol_viz": req.pol_viz,
        "budget_s": req.budget_s,
        "live": {},
    }
    job = JOBS.submit("antenna_fields", _run_fields, params)
    return {"job_id": job.id}


# ---------------------------------------------------------------- evolve

class EvolveReq(BaseModel):
    f_mhz: float = 2450.0
    medium: str = "air"
    topology: str = "meander"
    hadamard_order: int = 64
    max_steps: int = 2000
    T_start: float = 2.0
    T_end: float = 0.02
    cooling: float = 0.995
    budget_s: float = 120.0
    seed: int | None = None


def _zpair(z: complex) -> dict:
    return {"re": float(np.real(z)), "im": float(np.imag(z))}


def _run_evolve(job: Job):
    p = job.params
    medium = p["medium"]
    if medium not in MEDIA:  # defensive; checked at POST
        raise HTTPException(status_code=400, detail=f"unknown medium {medium!r}")
    live = p.setdefault("live", {})
    stop = _BudgetStop(job)
    rng = np.random.default_rng(p.get("seed"))

    def cb(stats: dict) -> None:
        stats = dict(stats)
        geom = stats.pop("geom", None) or {}
        s11 = complex(geom.get("s11", 0.0))
        s11_db = 20.0 * math.log10(max(abs(s11), 1e-12))
        report(
            job,
            engine="antenna_evolve",
            step=stats.get("step", 0),
            T=stats.get("T"),
            E=stats.get("E"),
            best_E=stats.get("best_E"),
            accepts=stats.get("accepts", 0),
            terms=geom.get("terms"),
            points=np.asarray(geom.get("points", [])).tolist(),
            z_in=_zpair(complex(geom.get("z_in_ohm", 0.0))),
            gain_dbi=geom.get("gain_dbi"),
            s11_db=s11_db,
        )

    best, info = antenna_evo.antenna_sa(
        p["f_mhz"] * 1e6,
        medium=medium,
        topology=p["topology"],
        hadamard_order=int(p["hadamard_order"]),
        T_start=float(p["T_start"]),
        T_end=float(p["T_end"]),
        cooling=float(p["cooling"]),
        max_steps=int(p["max_steps"]),
        callback=cb,
        stop_flag=stop,
        live_params=live,
        rng=rng,
    )

    # final frame: far-field pattern of the best design on a θ×φ grid
    design = info["best_design"]
    th = np.linspace(0.0, math.pi, 91)
    ph = np.linspace(0.0, 2.0 * math.pi, 181, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    pat = np.asarray(design["pattern"](TH, PH), dtype=np.float64)
    pattern_png_b64 = base64.b64encode(heatmap_png(pat, 512)).decode("ascii")
    report(
        job,
        engine="antenna_evolve",
        step=info["steps"],
        best_E=info["best_E"],
        accepts=info["accepts"],
        terms=best["terms"],
        points=np.asarray(best["points"]).tolist(),
        z_in=_zpair(complex(best["z_in_ohm"])),
        gain_dbi=best["gain_dbi"],
        s11_db=best["s11_db"],
        pattern_png_b64=pattern_png_b64,
    )
    return _jsafe({
        "steps": info["steps"],
        "accepts": info["accepts"],
        "best_E": info["best_E"],
        "elapsed_s": info["elapsed_s"],
        "z_in": _zpair(complex(best["z_in_ohm"])),
        "gain_dbi": best["gain_dbi"],
        "s11_db": best["s11_db"],
        "terms": best["terms"],
        "seed_row": best["seed_row"],
        "points": np.asarray(best["points"]).tolist(),
        "resonance_note": design.get("resonance_note"),
        "pattern_png_b64": pattern_png_b64,
    })


@router.post("/evolve")
def evolve_start(req: EvolveReq) -> dict:
    if not (100.0 <= req.f_mhz <= 6000.0):
        raise HTTPException(status_code=400, detail="f_mhz must be 100..6000 MHz")
    _check_medium(req.medium)
    if req.topology != "meander":
        raise HTTPException(
            status_code=400,
            detail=f"unknown topology {req.topology!r}; supported: 'meander'",
        )
    ho = req.hadamard_order
    if not (4 <= ho <= 128) or ho & (ho - 1):
        raise HTTPException(
            status_code=400,
            detail="hadamard_order must be a power of 2, 4 ≤ n ≤ 128",
        )
    if not (1 <= req.max_steps <= 20000):
        raise HTTPException(status_code=400, detail="max_steps must be 1..20000")
    if not (1e-3 <= req.T_start <= 100.0):
        raise HTTPException(status_code=400, detail="T_start must be 1e-3..100")
    if not (1e-6 <= req.T_end < req.T_start):
        raise HTTPException(status_code=400, detail="T_end must be 1e-6..T_start")
    if not (0.9 <= req.cooling < 1.0):
        raise HTTPException(status_code=400, detail="cooling must be 0.9..<1.0")
    if not (1.0 <= req.budget_s <= 3600.0):
        raise HTTPException(status_code=400, detail="budget_s must be 1..3600 s")
    params = {
        "f_mhz": req.f_mhz,
        "medium": req.medium,
        "topology": req.topology,
        "hadamard_order": req.hadamard_order,
        "max_steps": req.max_steps,
        "T_start": req.T_start,
        "T_end": req.T_end,
        "cooling": req.cooling,
        "budget_s": req.budget_s,
        "live": {},
    }
    if req.seed is not None:
        params["seed"] = req.seed
    job = JOBS.submit("antenna_evolve", _run_evolve, params)
    return {"job_id": job.id}


# ---------------------------------------------------------------- smith

class SmithReq(BaseModel):
    f_lo_mhz: float
    f_hi_mhz: float
    n_points: int = 21
    z0: float = 50.0
    medium: str = "air"
    source: str = "dipole"                    # "dipole" | "wire"
    points: list[list[float]] | None = None   # wire vertices (m) for "wire"
    wire_radius_m: float = 1e-4


@router.post("/smith")
def smith_sweep(req: SmithReq) -> dict:
    """Z_in(f) → Γ(f) sweep for the Smith-chart panel.

    The trace is computed with the real thin-wire MoM solver
    (`antenna_evo.wire_mom`) at each frequency — source "dipole" builds the
    textbook shortened half-wave wire from `em_physics.build_dipole`,
    source "wire" takes explicit polyline vertices (e.g. the last evolved
    geometry).  Γ = (Z−Z₀)/(Z+Z₀) is the exact bilinear transform.
    """
    _check_band(req.f_lo_mhz, req.f_hi_mhz)
    _check_medium(req.medium)
    if not (3 <= req.n_points <= 41):
        raise HTTPException(status_code=400, detail="n_points must be 3..41")
    if not (1.0 <= req.z0 <= 1000.0):
        raise HTTPException(status_code=400, detail="z0 must be 1..1000 Ω")
    f_c_hz = 0.5 * (req.f_lo_mhz + req.f_hi_mhz) * 1e6
    if req.source == "dipole":
        dims = em_build_dipole(f_c_hz, req.medium)["dimensions_m"]
        L = float(dims["length"])
        pts = np.array([[-L / 2, 0.0, 0.0], [L / 2, 0.0, 0.0]])
        radius = float(dims.get("wire_radius", req.wire_radius_m))
    elif req.source == "wire":
        if req.points is None:
            raise HTTPException(status_code=400, detail="source 'wire' needs points")
        pts = np.asarray(req.points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or not (2 <= pts.shape[0] <= 512):
            raise HTTPException(status_code=400, detail="points must be (M,3), 2 ≤ M ≤ 512")
        if not np.all(np.isfinite(pts)) or not np.any(np.ptp(pts, axis=0) > 0):
            raise HTTPException(status_code=400, detail="points must be finite with nonzero span")
        radius = req.wire_radius_m
    else:
        raise HTTPException(status_code=400, detail="source must be 'dipole' or 'wire'")
    if not (1e-7 <= radius <= 1e-2):
        raise HTTPException(status_code=400, detail="wire_radius_m out of range")

    sweep = []
    for f_mhz in np.linspace(req.f_lo_mhz, req.f_hi_mhz, req.n_points):
        try:
            r = antenna_evo.wire_mom(pts, float(f_mhz) * 1e6,
                                     radius_m=radius, medium=req.medium)
        except Exception as exc:  # singular geometry at this f — keep sweeping
            sweep.append({"f_mhz": float(f_mhz), "error": str(exc)[:200]})
            continue
        z = complex(r["z_in_ohm"])
        g = (z - req.z0) / (z + req.z0)
        sweep.append({
            "f_mhz": float(f_mhz),
            "z": _zpair(z),
            "gamma": _zpair(g),
            "s11_db": float(20.0 * math.log10(max(abs(g), 1e-12))),
        })
    return _jsafe({
        "source": req.source,
        "z0": req.z0,
        "medium": req.medium,
        "f_center_mhz": f_c_hz / 1e6,
        "sweep": sweep,
    })


# ---------------------------------------------------------------- survey

class SurveyPoint(BaseModel):
    lat: float
    lon: float
    h_m: float = 10.0


class SurveyReq(BaseModel):
    tx: SurveyPoint
    rx: SurveyPoint | None = None  # omitted / coincident → 1 km east hop
    f_mhz: float
    p_tx_dbw: float = 0.0
    g_tx_dbi: float = 2.15
    g_rx_dbi: float = 2.15
    medium: str = "air"
    n: int = 200
    zoom: int = 12


def _check_site(p: SurveyPoint) -> None:
    if not (-85.0 <= p.lat <= 85.0 and -180.0 <= p.lon <= 180.0):
        raise HTTPException(status_code=400, detail="lat/lon out of range")
    if not (0.0 <= p.h_m <= 500.0):
        raise HTTPException(status_code=400, detail="h_m must be 0..500 m")


@router.post("/survey")
def survey(req: SurveyReq) -> dict:
    """Virtual site survey over open SRTM terrain tiles (no API key).

    `site_survey.survey` builds the terrain profile along the great-circle
    path, applies the 4/3-earth bulge + first-Fresnel geometry, Deygout
    knife-edge diffraction loss, and closes the link with
    `em_physics.link_budget`.
    """
    _check_site(req.tx)
    rx = req.rx or req.tx
    _check_site(rx)
    if not (F_MIN_MHZ <= req.f_mhz <= F_MAX_MHZ):
        raise HTTPException(status_code=400, detail=f"f_mhz must be {F_MIN_MHZ:g}..{F_MAX_MHZ:g}")
    _check_medium(req.medium)
    if not (16 <= req.n <= 1000):
        raise HTTPException(status_code=400, detail="n must be 16..1000")
    if not (3 <= req.zoom <= 15):
        raise HTTPException(status_code=400, detail="zoom must be 3..15")
    try:
        out = site_survey.survey(
            req.tx.model_dump(), rx.model_dump(), req.f_mhz,
            p_tx_dbw=req.p_tx_dbw, g_tx_dbi=req.g_tx_dbi, g_rx_dbi=req.g_rx_dbi,
            medium=req.medium, n=req.n, zoom=req.zoom,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # tile fetch failures etc.
        raise HTTPException(status_code=502, detail=f"terrain fetch failed: {e}")
    return _jsafe(out)


class SurveyMapReq(BaseModel):
    tx: SurveyPoint
    rx: SurveyPoint | None = None
    zoom: int = 11
    size: int = 256
    heightmap: bool = False
    imagery: bool = False


@router.post("/survey/map")
def survey_map(req: SurveyMapReq) -> dict:
    """Terrain heatmap covering both sites (Terrarium tiles → PNG b64).

    Returns the geographic bbox so the frontend can draw the path/markers
    in canvas coordinates. When heightmap=True, also returns the raw 2-D
    elevation grid for the terrain 3-D view.
    """
    _check_site(req.tx)
    rx = req.rx or req.tx
    _check_site(rx)
    if not (3 <= req.zoom <= 15):
        raise HTTPException(status_code=400, detail="zoom must be 3..15")
    if not (64 <= req.size <= 1024):
        raise HTTPException(status_code=400, detail="size must be 64..1024")
    try:
        out = site_survey.terrain_map(
            req.tx.lat, req.tx.lon, rx.lat, rx.lon,
            zoom=req.zoom, size=req.size,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"terrain fetch failed: {e}")
    elev = np.asarray(out.pop("elev"), dtype=np.float64)
    lo, hi = float(np.min(elev)), float(np.max(elev))
    # colour the PNG by metres, but do NOT min-max a huge tile — the
    # window is already a tight path neighbourhood.  A flat field stays
    # nearly uniform (a 5 m ripple over 2 km is ~invisible, correctly).
    # 30 m of relief fills the ramp; bigger relief saturates, smaller
    # relief stays quiet.  This is a *display* stretch, not the 3-D scale.
    colour_span = max(30.0, hi - lo)
    norm = np.clip((elev - lo) / colour_span, 0.0, 1.0)
    resp = {
        **out,
        "elev_lo_m": lo,
        "elev_hi_m": hi,
        "relief_m": hi - lo,
        "map_png_b64": base64.b64encode(heatmap_png(norm, 512)).decode("ascii"),
    }
    if req.heightmap:
        resp["heightmap"] = [[float(v) for v in row] for row in elev.tolist()]
    if req.imagery:
        try:
            rgb = site_survey.imagery_map(
                resp["lat_lo"], resp["lat_hi"], resp["lon_lo"], resp["lon_hi"],
                zoom=min(16, req.zoom + 3), size=min(256, max(req.size, 128)),
            )
            resp["imagery_png_b64"] = base64.b64encode(write_png(rgb)).decode("ascii")
        except Exception as exc:
            resp["imagery_png_b64"] = None
            resp["imagery_error"] = str(exc)[:240]
    return _jsafe(resp)
