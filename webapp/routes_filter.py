"""PCB RF-filter lab — design, S-parameter sweep, KiCad export, SA evolve.

Wraps `rf_filter` the way `routes_antenna` wraps the antenna stack:

* ``POST /design`` / ``POST /sweep`` / ``POST /kicad`` are synchronous.
  KiCad uses the same one-shot token cache as the antenna exporter.
* ``POST /evolve`` runs `rf_filter.filter_sa` as a JobManager job;
  progress frames carry IL/RL/rejection + a layout preview, and the
  final result includes the best design + a full S-parameter sweep.
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from .. import kicad_gen, rf_filter
from .jobs import JOBS, Job, report
from .routes_hadamard import _jsafe
from .routes_search import _BudgetStop

router = APIRouter(prefix="/api/filter")

_KICAD_CACHE: OrderedDict[str, dict[str, str]] = OrderedDict()
_KICAD_KEPT = 8

F_MIN, F_MAX = 10.0, 40000.0


def _check_f(mhz: float) -> None:
    if not (F_MIN <= mhz <= F_MAX):
        raise HTTPException(
            status_code=400,
            detail=f"frequency must be {F_MIN:g}..{F_MAX:g} MHz",
        )


def _design_from_req(kind: str, proto: str, n: int, f_c_mhz: float | None,
                     f_lo_mhz: float | None, f_hi_mhz: float | None,
                     eps_r: float, h_mm: float, tan_delta: float,
                     ripple_db: float) -> dict:
    if kind not in rf_filter.KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown kind {kind!r}; expected {rf_filter.KINDS}",
        )
    if proto not in rf_filter.PROTOS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown proto {proto!r}; expected {rf_filter.PROTOS}",
        )
    if not (1 <= n <= 11):
        raise HTTPException(status_code=400, detail="n must be 1..11")
    try:
        return rf_filter.design_filter(
            kind,
            f_c=None if f_c_mhz is None else f_c_mhz * 1e6,
            f_lo=None if f_lo_mhz is None else f_lo_mhz * 1e6,
            f_hi=None if f_hi_mhz is None else f_hi_mhz * 1e6,
            n=n, proto=proto, eps_r=eps_r, h_m=h_mm * 1e-3,
            tan_delta=tan_delta, ripple_db=ripple_db,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _pack(design: dict, n_sweep: int = 81) -> dict:
    sw = rf_filter.sweep(design, n_points=n_sweep)
    met = rf_filter.metrics(design, sw)
    lay = rf_filter.layout_mm(design)
    prev = rf_filter.preview_from_layout(lay)
    # complex → {re,im} for JSON
    sw_out = {
        "f_mhz": [f / 1e6 for f in sw["f_hz"]],
        "s11_db": sw["s11_db"],
        "s21_db": sw["s21_db"],
        "s11": [{"re": z.real, "im": z.imag} for z in sw["s11"]],
        "s21": [{"re": z.real, "im": z.imag} for z in sw["s21"]],
    }
    return {
        "design": rf_filter.design_public(design),
        "params": rf_filter.design_params(design),
        "metrics": met,
        "sweep": sw_out,
        "preview": prev,
        "layout": lay,
    }


class DesignReq(BaseModel):
    kind: str = "lpf"
    proto: str = "butterworth"
    n: int = 5
    f_c_mhz: float | None = 2450.0
    f_lo_mhz: float | None = None
    f_hi_mhz: float | None = None
    eps_r: float = 4.4
    h_mm: float = 1.6
    tan_delta: float = 0.02
    ripple_db: float = 0.1
    n_sweep: int = 81


@router.post("/design")
def design(req: DesignReq) -> dict:
    if req.f_c_mhz is not None:
        _check_f(req.f_c_mhz)
    if req.f_lo_mhz is not None:
        _check_f(req.f_lo_mhz)
    if req.f_hi_mhz is not None:
        _check_f(req.f_hi_mhz)
    if not (1.0 <= req.eps_r <= 20.0):
        raise HTTPException(status_code=400, detail="eps_r must be 1..20")
    if not (0.05 <= req.h_mm <= 6.0):
        raise HTTPException(status_code=400, detail="h_mm must be 0.05..6")
    n_sweep = max(11, min(401, req.n_sweep))
    d = _design_from_req(req.kind, req.proto, req.n, req.f_c_mhz,
                         req.f_lo_mhz, req.f_hi_mhz, req.eps_r, req.h_mm,
                         req.tan_delta, req.ripple_db)
    return _jsafe(_pack(d, n_sweep))


class SweepReq(BaseModel):
    design: dict[str, Any]
    f_lo_mhz: float | None = None
    f_hi_mhz: float | None = None
    n_points: int = 81


@router.post("/sweep")
def sweep(req: SweepReq) -> dict:
    if "sections" not in req.design:
        raise HTTPException(status_code=400, detail="design.sections required")
    n = max(11, min(401, req.n_points))
    flo = None if req.f_lo_mhz is None else req.f_lo_mhz * 1e6
    fhi = None if req.f_hi_mhz is None else req.f_hi_mhz * 1e6
    sw = rf_filter.sweep(req.design, f_lo=flo, f_hi=fhi, n_points=n)
    met = rf_filter.metrics(req.design, sw)
    return _jsafe({
        "metrics": met,
        "sweep": {
            "f_mhz": [f / 1e6 for f in sw["f_hz"]],
            "s11_db": sw["s11_db"],
            "s21_db": sw["s21_db"],
        },
    })


class KicadReq(BaseModel):
    kind: str = "lpf"
    proto: str = "butterworth"
    n: int = 5
    f_c_mhz: float = 2450.0
    f_lo_mhz: float | None = None
    f_hi_mhz: float | None = None
    eps_r: float = 4.4
    h_mm: float = 1.6
    tan_delta: float = 0.02
    ripple_db: float = 0.1
    design: dict[str, Any] | None = None


@router.post("/kicad")
def kicad(req: KicadReq) -> dict:
    _check_f(req.f_c_mhz)
    if req.design is not None:
        design = req.design
        if "sections" not in design:
            raise HTTPException(status_code=400, detail="design.sections required")
    else:
        design = _design_from_req(req.kind, req.proto, req.n, req.f_c_mhz,
                                  req.f_lo_mhz, req.f_hi_mhz, req.eps_r,
                                  req.h_mm, req.tan_delta, req.ripple_db)
    try:
        files = kicad_gen.kicad_files(
            "filter", design["f_c"],
            design=design, eps_r=design.get("eps_r", 4.4),
            h_m=design.get("h_m", 1.6e-3),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    token = uuid.uuid4().hex[:8]
    _KICAD_CACHE[token] = dict(files)
    while len(_KICAD_CACHE) > _KICAD_KEPT:
        _KICAD_CACHE.popitem(last=False)
    packed = _pack(design)
    return _jsafe({
        "token": token,
        "files": sorted(files),
        "preview": packed["preview"],
        "params": packed["params"],
        "metrics": packed["metrics"],
    })


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
            status_code=404, detail="unknown token or file already consumed",
        )
    if not entry:
        _KICAD_CACHE.pop(token, None)
    return PlainTextResponse(
        content,
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


class EvolveReq(BaseModel):
    kind: str = "lpf"
    proto: str = "butterworth"
    n: int = 5
    f_c_mhz: float = 2450.0
    f_lo_mhz: float | None = None
    f_hi_mhz: float | None = None
    eps_r: float = 4.4
    h_mm: float = 1.6
    tan_delta: float = 0.02
    ripple_db: float = 0.1
    hadamard_order: int = 32
    max_steps: int = 200
    T_start: float = 1.0
    T_end: float = 0.02
    cooling: float = 0.995
    budget_s: float = 60.0
    seed: int | None = None


def _run_evolve(job: Job):
    p = job.params
    live = p.setdefault("live", {})
    stop = _BudgetStop(job)
    rng = __import__("numpy").random.default_rng(p.get("seed"))

    def cb(stats: dict) -> None:
        lay = stats.pop("layout", None)
        prev = rf_filter.preview_from_layout(lay) if lay else None
        report(
            job,
            engine="filter_evolve",
            step=stats.get("step", 0),
            T=stats.get("T"),
            E=stats.get("E"),
            best_E=stats.get("best_E"),
            accepts=stats.get("accepts", 0),
            il_db=stats.get("il_db"),
            rl_db=stats.get("rl_db"),
            rejection_db=stats.get("rejection_db"),
            preview=prev,
        )

    best, info = rf_filter.filter_sa(
        kind=p["kind"], proto=p["proto"], n=int(p["n"]),
        f_c=p["f_c_mhz"] * 1e6,
        f_lo=None if p.get("f_lo_mhz") is None else p["f_lo_mhz"] * 1e6,
        f_hi=None if p.get("f_hi_mhz") is None else p["f_hi_mhz"] * 1e6,
        eps_r=p["eps_r"], h_m=p["h_mm"] * 1e-3,
        tan_delta=p["tan_delta"], ripple_db=p["ripple_db"],
        hadamard_order=int(p["hadamard_order"]),
        T_start=float(p["T_start"]), T_end=float(p["T_end"]),
        cooling=float(p["cooling"]), max_steps=int(p["max_steps"]),
        callback=cb, stop_flag=stop, live_params=live, rng=rng,
    )
    packed = _pack(best["design"])
    report(
        job,
        engine="filter_evolve",
        step=info["steps"],
        best_E=info["best_E"],
        accepts=info["accepts"],
        il_db=best["terms"].get("il_db"),
        rl_db=best["terms"].get("rl_db"),
        rejection_db=best["terms"].get("rejection_db"),
        preview=packed["preview"],
    )
    return _jsafe({
        "steps": info["steps"],
        "accepts": info["accepts"],
        "best_E": info["best_E"],
        "elapsed_s": info["elapsed_s"],
        "terms": best["terms"],
        "design": packed["design"],
        "params": packed["params"],
        "metrics": packed["metrics"],
        "sweep": packed["sweep"],
        "preview": packed["preview"],
        "seed_row": best.get("seed_row"),
    })


@router.post("/evolve")
def evolve_start(req: EvolveReq) -> dict:
    _check_f(req.f_c_mhz)
    if req.kind not in rf_filter.KINDS:
        raise HTTPException(status_code=400, detail=f"unknown kind {req.kind!r}")
    if req.proto not in rf_filter.PROTOS:
        raise HTTPException(status_code=400, detail=f"unknown proto {req.proto!r}")
    ho = req.hadamard_order
    if not (4 <= ho <= 128) or ho & (ho - 1):
        raise HTTPException(
            status_code=400,
            detail="hadamard_order must be a power of 2, 4 ≤ n ≤ 128",
        )
    if not (1 <= req.max_steps <= 5000):
        raise HTTPException(status_code=400, detail="max_steps must be 1..5000")
    if not (1.0 <= req.budget_s <= 3600.0):
        raise HTTPException(status_code=400, detail="budget_s must be 1..3600 s")
    params = req.model_dump()
    params["live"] = {}
    job = JOBS.submit("filter_evolve", _run_evolve, params)
    return {"job_id": job.id}
