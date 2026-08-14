"""Micromagnetic simulation routes — Phase 3 live energy-field lab.

Runs `micromag.micromag_sa` as a JobManager job with the same
callback/stop_flag/live_params wiring as the search routes, but reports
the full physics: every 500-step frame carries the energy decomposition
(E_exch / E_dem / E_anis from `total_energy`), and every
`field_every_steps` the per-site `site_energy` density and the |dF|
`energy_gradient` map are rendered as heatmap PNGs (`heatmap_png`'s
blue→black→red ramp) for the three-panel live view.

Export reuses the kind-agnostic `/api/search/{job_id}/export` endpoint —
`_package_result` stores the verified matrix on `job.matrix`, which is
all that endpoint needs.  Progress streams over the existing
`WS /ws/job/{job_id}`; the `{"op":"set",...}` retune op writes
cooling/lam_ex/lam_ani/lam_goal into `job.params["live"]`, which
`micromag_sa` reads every 500 steps.

Goal attraction: `SimReq.goal_order` (must equal `order`, and the order
must be in the library) loads that library matrix as the anneal's target
— `micromag_sa` adds `lam_goal` per entry disagreeing with ±goal and the
frames carry `E_goal` / `goal_agree`.
"""

from __future__ import annotations

import base64
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import micromag
from ..hadamard import sylvester
from ._png import heatmap_png, matrix_png
from .jobs import JOBS, Job, report
from .routes_search import (
    LIB_DIR,
    PREVIEW_EVERY,
    PREVIEW_MAX,
    _BudgetStop,
    _package_result,
)

MAX_ORDER = 1024
FIELD_MAX = 512  # site-energy/gradient heatmaps only up to this order

router = APIRouter(prefix="/api")


def _start_matrix(job: Job) -> np.ndarray | None:
    """Resolve the optional start matrix; raises HTTPException(400)."""
    p = job.params
    order = p["order"]
    method = p.get("start", "random")
    if method == "random":
        return None
    if method == "sylvester":
        H = sylvester(order)
        if H is None:
            raise HTTPException(
                status_code=400, detail=f"sylvester needs a power-of-2 order, got {order}"
            )
        return H
    if method == "library":
        path = LIB_DIR / f"hadamard_{order}.csv"
        if not path.is_file():
            raise HTTPException(
                status_code=400, detail=f"order {order} not in library ({path})"
            )
        return np.loadtxt(path, delimiter=",", dtype=np.int8)
    raise HTTPException(status_code=400, detail=f"unknown start method {method!r}")


def _goal_matrix(job: Job) -> np.ndarray | None:
    """Resolve the optional goal-attraction target; raises HTTPException(400)."""
    p = job.params
    goal_order = p.get("goal_order")
    if goal_order is None:
        return None
    path = LIB_DIR / f"hadamard_{goal_order}.csv"
    if not path.is_file():
        raise HTTPException(
            status_code=400, detail=f"goal order {goal_order} not in library ({path})"
        )
    return np.loadtxt(path, delimiter=",", dtype=np.int8)


def _sim_reporter(job: Job, order: int, field_every: int,
                  lam_ex0: float, lam_ani0: float):
    """Build the SA callback: stat frames with energy decomposition,
    throttled matrix previews, and field/gradient heatmap frames."""
    t0 = time.monotonic()
    last_preview = [0.0]
    live = job.params["live"]

    def cb(stats: dict) -> None:
        stats = dict(stats)
        H = stats.pop("H", None)
        lam_ex = float(live.get("lam_ex") or lam_ex0)
        lam_ani = float(live.get("lam_ani") or lam_ani0)
        frame: dict[str, Any] = {
            "engine": "micromag_sim",
            "elapsed_s": round(time.monotonic() - t0, 3),
            **stats,
        }
        if H is not None:
            _, E_exch, E_dem, E_anis = micromag.total_energy(
                H, lam_ex=lam_ex, lam_ani=lam_ani
            )
            frame.update(E_exch=E_exch, E_dem=E_dem, E_anis=E_anis)
        report(job, **frame)

        if H is None:
            return
        now = time.monotonic()
        step = stats.get("step", 0)
        if order <= FIELD_MAX and field_every > 0 and step % field_every == 0:
            field = micromag.site_energy(H, lam_ex=lam_ex, lam_anis=lam_ani)
            grad = micromag.energy_gradient(H)
            flux = micromag.flux_map(H)
            report(
                job,
                engine="micromag_sim",
                step=step,
                field_png_b64=base64.b64encode(heatmap_png(field, 512)).decode("ascii"),
                grad_png_b64=base64.b64encode(heatmap_png(grad, 512)).decode("ascii"),
                flux_png_b64=base64.b64encode(heatmap_png(flux, 512)).decode("ascii"),
            )
        if order <= PREVIEW_MAX and now - last_preview[0] >= PREVIEW_EVERY:
            last_preview[0] = now
            png = matrix_png(np.asarray(H, dtype=np.int8), 256)
            report(
                job,
                engine="micromag_sim",
                matrix_png_b64=base64.b64encode(png).decode("ascii"),
            )

    return cb


def _run_sim(job: Job):
    p = job.params
    order = p["order"]
    live = p.setdefault("live", {})
    rng = np.random.default_rng(p.get("seed"))
    start = _start_matrix(job)
    goal = _goal_matrix(job)
    H, info = micromag.micromag_sa(
        order,
        T_start=p.get("T_start", 10.0),
        T_end=p.get("T_end", 0.01),
        cooling=p.get("cooling", 0.999),
        lam_ex=p.get("lam_ex", 0.0),
        lam_ani=p.get("lam_ani", 0.0),
        n_swap=int(p.get("n_swap", 3)),
        max_steps=int(p.get("max_steps", 10**9)),
        rng=rng,
        start=start,
        goal=goal,
        lam_goal=float(p.get("lam_goal", 0.5)),
        callback=_sim_reporter(
            job,
            order,
            int(p.get("field_every_steps", 2500)),
            float(p.get("lam_ex", 0.0)),
            float(p.get("lam_ani", 0.0)),
        ),
        stop_flag=_BudgetStop(job),
        live_params=live,
    )
    return _package_result(
        job, H, {"best_E": info.get("best_E"), "info": info}, "micromag_sim"
    )


class SimReq(BaseModel):
    order: int
    T_start: float = 10.0
    T_end: float = 0.01
    cooling: float = 0.999
    lam_ex: float = 0.0
    lam_ani: float = 0.0
    n_swap: int = 3
    budget_s: float = 30.0
    seed: int | None = None
    field_every_steps: int = 2500
    start: str = "random"
    goal_order: int | None = None
    lam_goal: float = 0.5


@router.post("/sim/micromag")
def sim_start(req: SimReq) -> dict:
    if not (4 <= req.order <= MAX_ORDER) or req.order % 4 != 0:
        raise HTTPException(
            status_code=400, detail=f"order must be a multiple of 4, 4 ≤ n ≤ {MAX_ORDER}"
        )
    if req.start not in ("random", "sylvester", "library"):
        raise HTTPException(status_code=400, detail=f"unknown start method {req.start!r}")
    if req.start == "sylvester" and sylvester(req.order) is None:
        raise HTTPException(
            status_code=400, detail=f"sylvester needs a power-of-2 order, got {req.order}"
        )
    if req.start == "library" and not (LIB_DIR / f"hadamard_{req.order}.csv").is_file():
        raise HTTPException(
            status_code=400, detail=f"order {req.order} not in library"
        )
    if req.goal_order is not None:
        if req.goal_order != req.order:
            raise HTTPException(
                status_code=400,
                detail=f"goal_order {req.goal_order} must equal order {req.order}",
            )
        if req.start == "library":
            raise HTTPException(
                status_code=400,
                detail="start=library with a goal is degenerate "
                       "(the start already is the goal)",
            )
        if not (LIB_DIR / f"hadamard_{req.goal_order}.csv").is_file():
            raise HTTPException(
                status_code=400, detail=f"goal order {req.goal_order} not in library"
            )
    params = {
        "order": req.order,
        "T_start": req.T_start,
        "T_end": req.T_end,
        "cooling": req.cooling,
        "lam_ex": req.lam_ex,
        "lam_ani": req.lam_ani,
        "n_swap": req.n_swap,
        "budget_s": req.budget_s,
        "field_every_steps": req.field_every_steps,
        "start": req.start,
        "lam_goal": req.lam_goal,
        "live": {},
    }
    if req.goal_order is not None:
        params["goal_order"] = req.goal_order
    if req.seed is not None:
        params["seed"] = req.seed
    job = JOBS.submit("micromag_sim", _run_sim, params)
    return {"job_id": job.id}
