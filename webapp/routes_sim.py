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

Goal attraction: `SimReq.goal_order` (must be in the library, ≥ `order`,
and ≤ MAX_ORDER) loads that library matrix as the anneal's target —
`micromag_sa` adds `lam_goal` per entry disagreeing with ±goal and the
frames carry `E_goal` / `goal_agree`.  When `goal_order > order` the
anneal runs at the goal's order and a library start is Kronecker-lifted,
H(order) ⊗ H(goal_order/order) — evolving *from* a smaller known solution
*toward* a larger one.  A frozen anneal is reheated (best-so-far perturbed
5%) and run again until the `budget_s` stop fires or Hadamard is found.
"""

from __future__ import annotations

import base64
import math
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from .. import micromag
from ..hadamard import perturb, random_seed, sylvester
from ._png import heatmap_png, matrix_png
from .jobs import JOBS, Job, report
from .routes_hadamard import _jsafe
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


def _start_matrix(job: Job, sim_order: int) -> np.ndarray | None:
    """Resolve the optional start matrix; raises HTTPException(400).

    When a larger library goal is active (`sim_order > order`) a sylvester
    or library start is Kronecker-lifted, H(order) ⊗ H(sim_order/order),
    so the anneal begins from the known solution embedded at the goal's
    order — the quotient order must be in the library.
    """
    p = job.params
    order = p["order"]
    method = p.get("start", "random")
    if method == "random":
        return None
    if method == "sylvester":
        H = sylvester(order)
        if H is None:
            raise HTTPException(
                status_code=400,
                detail=f"sylvester needs a power-of-2 order, got {order}",
            )
    elif method == "library":
        path = LIB_DIR / f"hadamard_{order}.csv"
        if not path.is_file():
            raise HTTPException(
                status_code=400, detail=f"order {order} not in library ({path})"
            )
        H = np.loadtxt(path, delimiter=",", dtype=np.int8)
    else:
        raise HTTPException(status_code=400, detail=f"unknown start method {method!r}")
    if sim_order > order:
        q = sim_order // order
        lift_path = LIB_DIR / f"hadamard_{q}.csv"
        if not lift_path.is_file():
            raise HTTPException(
                status_code=400,
                detail=f"cannot lift start {order} → {sim_order}: "
                       f"quotient order {q} not in library",
            )
        H = np.kron(H, np.loadtxt(lift_path, delimiter=",", dtype=np.int8))
    return np.asarray(H, dtype=np.int8)


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
    last_field = [True]  # force a heatmap+tiles frame on the first callback
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
            # cheap O(n²); ride every progress frame so the FLUX TILES
            # panel does not wait for field_every (default 2500 steps)
            frame["flux_tiles"] = micromag.flux_tiles(H)
        report(job, **frame)

        if H is None:
            return
        now = time.monotonic()
        step = stats.get("step", 0)
        first_field = last_field[0]
        if (order <= FIELD_MAX and field_every > 0
                and (first_field or step % field_every == 0)):
            last_field[0] = False
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
                flux_tiles=micromag.flux_tiles(H),
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
    sim_order = int(p.get("goal_order") or order)  # anneal at the goal's order
    live = p.setdefault("live", {})
    rng = np.random.default_rng(p.get("seed"))
    start = _start_matrix(job, sim_order)
    goal = _goal_matrix(job)

    # A single anneal freezes after ln(T_end/T_start)/ln(cooling) steps
    # (~7k with the defaults — a fraction of a second), long before
    # `budget_s` matters.  Run it as a reheat chain: when a segment
    # freezes without finding Hadamard, perturb the best-so-far matrix
    # and anneal again until the budget stop fires or Hadamard is found.
    field_every = int(p.get("field_every_steps", 2500))
    lam_ex0 = float(p.get("lam_ex", 0.0))
    lam_ani0 = float(p.get("lam_ani", 0.0))
    stop = _BudgetStop(job)

    best_H: np.ndarray | None = None
    best_E = math.inf
    agg = {"steps": 0, "accepts": 0, "segments": 0}
    step_off = 0

    cur = start
    while not stop.is_set():
        base_cb = _sim_reporter(job, sim_order, field_every, lam_ex0, lam_ani0)

        def cb(stats: dict, _off=step_off, _base=base_cb) -> None:
            stats = dict(stats)
            stats["step"] = stats.get("step", 0) + _off
            _base(stats)

        H, info = micromag.micromag_sa(
            sim_order,
            T_start=p.get("T_start", 10.0),
            T_end=p.get("T_end", 0.01),
            cooling=p.get("cooling", 0.999),
            lam_ex=lam_ex0,
            lam_ani=lam_ani0,
            n_swap=int(p.get("n_swap", 3)),
            max_steps=int(p.get("max_steps", 10**9)),
            rng=rng,
            start=cur,
            goal=goal,
            lam_goal=float(p.get("lam_goal", 0.5)),
            lam_tile=float(p.get("lam_tile", 0.0)),
            callback=cb,
            stop_flag=stop,
            live_params=live,
        )
        step_off += info["steps"]
        agg["steps"] += info["steps"]
        agg["accepts"] += info["accepts"]
        agg["segments"] += 1
        if info["best_E"] < best_E:
            best_H, best_E = H, info["best_E"]
        if info["hadamard"] or stop.is_set():
            break
        report(job, engine="micromag_sim", step=step_off,
               reheat=agg["segments"], best_E=best_E)
        cur = perturb(best_H, rng=rng, frac=0.05)

    if best_H is None:  # cancelled before the first segment reported
        best_H = random_seed(sim_order, rng).astype(np.int8)
        best_E = math.inf
    info_out = {**agg, "best_E": best_E,
                "hadamard": bool(best_E < 1e-6)}
    if goal is not None and math.isfinite(best_E):
        G = np.asarray(goal, dtype=np.int8).astype(np.int64)
        corr = int((best_H.astype(np.int64) * G).sum())
        n = int(best_H.shape[0])
        info_out["goal_agree"] = (n * n + abs(corr)) / (2.0 * n * n)
    return _package_result(job, best_H, info_out, "micromag_sim")


class SimReq(BaseModel):
    order: int
    T_start: float = 10.0
    T_end: float = 0.01
    cooling: float = 0.999
    lam_ex: float = 0.0
    lam_ani: float = 0.0
    n_swap: int = 3
    budget_s: float = 300.0
    seed: int | None = None
    field_every_steps: int = 2500
    start: str = "random"
    goal_order: int | None = None
    lam_goal: float = 0.5
    lam_tile: float = 0.0


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
        if not (LIB_DIR / f"hadamard_{req.goal_order}.csv").is_file():
            raise HTTPException(
                status_code=400, detail=f"goal order {req.goal_order} not in library"
            )
        if req.goal_order > MAX_ORDER:
            raise HTTPException(
                status_code=400, detail=f"goal order {req.goal_order} > {MAX_ORDER}"
            )
        if req.goal_order < req.order:
            raise HTTPException(
                status_code=400,
                detail=f"goal_order {req.goal_order} < order {req.order}",
            )
        if req.goal_order == req.order and req.start == "library":
            raise HTTPException(
                status_code=400,
                detail="start=library with an equal-order goal is degenerate "
                       "(the start already is the goal)",
            )
        if req.goal_order > req.order:
            # the sim anneals at the goal's order; a sylvester/library
            # start is Kronecker-lifted H(order) ⊗ H(goal_order/order),
            # which needs the quotient order in the library
            if req.goal_order % req.order != 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"goal_order {req.goal_order} is not a multiple "
                           f"of order {req.order}",
                )
            if req.start in ("sylvester", "library") and not (
                LIB_DIR / f"hadamard_{req.goal_order // req.order}.csv"
            ).is_file():
                raise HTTPException(
                    status_code=400,
                    detail=f"cannot lift start {req.order} → {req.goal_order}: "
                           f"quotient order {req.goal_order // req.order} "
                           "not in library",
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
        "lam_tile": req.lam_tile,
        "live": {},
    }
    if req.goal_order is not None:
        params["goal_order"] = req.goal_order
    if req.seed is not None:
        params["seed"] = req.seed
    job = JOBS.submit("micromag_sim", _run_sim, params)
    return {"job_id": job.id}


@router.get("/sim/flux-tiles")
def flux_tiles_get(
    order: int = Query(..., ge=4, le=MAX_ORDER),
    start: str = Query("sylvester"),
) -> dict:
    """Inspect the H.8 flux-tile catalog of a start matrix (no anneal)."""
    if order % 4 != 0:
        raise HTTPException(status_code=400, detail="order must be a multiple of 4")
    if start not in ("sylvester", "library"):
        raise HTTPException(
            status_code=400,
            detail="start must be 'sylvester' or 'library' (need a concrete H)",
        )
    if start == "sylvester":
        H = sylvester(order)
        if H is None:
            raise HTTPException(
                status_code=400,
                detail=f"sylvester needs a power-of-2 order, got {order}",
            )
    else:
        path = LIB_DIR / f"hadamard_{order}.csv"
        if not path.is_file():
            raise HTTPException(
                status_code=400, detail=f"order {order} not in library",
            )
        H = np.loadtxt(path, delimiter=",", dtype=np.int8)
    flux = micromag.flux_map(H)
    return _jsafe({
        "order": order,
        "start": start,
        "flux_tiles": micromag.flux_tiles(H),
        "flux_png_b64": base64.b64encode(heatmap_png(flux, 512)).decode("ascii"),
    })
