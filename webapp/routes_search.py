"""Search API routes — live-streamed heuristic Hadamard search (Phase 2).

The long-running engines (max-det ILS, micromag SA, tile SA, Gerzon AB,
Williamson / Goethals-Seidel PSD descent, circulant PSD descent) run on the JobManager
thread pool (`jobs.py`); their progress callbacks feed `report()`, which
queues frames for the WebSocket at /ws/job/{job_id} and appends them to
`job.history` for mid-run replay.

Engine return signatures are heterogeneous, so each engine gets a small
adapter runner that returns `(H | None, info_dict)` uniformly; `_execute`
then verifies, normalizes, and stores the result.  The winning matrix is
kept on `job.matrix` (a plain attribute, never part of the JSON result) so
/api/search/{id} stays lightweight while /export can re-verify and write
the CSV without re-serializing a ±1 grid through JSON.

SA-mode budget: `micromag_sa` / `tile_sa_swap` / the PSD descents have no
`time_budget` kwarg, so the runner passes a `_BudgetStop` facade as
`stop_flag` — it reports set when the job is cancelled OR the wall-clock
budget is exhausted, without touching `job.cancel` (a budget-exhausted job
is "done", not "cancelled").

Live matrix previews: SA-mode engine callbacks carry the current best
matrix under the "H" key.  `_sa_reporter` pops it before `report()` (so
queues/history never retain arrays) and, for orders ≤ 512, re-renders a
PNG at most once every 2 s as a `matrix_png_b64` frame.
"""

from __future__ import annotations

import asyncio
import base64
import math
import queue
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..hadamard import check, normalize, verify
from ._png import matrix_png
from .jobs import JOBS, Job, report
from .routes_hadamard import _jsafe

LIB_DIR = Path.home() / "open_hadamard"
MAX_ORDER = 4000
DET_MAX = 500  # compute det_log10 only up to this order
PREVIEW_MAX = 512  # live matrix previews only up to this order
PREVIEW_EVERY = 2.0  # seconds between preview frames

router = APIRouter(prefix="/api")
ws_router = APIRouter()

_TERMINAL = {"done", "error", "cancelled"}


class _BudgetStop:
    """stop_flag facade: set when the job is cancelled OR budget_s elapsed.

    Passed to engines that lack a `time_budget` kwarg (SA single runs).
    Budget expiry does NOT set `job.cancel`, so the JobManager still marks
    the job "done" rather than "cancelled".
    """

    def __init__(self, job: Job):
        self._job = job
        self._t0 = time.monotonic()

    def is_set(self) -> bool:
        if self._job.cancel.is_set():
            return True
        budget = self._job.params.get("budget_s")
        return bool(budget) and (time.monotonic() - self._t0) >= budget


def _sa_reporter(job: Job, order: int):
    """Build the engine callback for SA-mode runs.

    Forwards stat frames to report(); pops the "H" conduit key and emits a
    throttled `matrix_png_b64` preview frame for orders ≤ PREVIEW_MAX.
    """
    engine = job.params["engine"]
    t0 = time.monotonic()
    last_preview = [t0 - PREVIEW_EVERY]  # first H paints immediately

    def cb(stats: dict) -> None:
        stats = dict(stats)
        H = stats.pop("H", None)
        stats.pop("elapsed_s", None)  # ILS iter frames carry their own; keep ours
        report(job, engine=engine, elapsed_s=round(time.monotonic() - t0, 3), **stats)
        now = time.monotonic()
        if (
            H is not None
            and order <= PREVIEW_MAX
            and now - last_preview[0] >= PREVIEW_EVERY
        ):
            last_preview[0] = now
            png = matrix_png(np.asarray(H, dtype=np.int8), 256)
            report(job, engine=engine, matrix_png_b64=base64.b64encode(png).decode("ascii"))

    return cb


# ---------------------------------------------------------------- runners
# Each runner takes the Job and returns (H | None, info_dict).

_ENGINES: dict[str, Any] = {}


def _register(name: str):
    def deco(fn):
        _ENGINES[name] = fn
        return fn

    return deco


def _reheat_sa(job: Job, order: int, rng, run_sa):
    """Cool → perturb → reheat until the job budget/cancel fires.

    One SA cool-down on a modest order finishes in well under a second, so
    Search Studio used to return "no Hadamard" before the matrix preview
    ever painted.  Same reheat chain as the micromag sim lab.
    """
    from ..hadamard import perturb

    stop = _BudgetStop(job)
    cb = _sa_reporter(job, order)
    cur = None
    best_H, best_info, best_E = None, None, float("inf")
    while not stop.is_set():
        H, info = run_sa(start=cur, callback=cb, stop_flag=stop, rng=rng)
        E = info.get("best_E")
        if E is None:
            E = info.get("f", float("inf"))
        if H is not None and E < best_E:
            best_H, best_E, best_info = H, E, info
        if info.get("hadamard") or (isinstance(E, (int, float)) and E < 1e-9):
            break
        if stop.is_set() or best_H is None:
            break
        cur = perturb(best_H, rng, frac=0.05)
    return best_H, {**(best_info or {}), "best_E": best_E}


@_register("maxdet")
def _run_maxdet(job: Job):
    from .. import hadamard as hd

    p = job.params
    rng = np.random.default_rng(p.get("seed"))
    # Search Studio is a *search* — do not start from a library/construction
    # matrix or order 64 (Sylvester) finishes in one local-search pass.
    # warm_start is for the selftest / CLI-style "known seed first" path.
    seeds = None if p.get("warm_start") else [hd.random_seed(p["order"], rng)]
    H, st = hd.ils_search(
        p["order"],
        seeds=seeds,
        time_budget=p["budget_s"],
        seed_int=p.get("seed"),
        print_progress=False,
        iter_callback=_sa_reporter(job, p["order"]),
        stop_flag=_BudgetStop(job),
    )
    if H is None:  # cancelled before the first iteration
        return None, {"best_f": None}
    return H, {"best_f": st.get("f"), "is_hadamard": st.get("is_hadamard")}


@_register("micromag")
def _run_micromag(job: Job):
    from .. import micromag

    p = job.params
    order = p["order"]
    live = p.setdefault("live", {})  # WS {"op":"set",...} retunes this mid-run
    rng = np.random.default_rng(p.get("seed"))
    stop = _BudgetStop(job)
    if p.get("mode") == "sa":
        def _once(*, start, callback, stop_flag, rng):
            return micromag.micromag_sa(
                order,
                T_start=p.get("T_start", 10.0),
                cooling=p.get("cooling", 0.999),
                lam_ex=p.get("lam_ex", 0.0),
                lam_ani=p.get("lam_ani", 0.0),
                max_steps=int(p.get("max_steps", 10**9)),
                callback=callback,
                stop_flag=stop_flag,
                live_params=live,
                rng=rng,
                start=start,
            )
        H, info = _reheat_sa(job, order, rng, _once)
        return H, {"best_E": info.get("best_E"), "info": info}
    H, best_f, ok = micromag.micromag_ils_robust(
        order, time_budget=p["budget_s"], stop_flag=stop, rng=rng,
        progress_callback=_sa_reporter(job, order),
    )
    return H, {"best_f": best_f, "is_hadamard": bool(ok)}


@_register("tile")
def _run_tile(job: Job):
    from .. import tile_search

    p = job.params
    order = p["order"]
    rng = np.random.default_rng(p.get("seed"))
    stop = _BudgetStop(job)
    if p.get("mode") == "sa":
        def _once(*, start, callback, stop_flag, rng):
            return tile_search.tile_sa_swap(
                order,
                T_start=p.get("T_start", 20.0),
                cooling=p.get("cooling", 0.9995),
                max_steps=int(p.get("max_steps", 10**9)),
                callback=callback,
                stop_flag=stop_flag,
                rng=rng,
                start=start,
            )
        H, info = _reheat_sa(job, order, rng, _once)
        return H, {"best_E": info.get("best_E"), "info": info}
    H, best_f, ok = tile_search.tile_ils(
        order, time_budget=p["budget_s"], stop_flag=stop, rng=rng,
        progress_callback=_sa_reporter(job, order),
    )
    return H, {"best_f": best_f, "is_hadamard": bool(ok)}


@_register("gerzon")
def _run_gerzon(job: Job):
    from .. import gerzon

    p = job.params
    order = p["order"]
    rng = np.random.default_rng(p.get("seed"))
    stop = _BudgetStop(job)
    lam_z = float(p.get("lam_z", 1.0))
    if p.get("mode") == "sa":
        def _once(*, start, callback, stop_flag, rng):
            return gerzon.gerzon_sa(
                order,
                T_start=p.get("T_start", 20.0),
                cooling=p.get("cooling", 0.9995),
                max_steps=int(p.get("max_steps", 10**9)),
                lam_z=lam_z,
                callback=callback,
                stop_flag=stop_flag,
                rng=rng,
                start=start,
            )
        H, info = _reheat_sa(job, order, rng, _once)
        return H, {"best_E": info.get("best_E"), "info": info}
    H, best_f, ok = gerzon.gerzon_ils(
        order, time_budget=p["budget_s"], lam_z=lam_z,
        stop_flag=stop, rng=rng,
        progress_callback=_sa_reporter(job, order),
    )
    return H, {"best_f": best_f, "is_hadamard": bool(ok)}


@_register("holographic")
def _run_holographic(job: Job):
    from .. import holographic as holo

    p = job.params
    order = p["order"]
    rng = np.random.default_rng(p.get("seed"))
    stop = _BudgetStop(job)
    lam_h = float(p.get("lam_h", 1.0))
    if p.get("mode") == "sa":
        def _once(*, start, callback, stop_flag, rng):
            return holo.holo_sa(
                order,
                T_start=p.get("T_start", 20.0),
                cooling=p.get("cooling", 0.9995),
                max_steps=int(p.get("max_steps", 10**9)),
                lam_h=lam_h,
                callback=callback,
                stop_flag=stop_flag,
                rng=rng,
                start=start,
            )
        H, info = _reheat_sa(job, order, rng, _once)
        return H, {"best_E": info.get("best_E"), "info": info}
    H, best_f, ok = holo.holo_ils(
        order, time_budget=p["budget_s"], lam_h=lam_h,
        stop_flag=stop, rng=rng,
        progress_callback=_sa_reporter(job, order),
    )
    return H, {"best_f": best_f, "is_hadamard": bool(ok)}


@_register("crown")
def _run_crown(job: Job):
    from .. import crown

    p = job.params
    order = p["order"]
    rng = np.random.default_rng(p.get("seed"))
    stop = _BudgetStop(job)
    lam_c = float(p.get("lam_c", 1.0))
    if p.get("mode") == "sa":
        def _once(*, start, callback, stop_flag, rng):
            return crown.crown_sa(
                order,
                T_start=p.get("T_start", 20.0),
                cooling=p.get("cooling", 0.9995),
                max_steps=int(p.get("max_steps", 10**9)),
                lam_c=lam_c,
                callback=callback,
                stop_flag=stop_flag,
                rng=rng,
                start=start,
            )
        H, info = _reheat_sa(job, order, rng, _once)
        return H, {"best_E": info.get("best_E"), "info": info}
    H, best_f, ok = crown.crown_ils(
        order, time_budget=p["budget_s"], lam_c=lam_c,
        stop_flag=stop, rng=rng,
        progress_callback=_sa_reporter(job, order),
    )
    return H, {"best_f": best_f, "is_hadamard": bool(ok)}


@_register("brillouin")
def _run_brillouin(job: Job):
    from .. import brillouin

    p = job.params
    order = p["order"]
    rng = np.random.default_rng(p.get("seed"))
    stop = _BudgetStop(job)
    lam_b = float(p.get("lam_b", 1.0))
    if p.get("mode") == "sa":
        def _once(*, start, callback, stop_flag, rng):
            return brillouin.bzf_sa(
                order,
                T_start=p.get("T_start", 20.0),
                cooling=p.get("cooling", 0.9995),
                max_steps=int(p.get("max_steps", 10**9)),
                lam_b=lam_b,
                callback=callback,
                stop_flag=stop_flag,
                rng=rng,
                start=start,
            )
        H, info = _reheat_sa(job, order, rng, _once)
        return H, {"best_E": info.get("best_E"), "info": info}
    H, best_f, ok = brillouin.bzf_ils(
        order, time_budget=p["budget_s"], lam_b=lam_b,
        stop_flag=stop, rng=rng,
        progress_callback=_sa_reporter(job, order),
    )
    return H, {"best_f": best_f, "is_hadamard": bool(ok)}


@_register("sudoku")
def _run_sudoku(job: Job):
    from .. import sudoku

    p = job.params
    order = p["order"]
    rng = np.random.default_rng(p.get("seed"))
    stop = _BudgetStop(job)
    method = p.get("method")
    if p.get("mode") == "sa":
        def _once(*, start, callback, stop_flag, rng):
            return sudoku.sudoku_sa(
                order,
                T_start=p.get("T_start", 20.0),
                cooling=p.get("cooling", 0.9995),
                max_steps=int(p.get("max_steps", 10**9)),
                method=method or "stochastic",
                callback=callback,
                stop_flag=stop_flag,
                rng=rng,
                start=start,
            )
        H, info = _reheat_sa(job, order, rng, _once)
        return H, {"best_E": info.get("best_E"), "info": info}
    H, best_f, ok = sudoku.sudoku_ils(
        order, time_budget=p["budget_s"], method=method,
        stop_flag=stop, rng=rng,
        progress_callback=_sa_reporter(job, order),
    )
    return H, {"best_f": best_f, "is_hadamard": bool(ok)}


def _run_quad(job: Job, search, ils, to_hadamard):
    """Shared runner for the Williamson / Goethals-Seidel engines."""
    p = job.params
    order = p["order"]
    k = order // 4
    rng = np.random.default_rng(p.get("seed"))
    stop = _BudgetStop(job)
    if p.get("mode") == "sa":
        a, b, c, d, st = search(
            k,
            max_flips=int(p.get("max_flips", 10**9)),
            callback=_sa_reporter(job, order),
            stop_flag=stop,
            rng=rng,
        )
        H, method = to_hadamard(k, a, b, c, d)
        return H, {"best_f": st.get("f"), "method": method}
    H, method, best_f, ok = ils(k, time_budget=p["budget_s"], stop_flag=stop, rng=rng,
                                progress_callback=_sa_reporter(job, order))
    return H, {"best_f": best_f, "method": method, "is_hadamard": bool(ok)}


@_register("williamson")
def _run_williamson(job: Job):
    from .. import williamson as wm

    return _run_quad(job, wm.williamson_search, wm.williamson_ils,
                     wm.williamson_to_hadamard)


@_register("gs")
def _run_gs(job: Job):
    from .. import williamson as wm

    return _run_quad(job, wm.gs_circulant_search, wm.gs_circulant_ils,
                     wm.williamson_to_hadamard)


@_register("circulant")
def _run_circulant(job: Job):
    from .. import circulant_search as cs

    p = job.params
    order = p["order"]
    rng = np.random.default_rng(p.get("seed"))
    stop = _BudgetStop(job)
    if p.get("mode") == "sa":
        best_a, best_f, _, _ = cs.psd_search(
            order,
            max_flips=int(p.get("max_flips", 10**9)),
            callback=_sa_reporter(job, order),
            stop_flag=stop,
            rng=rng,
        )
        H = cs.circulant_matrix(np.round(best_a).astype(np.int8))
        return H, {"best_f": best_f}
    H, best_f, ok = cs.search_ils(
        order, time_budget=p["budget_s"], stop_flag=stop, rng=rng,
        progress_callback=_sa_reporter(job, order),
    )
    return H, {"best_f": best_f, "is_hadamard": bool(ok)}


def _package_result(job: Job, H, info: dict, label: str) -> dict:
    """Shared result packaging for search/sim runners: verify, normalize,
    keep the matrix on `job.matrix` (off-JSON), return the result dict."""
    if H is not None and verify(H):
        H = normalize(np.asarray(H, dtype=np.int8))
        job.matrix = H  # kept off the JSON result; /export re-reads it
        n = int(H.shape[0])
        from ..micromag import flux_tiles
        gz = None
        ho = None
        cr = None
        try:
            from ..gerzon import analyze as gerzon_analyze
            gz = gerzon_analyze(H)
            gz.pop("Z_wall", None)
        except Exception:
            gz = None
        try:
            from ..holographic import analyze as holo_analyze
            ho = holo_analyze(H)
        except Exception:
            ho = None
        try:
            from ..crown import analyze as crown_analyze
            cr = crown_analyze(H)
        except Exception:
            cr = None
        bz = None
        try:
            from ..brillouin import analyze as bzf_analyze
            bz = bzf_analyze(H)
        except Exception:
            bz = None
        su = None
        try:
            from ..sudoku import analyze as sudoku_analyze
            su = sudoku_analyze(H)
        except Exception:
            su = None
        return {
            "ok": True,
            "order": n,
            "engine": label,
            "stats": check(H, det=n <= DET_MAX),
            "flux_tiles": flux_tiles(H),
            "gerzon": gz,
            "holographic": ho,
            "crown": cr,
            "brillouin": bz,
            "sudoku": su,
            "png_b64": base64.b64encode(matrix_png(H, 512)).decode("ascii"),
        }
    return {"ok": False, **_jsafe(info)}


def _execute(job: Job) -> dict:
    """JobManager entry point: run the engine, verify, package the result."""
    engine = job.params["engine"]
    H, info = _ENGINES[engine](job)
    return _package_result(job, H, info, engine)


# ---------------------------------------------------------------- rest

class SearchReq(BaseModel):
    engine: str
    order: int
    budget_s: float = 30.0
    mode: str = "ils"
    seed: int | None = None
    params: dict[str, Any] = {}


def _validate(engine: str, order: int) -> str | None:
    if not (1 <= order <= MAX_ORDER) or (order not in (1, 2) and order % 4 != 0):
        return f"order must be 1, 2 or a multiple of 4, ≤ {MAX_ORDER}"
    if engine in ("williamson", "gs") and order % 4 != 0:
        return f"{engine} search needs order = 4k"
    if engine == "circulant":
        u = math.isqrt(order // 4) if order % 4 == 0 else 0
        if order % 4 != 0 or u * u != order // 4:
            return "circulant search needs order = 4u² (order//4 a perfect square)"
    return None


@router.get("/algorithms")
def algorithms_get() -> dict:
    """Search / construct / sim algorithm catalogs (UI dropdowns)."""
    from ..algorithms import catalog
    return catalog()


@router.post("/search")
def search_start(req: SearchReq) -> dict:
    if req.engine not in _ENGINES:
        raise HTTPException(
            status_code=400,
            detail=f"unknown engine {req.engine!r} (have: {sorted(_ENGINES)})",
        )
    if req.mode not in ("ils", "sa"):
        raise HTTPException(status_code=400, detail="mode must be 'ils' or 'sa'")
    err = _validate(req.engine, req.order)
    if err:
        raise HTTPException(status_code=400, detail=err)
    params = {
        "engine": req.engine,
        "order": req.order,
        "budget_s": float(req.budget_s),
        "mode": req.mode,
        **req.params,
    }
    if req.seed is not None:
        params["seed"] = req.seed
    job = JOBS.submit("search", _execute, params)
    return {"job_id": job.id}


def _job_brief(job: Job) -> dict:
    return {
        "id": job.id,
        "kind": job.kind,
        "status": job.status,
        "created": job.created,
        "params": {k: v for k, v in job.params.items() if k != "live"},
    }


@router.get("/search")
def search_list() -> dict:
    return _jsafe({"jobs": [_job_brief(j) for j in JOBS.list()[:50]]})


@router.get("/search/{job_id}")
def search_status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
    return _jsafe(
        {
            **_job_brief(job),
            "finished": job.finished,
            "error": job.error,
            "result": job.result,
        }
    )


@router.post("/search/{job_id}/cancel")
def search_cancel(job_id: str) -> dict:
    if JOBS.get(job_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
    return {"cancelled": JOBS.cancel(job_id)}


def _write_matrix_csv(job: Job) -> Path:
    """Shared export: re-verify `job.matrix` and write it to the library.

    Kind-agnostic — works for any job whose runner produced a verified
    matrix ("search" and "micromag_sim" alike).  Raises HTTPException.
    """
    if job.status != "done" or not isinstance(job.result, dict) or not job.result.get("ok"):
        raise HTTPException(status_code=400, detail="job has no verified Hadamard result")
    H = job.matrix
    if H is None or not verify(H):
        raise HTTPException(status_code=500, detail="stored matrix failed re-verification")
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    path = LIB_DIR / f"hadamard_{int(H.shape[0])}.csv"
    if path.exists():
        raise HTTPException(status_code=409, detail=f"{path} already exists")
    np.savetxt(path, np.asarray(H, dtype=np.int8), delimiter=",", fmt="%d")
    return path


@router.post("/search/{job_id}/export")
def search_export(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
    return {"path": str(_write_matrix_csv(job))}


# ---------------------------------------------------------------- websocket

@ws_router.websocket("/ws/job/{job_id}")
async def job_ws(ws: WebSocket, job_id: str) -> None:
    await ws.accept()
    job = JOBS.get(job_id)
    if job is None:
        await ws.send_json({"type": "error", "error": f"unknown job {job_id!r}"})
        await ws.close()
        return

    # Drain queued frames already covered by the history replay (favor a
    # possible duplicate over a dropped frame), then send the snapshot.
    while True:
        try:
            job.progress.get_nowait()
        except queue.Empty:
            break
    await ws.send_json(
        _jsafe({"type": "snapshot", "status": job.status, "history": list(job.history)})
    )

    async def pump() -> None:
        while True:
            try:
                item = await asyncio.to_thread(job.progress.get, True, 0.5)
            except queue.Empty:
                if job.status in _TERMINAL and job.progress.empty():
                    await ws.send_json({"type": "end", "status": job.status})
                    return
                continue
            await ws.send_json(_jsafe(item))
            if item.get("type") == "end":
                return

    async def listen() -> None:
        while True:
            msg = await ws.receive_json()
            if not isinstance(msg, dict):
                continue
            op = msg.get("op")
            if op == "cancel":
                JOBS.cancel(job.id)
            elif op == "set":
                live = job.params.setdefault("live", {})
                for key in ("cooling", "lam_ex", "lam_ani", "lam_goal",
                            "lam_tile", "lam_z", "lam_h"):
                    if msg.get(key) is not None:
                        live[key] = float(msg[key])

    pump_task = asyncio.create_task(pump())
    listen_task = asyncio.create_task(listen())
    try:
        await asyncio.wait(
            [pump_task, listen_task], return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        for t in (pump_task, listen_task):
            t.cancel()
        # Client disconnects never cancel the job — it keeps running.
        try:
            await ws.close()
        except Exception:  # already disconnected
            pass
