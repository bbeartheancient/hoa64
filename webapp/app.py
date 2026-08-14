"""FastAPI application factory and entry point for the hoa64 webapp.

`create_app()` wires the Hadamard API router under /api and serves the
build-step-free vanilla-JS frontend from `webapp/static/` at /.  CORS is
allow-all: the app binds to 127.0.0.1 by default and is a trusted-local
lab tool, same posture as `server.py`.  Run directly:

    python -m hoa64.webapp.app --port 8770

or via the CLI (`hoa64 webapp`), which lazy-imports `main` so the base
CLI keeps working without fastapi/uvicorn installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routes_antenna import router as antenna_router
from .routes_gen import router as gen_router
from .routes_hadamard import router as hadamard_router
from .routes_library import router as library_router
from .routes_hoa import router as hoa_router
from .routes_noise import router as noise_router
from .routes_palettes import router as palettes_router
from .routes_search import router as search_router, ws_router as search_ws_router
from .routes_sim import router as sim_router

STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="hoa64 webapp")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(hadamard_router)
    app.include_router(gen_router)
    app.include_router(library_router)
    app.include_router(search_router)
    app.include_router(sim_router)
    app.include_router(hoa_router)
    app.include_router(palettes_router)
    app.include_router(antenna_router)
    app.include_router(noise_router)
    app.include_router(search_ws_router)

    # Any /api/* request that matched no router would otherwise fall
    # through to the StaticFiles mount below, which is GET/HEAD-only and
    # answers POSTs with a misleading 405 ("launch failed: Method Not
    # Allowed" on a stale server). Answer with a precise JSON 404 instead.
    @app.api_route(
        "/api/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        include_in_schema=False,
    )
    def api_catch_all(path: str) -> None:
        raise HTTPException(status_code=404, detail=f"no such API endpoint: /api/{path}")

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def main(host: str = "127.0.0.1", port: int = 8770) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="hoa64 webapp (FastAPI)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    main(host=args.host, port=args.port)
