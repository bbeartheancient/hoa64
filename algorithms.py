"""Canonical algorithm catalogs shared by Search Studio, Matrix Lab, and Sim.

Search engines are the JobManager solvers registered in
``webapp.routes_search``.  Construct methods are the synchronous
builders behind ``POST /api/construct``.  Sim algorithms are the
engines the micromag field lab can run as a reheat segment — every
search engine that produces a live ±1 matrix.
"""
from __future__ import annotations

# Order is the UI order.  Keep these three lists in sync with
# webapp/static/js/algorithms.js.
SEARCH_ENGINES = (
    "maxdet",
    "micromag",
    "tile",
    "gerzon",
    "holographic",
    "crown",
    "brillouin",
    "williamson",
    "gs",
    "circulant",
)

CONSTRUCT_METHODS = (
    "auto",
    "sylvester",
    "paley",
    "miyamoto",
    "cw",
    "gcp",
    "row_builder",
)

# Same as SEARCH_ENGINES — the sim lab can run any of them.
SIM_ALGORITHMS = SEARCH_ENGINES


def catalog() -> dict:
    return {
        "search": list(SEARCH_ENGINES),
        "construct": list(CONSTRUCT_METHODS),
        "sim": list(SIM_ALGORITHMS),
    }
