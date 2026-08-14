"""hoa64 webapp — FastAPI + vanilla-JS GUI for the Hadamard lab (Phase 1).

Phase 1 exposes the construction/verification side of the toolchain:
classical constructions (Sylvester, Paley, Miyamoto, Cooper-Wallis, GCP,
row-builder), the ~/open_hadamard CSV library, and the check/verify
statistics — wrapped in JSON endpoints plus a dark, green-on-black
"Matrix Lab" frontend.  Phase 2 will attach the long-running search
engines (micromag SA, Williamson/GS PSD minimization) through the
JobManager + WebSocket plumbing already stubbed in `jobs.py` / `ws.js`.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .app import create_app

__all__ = ["create_app", "__version__"]
