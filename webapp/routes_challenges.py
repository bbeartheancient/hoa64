"""DARPA-challenges API route — the lab's high-level program frame.

Wraps `darpa_challenges` (the 23 DARPA mathematical challenges + this
package's honest tooling alignment):

* ``GET /api/challenges`` — the 23 challenges, the per-challenge
  alignment (status / engines / note), and a summary count.  Static
  data; no jobs, no caching needed.
"""

from __future__ import annotations

from fastapi import APIRouter

from .. import darpa_challenges
from .routes_hadamard import _jsafe

router = APIRouter(prefix="/api/challenges")


@router.get("")
def challenges() -> dict:
    return _jsafe({
        "challenges": darpa_challenges.CHALLENGES,
        "alignment": darpa_challenges.ALIGNMENT,
        "summary": darpa_challenges.summary(),
    })
