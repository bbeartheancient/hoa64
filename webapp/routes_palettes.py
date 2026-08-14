"""Game Boy palette packs — Phase 7: `GET /api/palettes`.

Walks `Assets/gb/common/Palettes/` (the user's emulator palette library,
subfolders = categories, possibly nested) and parses every `.pal` file.

File format (verified against all 699 shipped files): binary, exactly
56 bytes — the 4-color RGB24 palette (12 bytes, dark→light order NOT
guaranteed) repeated for the SGB/BIOS slots, terminated by a `\\x81APGB`
magic.  Only the first 12 bytes matter here → four `#rrggbb` colors.

The payload is cached for 5 minutes (the library is static in practice).
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter

router = APIRouter()

PALETTES_DIR = (
    Path(__file__).resolve().parent.parent / "Assets" / "gb" / "common" / "Palettes"
)
CACHE_TTL_S = 300  # 5 min
_TRAILER = b"\x81APGB"

_cache: tuple[float, dict] | None = None


def _parse_pal(path: Path) -> list[str] | None:
    """First-slot RGB24 colors of a .pal file → 4 `#rrggbb` hexes (or None)."""
    try:
        b = path.read_bytes()
    except OSError:
        return None
    if len(b) < 17 or not b.endswith(_TRAILER):
        return None
    return [f"#{b[i]:02x}{b[i + 1]:02x}{b[i + 2]:02x}" for i in range(0, 12, 3)]


def _scan() -> dict:
    palettes = []
    if PALETTES_DIR.is_dir():
        for path in sorted(PALETTES_DIR.rglob("*.pal")):
            colors = _parse_pal(path)
            if not colors:
                continue
            category = str(path.parent.relative_to(PALETTES_DIR)) or "."
            palettes.append(
                {"category": category, "name": path.stem, "colors": colors}
            )
    return {"palettes": palettes, "count": len(palettes)}


@router.get("/api/palettes")
async def list_palettes() -> dict:
    global _cache
    now = time.monotonic()
    if _cache is None or now - _cache[0] > CACHE_TTL_S:
        _cache = (now, _scan())
    return _cache[1]


if __name__ == "__main__":
    d = _scan()
    print(f"{d['count']} palettes")
    for p in d["palettes"][:5]:
        print(f"  {p['category']}/{p['name']}: {' '.join(p['colors'])}")
