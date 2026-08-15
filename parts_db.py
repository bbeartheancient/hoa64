"""Antenna parts database — loader and matcher for the curated catalog in
``webapp/data/antenna_parts.json``.

Why a local database
--------------------
everythingRF (https://www.everythingrf.com/search/all-antennas) is the
reference catalog for off-the-shelf antennas, but it exposes no public
API and blocks automated access.  Rather than scrape it, the package
ships a small, community-editable JSON database whose row fields mirror
the everythingRF antenna listing columns (frequency range, gain,
polarization, VSWR, size, mount).  Each row carries ``erf_url`` — the
everythingRF antenna search entry point; open it and paste the part
number into the search box to cross-check against the full catalog.

Row schema (all fields required)
--------------------------------
part                          manufacturer part number (dual-band parts
                              appear once per band)
mfr                           manufacturer name
type                          chip | pcb | whip | dipole | patch | flex |
                              helical | loop
freq_lo_mhz / freq_hi_mhz     operating band edges
gain_dbi                      nominal peak gain (dBic for circular patches)
polarization                  linear | RHCP | circular
vswr                          max VSWR across the band
size_mm                       [L, W, H] in millimetres
mount                         smd | u.fl | sma | adhesive | through-hole
medium                        propagation media, normally ["air"]
datasheet_url                 manufacturer or distributor lookup URL
erf_url                       everythingRF antenna search entry point

Electrical values are nominal catalog numbers — always confirm against
the manufacturer datasheet before committing a design.

Matching
--------
``match(spec)`` filters by the spec keys ``gain_dbi_min``,
``polarization``, ``max_size_mm`` (largest linear dimension), ``mount``
and ``type``, then applies the frequency gate: a part qualifies only if
its range covers ``[f_lo_hz, f_hi_hz]`` — or, with ``partial=True`` in
the spec, merely overlaps it (such rows are penalized and carry a
``coverage_note``).  Remaining parts are scored by closeness: least
excess gain above the requirement and most compact footprint win.
Each result row is the part record plus ``score``, ``margin_db``
(part gain minus ``gain_dbi_min``, ``None`` when no gain was specified)
and, for partial matches, ``coverage_note``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from os import PathLike
from pathlib import Path

_DB_PATH = Path(__file__).resolve().parent / "webapp" / "data" / "antenna_parts.json"


@lru_cache(maxsize=4)
def load_db(path: str | PathLike | None = None) -> dict:
    """Load the antenna parts database.

    ``path`` defaults to the packaged ``webapp/data/antenna_parts.json``.
    Results are cached; pass a path to load an edited copy.
    """
    p = Path(path) if path is not None else _DB_PATH
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def match(spec: dict, db: dict | None = None, limit: int = 10) -> list[dict]:
    """Find catalog parts fitting ``spec`` (all keys optional).

    Spec keys: ``f_lo_hz``/``f_hi_hz`` (band, give both or neither),
    ``partial`` (allow overlapping instead of covering parts),
    ``gain_dbi_min``, ``polarization`` ("circular" matches RHCP),
    ``max_size_mm`` (largest linear dimension), ``mount``, ``type``.
    Returns up to ``limit`` part records sorted by descending ``score``.
    """
    db = db if db is not None else load_db()
    f_lo = spec.get("f_lo_hz")
    f_hi = spec.get("f_hi_hz")
    if (f_lo is None) != (f_hi is None):
        raise ValueError("spec must give both f_lo_hz and f_hi_hz, or neither")
    partial = bool(spec.get("partial"))
    gain_min = spec.get("gain_dbi_min")
    pol = spec.get("polarization")
    mount = spec.get("mount")
    ptype = spec.get("type")
    max_size = spec.get("max_size_mm")

    out: list[dict] = []
    for p in db["parts"]:
        if gain_min is not None and p["gain_dbi"] < gain_min:
            continue
        if pol is not None:
            if pol == "circular":
                if p["polarization"] not in ("circular", "RHCP", "LHCP"):
                    continue
            elif p["polarization"] != pol:
                continue
        if mount is not None and p["mount"] != mount:
            continue
        if ptype is not None and p["type"] != ptype:
            continue
        maxdim = max(p["size_mm"])
        if max_size is not None and maxdim > max_size:
            continue

        note = None
        if f_lo is not None:
            lo = p["freq_lo_mhz"] * 1e6
            hi = p["freq_hi_mhz"] * 1e6
            if lo <= f_lo and hi >= f_hi:
                pass  # full coverage
            elif partial and lo <= f_hi and hi >= f_lo:
                note = (f"partial overlap: part covers {p['freq_lo_mhz']:g}–"
                        f"{p['freq_hi_mhz']:g} MHz, spec "
                        f"{f_lo / 1e6:g}–{f_hi / 1e6:g} MHz")
            else:
                continue

        margin = None if gain_min is None else p["gain_dbi"] - gain_min
        score = 100.0
        if margin is not None:
            score -= 2.0 * margin      # least excess gain above the ask wins
        score -= 0.05 * maxdim         # weak preference for compact parts
        if note is not None:
            score -= 25.0              # partial coverage penalty

        row = dict(p)
        row["score"] = round(score, 3)
        row["margin_db"] = margin
        if note is not None:
            row["coverage_note"] = note
        out.append(row)

    out.sort(key=lambda r: (-r["score"], r["part"], r["freq_lo_mhz"]))
    return out[:limit]


if __name__ == "__main__":
    db = load_db()
    parts = db["parts"]
    REQUIRED = ("part", "mfr", "type", "freq_lo_mhz", "freq_hi_mhz",
                "gain_dbi", "polarization", "vswr", "size_mm", "mount",
                "medium", "datasheet_url", "erf_url")
    TYPES = {"chip", "pcb", "whip", "dipole", "patch", "flex", "helical", "loop"}
    MOUNTS = {"smd", "u.fl", "sma", "adhesive", "through-hole"}
    POLS = {"linear", "RHCP", "circular"}
    assert len(parts) >= 20, f"only {len(parts)} rows"
    for p in parts:
        for k in REQUIRED:
            assert k in p, (k, p.get("part"))
        assert isinstance(p["part"], str) and p["part"]
        assert isinstance(p["mfr"], str) and p["mfr"]
        assert p["type"] in TYPES, p["part"]
        assert p["mount"] in MOUNTS, p["part"]
        assert p["polarization"] in POLS, p["part"]
        assert 0 < p["freq_lo_mhz"] < p["freq_hi_mhz"], p["part"]
        assert isinstance(p["gain_dbi"], (int, float)), p["part"]
        assert len(p["size_mm"]) == 3 and all(
            isinstance(v, (int, float)) and v > 0 for v in p["size_mm"]), p["part"]
        assert isinstance(p["medium"], list) and p["medium"], p["part"]

    # 2.4 GHz ISM band, gain ≥ 0 dBi
    r = match({"f_lo_hz": 2400e6, "f_hi_hz": 2485e6, "gain_dbi_min": 0.0})
    assert len(r) >= 4, f"only {len(r)} 2.4 GHz matches"
    assert all(x["freq_lo_mhz"] <= 2400 and x["freq_hi_mhz"] >= 2485 for x in r)
    assert all(x["gain_dbi"] >= 0.0 for x in r)
    assert all(r[i]["score"] >= r[i + 1]["score"] for i in range(len(r) - 1))

    # GNSS L1 1575.42 ± 5 MHz, circular — patch band is narrower than the
    # spec window, so allow partial overlaps and demand an RHCP result.
    g = match({"f_lo_hz": 1570.42e6, "f_hi_hz": 1580.42e6,
               "polarization": "circular", "partial": True})
    assert len(g) >= 1 and any(x["polarization"] == "RHCP" for x in g), g

    # wideband mmWave: no catalog parts — empty list, not an error.
    assert match({"f_lo_hz": 28e9, "f_hi_hz": 29e9}) == []

    print(f"parts_db self-check OK: {len(parts)} rows, "
          f"{len(r)} parts cover 2400–2485 MHz @ gain ≥ 0 dBi, "
          f"GNSS circular match: {g[0]['part']} ({g[0]['mfr']})")
