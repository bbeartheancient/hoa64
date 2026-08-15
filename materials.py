"""Materials lab — physical realisations of the H.8 flux-tile catalog.

Three homes for the 4-tile Walsh wall pattern of Sylvester / A ⊗ H₈
(see `micromag.flux_tiles`).  The catalog is not a named object in the
Hadamard literature; these are concrete layouts that *use* it.

The working field is the **flux polarity**, not the Hadamard signs:

    P = 2W − 1 ∈ {−1, 0, +1}

W = 0 (aligned, no wall) → P = −1, W = ½ (one wall) → P = 0,
W = 1 (corner, two walls) → P = +1.  That is the ternary tile
visible on the Micromag FLUX layer.

**Cloth / knit** (`cloth`).  Three yarn nets on an open sheet (no
toroidal wrap).  P = +1 is face conductor (F.Cu), P = −1 is reverse
(B.Cu), P = 0 is the mid / ground yarn.  A run breaks where P
changes.  Connected components of each polarity are electrodes.

**Capacitive touchpad** (`touchpad`).  Same ternary copper.  Every
bond between unlike P is a mutual-cap.  Three electrode families
(+/0/−); the H.8 tile is one balanced sensor cell.

**Metamaterial / spin-ice** (`metamaterial`).  The full n×n flux
sheet (so the preview *is* the tile, not H) plus a callout of the
H.8 atom and the Walsh lattice of the four tiles.  P = +1 vertices,
P = 0 edges, P = −1 bulk.

Pitch is millimetres per matrix cell.  Layouts use the same
``{rects, pads, vias, bbox}`` schema as `kicad_gen.footprint_from_layout`.
"""

from __future__ import annotations

import numpy as np

from .hadamard import sylvester, normalize
from .micromag import flux_map, flux_tiles

KINDS = ("cloth", "touchpad", "metamaterial")
STARTS = ("sylvester", "library")


def load_H(order: int, start: str = "sylvester",
           lib_dir=None) -> np.ndarray:
    """Resolve a concrete ±1 matrix for the materials lab."""
    order = int(order)
    if start == "sylvester":
        H = sylvester(order)
        if H is None:
            raise ValueError(f"sylvester needs a power-of-2 order, got {order}")
        return np.asarray(H, dtype=np.int8)
    if start == "library":
        from pathlib import Path
        d = Path(lib_dir) if lib_dir else Path.home() / "open_hadamard"
        path = d / f"hadamard_{order}.csv"
        if not path.is_file():
            raise ValueError(f"order {order} not in library ({path})")
        return np.loadtxt(path, delimiter=",", dtype=np.int8)
    raise ValueError(f"unknown start {start!r}; expected {STARTS}")


def flux_polarity(H) -> np.ndarray:
    """Ternary flux field P = 2W − 1 ∈ {−1, 0, +1}.

    W is `flux_map` (0 / ½ / 1).  Thresholds, not arithmetic, so the
    ½-level stays exactly 0 under float rounding.
    """
    W = flux_map(H)
    return np.where(W >= 0.75, np.int8(1),
                    np.where(W >= 0.25, np.int8(0), np.int8(-1)))


def _layer_of(p: int) -> str:
    if p > 0:
        return "F.Cu"
    if p < 0:
        return "B.Cu"
    return "F.SilkS"


# --- connected components (open sheet, no wrap) ------------------------------


def _components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected labels on a boolean mask.  −1 = background."""
    n = mask.shape[0]
    lab = np.full((n, n), -1, dtype=np.int32)
    nid = 0
    for i in range(n):
        for j in range(n):
            if not mask[i, j] or lab[i, j] >= 0:
                continue
            stack = [(i, j)]
            lab[i, j] = nid
            while stack:
                x, y = stack.pop()
                for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                    xx, yy = x + dx, y + dy
                    if (0 <= xx < n and 0 <= yy < n
                            and mask[xx, yy] and lab[xx, yy] < 0):
                        lab[xx, yy] = nid
                        stack.append((xx, yy))
            nid += 1
    return lab, nid


def _runs(row: np.ndarray) -> list[tuple[int, int, int]]:
    """Maximal constant-sign runs: (start, end_inclusive, sign)."""
    out = []
    s = 0
    for k in range(1, len(row) + 1):
        if k == len(row) or row[k] != row[s]:
            out.append((s, k - 1, int(row[s])))
            s = k
    return out


def _bbox_of(rects, pads) -> dict:
    xs, ys = [], []
    for r in rects:
        xs += [r["x"] - r["w"] / 2, r["x"] + r["w"] / 2]
        ys += [r["y"] - r["h"] / 2, r["y"] + r["h"] / 2]
    for p in pads:
        xs += [p["x"] - p["w"] / 2, p["x"] + p["w"] / 2]
        ys += [p["y"] - p["h"] / 2, p["y"] + p["h"] / 2]
    if not xs:
        xs = ys = [0.0]
    return {"xmin": min(xs), "xmax": max(xs),
            "ymin": min(ys), "ymax": max(ys)}


def preview_from_layout(layout: dict) -> dict:
    """Same prims schema as kicad_gen.preview_from_sexpr."""
    prims = []
    for r in layout.get("rects") or []:
        prims.append({
            "kind": "rect",
            "a": [r["x"] - r["w"] / 2, r["y"] - r["h"] / 2],
            "b": [r["x"] + r["w"] / 2, r["y"] + r["h"] / 2],
            "layer": r.get("layer", "F.Cu"),
            "polarity": r.get("polarity"),
        })
    for p in layout.get("pads") or []:
        prims.append({
            "kind": "pad", "name": p.get("name", ""),
            "c": [p["x"], p["y"]], "size": [p["w"], p["h"]],
            "layer": p.get("layer", "F.Cu"),
        })
    for v in layout.get("vias") or []:
        prims.append({
            "kind": "circle", "c": [v["x"], v["y"]],
            "r": float(v["d"]) / 2.0, "w": 0.15, "layer": "F.Cu",
        })
    return {"prims": prims, "bbox": layout["bbox"]}


def _sheet(P: np.ndarray, pitch: float, gap_frac: float):
    """n×n ternary cells → rects.  P ∈ {−1, 0, +1}."""
    n = P.shape[0]
    cell = pitch * (1.0 - float(gap_frac))
    origin = -0.5 * (n - 1) * pitch
    rects = []
    for i in range(n):
        for j in range(n):
            p = int(P[i, j])
            rects.append({
                "x": origin + j * pitch,
                "y": origin + (n - 1 - i) * pitch,
                "w": cell, "h": cell,
                "layer": _layer_of(p),
                "polarity": p,
            })
    return rects, origin, cell


def _polarity_stats(P: np.ndarray) -> dict:
    n = P.shape[0]
    plus_lab, n_plus = _components(P == 1)
    zero_lab, n_zero = _components(P == 0)
    minus_lab, n_minus = _components(P == -1)
    unlike = int(np.sum(P[:, :-1] != P[:, 1:])) + int(np.sum(P[:-1, :] != P[1:, :]))
    return {
        "n_plus": int(n_plus),
        "n_zero": int(n_zero),
        "n_minus": int(n_minus),
        "sites_plus": int(np.sum(P == 1)),
        "sites_zero": int(np.sum(P == 0)),
        "sites_minus": int(np.sum(P == -1)),
        "unlike_bonds": unlike,
        "warp_runs": sum(len(_runs(P[:, j])) for j in range(n)),
        "weft_runs": sum(len(_runs(P[i])) for i in range(n)),
        "plus_lab": plus_lab, "zero_lab": zero_lab, "minus_lab": minus_lab,
    }


# --- cloth -------------------------------------------------------------------


def cloth(H, pitch_mm: float = 1.0, gap_frac: float = 0.18) -> dict:
    """Ternary cloth from flux polarity P = 2W−1, not from H."""
    P = flux_polarity(H)
    n = P.shape[0]
    pitch = float(pitch_mm)
    rects, origin, cell = _sheet(P, pitch, gap_frac)
    st = _polarity_stats(P)
    pads = []
    for name, mask, layer in (
        ("+", P[:, 0] == 1, "F.Cu"),
        ("0", P[:, 0] == 0, "F.SilkS"),
        ("-", P[:, 0] == -1, "B.Cu"),
    ):
        hits = np.argwhere(mask)
        if len(hits):
            i = int(hits[0, 0])
            pads.append({"name": name, "x": origin - pitch,
                         "y": origin + (n - 1 - i) * pitch,
                         "w": cell, "h": cell, "layer": layer})
    layout = {"rects": rects, "pads": pads, "vias": [],
              "bbox": _bbox_of(rects, pads), "kind": "cloth"}
    stats = {
        "kind": "cloth", "n": n, "pitch_mm": pitch,
        "n_plus": st["n_plus"], "n_zero": st["n_zero"], "n_minus": st["n_minus"],
        "sites_plus": st["sites_plus"], "sites_zero": st["sites_zero"],
        "sites_minus": st["sites_minus"],
        "warp_runs": st["warp_runs"], "weft_runs": st["weft_runs"],
        "unlike_bonds": st["unlike_bonds"],
        "mean_w": float(flux_map(H).mean()),
        "fill": 1.0 - gap_frac,
        "field": "flux P=2W-1",
    }
    return {"layout": layout, "stats": stats, "preview": preview_from_layout(layout)}


# --- touchpad ----------------------------------------------------------------


def touchpad(H, pitch_mm: float = 2.0, gap_frac: float = 0.2) -> dict:
    """Mutual-cap pad on the ternary flux field.  Unlike-P bonds are caps."""
    P = flux_polarity(H)
    n = P.shape[0]
    pitch = float(pitch_mm)
    rects, origin, cell = _sheet(P, pitch, gap_frac)
    st = _polarity_stats(P)
    labs = {1: st["plus_lab"], 0: st["zero_lab"], -1: st["minus_lab"]}
    caps = []
    for i in range(n):
        for j in range(n - 1):
            if P[i, j] == P[i, j + 1]:
                continue
            caps.append({"a": int(P[i, j]), "b": int(P[i, j + 1]),
                         "ia": int(labs[int(P[i, j])][i, j]),
                         "ib": int(labs[int(P[i, j + 1])][i, j + 1]),
                         "dir": "h"})
    for i in range(n - 1):
        for j in range(n):
            if P[i, j] == P[i + 1, j]:
                continue
            caps.append({"a": int(P[i, j]), "b": int(P[i + 1, j]),
                         "ia": int(labs[int(P[i, j])][i, j]),
                         "ib": int(labs[int(P[i + 1, j])][i + 1, j]),
                         "dir": "v"})
    pads = []
    for pval, lab, n_el, prefix in (
        (1, st["plus_lab"], st["n_plus"], "P"),
        (0, st["zero_lab"], st["n_zero"], "Z"),
        (-1, st["minus_lab"], st["n_minus"], "M"),
    ):
        for eid in range(n_el):
            cells = np.argwhere(lab == eid)
            cy, cx = cells.mean(axis=0)
            pads.append({
                "name": f"{prefix}{eid}",
                "x": origin + float(cx) * pitch,
                "y": origin + (n - 1 - float(cy)) * pitch,
                "w": cell * 0.6, "h": cell * 0.6,
                "layer": _layer_of(pval),
            })
    layout = {"rects": rects, "pads": pads, "vias": [],
              "bbox": _bbox_of(rects, pads), "kind": "touchpad"}
    n_el = st["n_plus"] + st["n_zero"] + st["n_minus"]
    stats = {
        "kind": "touchpad", "n": n, "pitch_mm": pitch,
        "n_electrodes": n_el,
        "n_plus": st["n_plus"], "n_zero": st["n_zero"], "n_minus": st["n_minus"],
        "sites_plus": st["sites_plus"], "sites_zero": st["sites_zero"],
        "sites_minus": st["sites_minus"],
        "n_caps": len(caps),
        "caps_per_cell": len(caps) / max(n * n, 1),
        "mean_w": float(flux_map(H).mean()),
        "field": "flux P=2W-1",
    }
    return {"layout": layout, "stats": stats,
            "preview": preview_from_layout(layout),
            "caps": caps[:64]}


# --- metamaterial ------------------------------------------------------------


def metamaterial(H, pitch_mm: float = 2.0, gap_frac: float = 0.2) -> dict:
    """Full ternary flux sheet + H.8 atom callout + Walsh tile lattice."""
    P = flux_polarity(H)
    n = P.shape[0]
    pitch = float(pitch_mm)
    atom = 8 if n % 8 == 0 else (4 if n % 4 == 0 else n)
    W = flux_map(H)
    tiles = flux_tiles(H, tile=atom)
    catalog = {}
    grid = np.full((n // atom, n // atom), -1, dtype=np.int32)
    for bi, i in enumerate(range(0, n, atom)):
        for bj, j in enumerate(range(0, n, atom)):
            block = W[i:i + atom, j:j + atom]
            key = block.tobytes()
            if key not in catalog:
                catalog[key] = {"id": len(catalog), "W": block.copy()}
            grid[bi, bj] = catalog[key]["id"]
    # main view: the whole flux sheet (this IS the tile, not H)
    rects, origin, cell = _sheet(P, pitch, gap_frac)
    st = _polarity_stats(P)
    pads = [{"name": "UC", "x": origin, "y": origin + (n - 1) * pitch,
             "w": cell * 0.4, "h": cell * 0.4, "layer": "F.Cu"}]
    # lattice map to the right, one square per 8×8 atom
    lat_pitch = pitch * atom * 0.35
    lat0x = origin + n * pitch + 4.0
    lat0y = origin
    nb = grid.shape[0]
    for bi in range(nb):
        for bj in range(nb):
            tid = int(grid[bi, bj])
            layer = ("F.Cu", "B.Cu", "F.SilkS", "B.SilkS")[tid % 4]
            rects.append({
                "x": lat0x + bj * lat_pitch,
                "y": lat0y + (nb - 1 - bi) * lat_pitch,
                "w": lat_pitch * 0.85, "h": lat_pitch * 0.85,
                "layer": layer,
                "polarity": (1, -1, 0, 0)[tid % 4],
            })
    layout = {"rects": rects, "pads": pads, "vias": [],
              "bbox": _bbox_of(rects, pads), "kind": "metamaterial"}
    stats = {
        "kind": "metamaterial", "n": n, "atom": atom, "pitch_mm": pitch,
        "n_atoms": len(catalog), "n_blocks": int(nb * nb),
        "tile_counts": tiles.get("counts"),
        "h8_agree": tiles.get("h8_agree"),
        "kronecker_h8": tiles.get("kronecker_h8"),
        "n_plus": st["n_plus"], "n_zero": st["n_zero"], "n_minus": st["n_minus"],
        "sites_plus": st["sites_plus"],
        "sites_zero": st["sites_zero"],
        "sites_minus": st["sites_minus"],
        "n_vertices": st["sites_plus"],
        "n_edges": st["sites_zero"],
        "n_bulk": st["sites_minus"],
        "mean_w": float(W.mean()),
        "nested": tiles.get("nested"),
        "field": "flux P=2W-1",
    }
    return {"layout": layout, "stats": stats,
            "preview": preview_from_layout(layout),
            "lattice": grid.tolist()}


def design(kind: str, order: int, start: str = "sylvester",
           pitch_mm: float = 1.0, lib_dir=None) -> dict:
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected {KINDS}")
    H = normalize(load_H(order, start, lib_dir=lib_dir))
    pitch = float(pitch_mm)
    if kind == "cloth":
        out = cloth(H, pitch_mm=pitch)
    elif kind == "touchpad":
        out = touchpad(H, pitch_mm=max(pitch, 1.5))
    else:
        out = metamaterial(H, pitch_mm=max(pitch, 1.5))
    out["order"] = int(order)
    out["start"] = start
    out["tiles"] = flux_tiles(H)
    return out


# --- self-check --------------------------------------------------------------

if __name__ == "__main__":
    H8 = sylvester(8)
    P = flux_polarity(H8)
    assert set(np.unique(P).tolist()) == {-1, 0, 1}
    assert int(np.sum(P == 1)) == int(np.sum(flux_map(H8) >= 0.75))
    # must NOT be a ±1 copy of H
    assert not np.array_equal(P, H8) and not np.array_equal(P, -H8)
    c = cloth(H8, pitch_mm=1.0)
    assert c["stats"]["sites_zero"] > 0
    assert c["stats"]["field"] == "flux P=2W-1"
    assert c["preview"]["prims"]
    assert c["stats"]["warp_runs"] > 8
    print(f"PASS  cloth H8 flux: +{c['stats']['sites_plus']} "
          f"0={c['stats']['sites_zero']} −{c['stats']['sites_minus']} "
          f"runs={c['stats']['warp_runs']}")

    t = touchpad(H8, pitch_mm=2.0)
    assert t["stats"]["n_caps"] > 0
    assert t["stats"]["n_electrodes"] == (
        t["stats"]["n_plus"] + t["stats"]["n_zero"] + t["stats"]["n_minus"])
    print(f"PASS  touchpad H8: {t['stats']['n_electrodes']} electrodes "
          f"(+/0/−), {t['stats']['n_caps']} caps")

    m = metamaterial(sylvester(16), pitch_mm=2.0)
    assert m["stats"]["n_atoms"] == 4
    assert m["stats"]["n_vertices"] > 0 and m["stats"]["n_edges"] > 0
    print(f"PASS  meta H16: {m['stats']['n_atoms']} atoms, "
          f"V={m['stats']['n_vertices']} E={m['stats']['n_edges']} "
          f"bulk={m['stats']['n_bulk']}")

    d = design("cloth", 16, "sylvester")
    assert d["tiles"]["kronecker_h8"]
    print(f"PASS  design cloth/16: H8 agree {d['tiles']['h8_agree']:.3f}")

    print("materials self-check: all checks passed")
