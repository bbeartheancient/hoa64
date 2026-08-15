"""Materials lab — physical realisations of the H.8 flux-tile catalog.

Three homes for the 4-tile Walsh wall pattern of Sylvester / A ⊗ H₈
(see `micromag.flux_tiles`).  The catalog is not a named object in the
Hadamard literature; these are concrete layouts that *use* it.

Two fields, because the stack is **two copper layers**:

    H ∈ {+1, −1}     layer assignment (face / reverse)
    W ∈ {0, ½, 1}    flux tile (bulk / edge / vertex)
    P = 2W − 1       ternary colour of the tile, *not* a 3rd net

P = 0 (W = ½) is a dielectric gap, not a conductor.  Putting it on a
third yarn is not fabricable on 2-layer cloth or 2-layer PCB, and
the W = 0 cells are almost all isolated pixels if used as a net.
Copper therefore follows H (two continuous face/reverse electrodes);
the flux tile is the *map* — fill colour and a smaller pad on edge
cells — so the preview matches ``fluxtile.png`` while the gerber
stays two-layer.

**Cloth.**  Warp/weft runs follow H; a run breaks where H changes
(the domain wall).  Face = H=+1 (F.Cu), reverse = H=−1 (B.Cu).

**Touchpad.**  Electrodes are the 4-connected components of each H
sign.  Each H-changing bond is a mutual-cap; W at the two cells
says whether that gap is a straight wall (½) or a corner (1).

**Metamaterial.**  Same 2-layer copper; the flux sheet is the
patterned dielectric / via map (vertices W=1, edges W=½, bulk W=0)
plus the Walsh lattice of the four H.8 atoms.

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


def _layer_of_H(h: int) -> str:
    """2-layer stack: H=+1 face (F.Cu), H=−1 reverse (B.Cu)."""
    return "F.Cu" if int(h) > 0 else "B.Cu"


def _layer_of(p: int) -> str:
    """Legacy name — flux polarity is a *colour*, not a copper layer."""
    if p > 0:
        return "F.Cu"
    if p < 0:
        return "B.Cu"
    return "F.SilkS"


MAP_KEY = {
    "stack": "2-layer: copper follows H (±1); fill colour is flux P=2W−1",
    "fill": [
        {"polarity": 1, "w": 1.0, "swatch": "F.Cu", "layer": "F.Cu",
         "name": "F.Cu", "means": "top copper (red) — H=+1 face"},
        {"polarity": -1, "w": 0.0, "swatch": "B.Cu", "layer": "B.Cu",
         "name": "B.Cu", "means": "bottom copper (blue) — H=−1 reverse"},
        {"polarity": 0, "w": 0.5, "swatch": "",
         "name": "gap", "means": "W=½ edge — unfilled dielectric, not a 3rd net"},
    ],
    "copper": [
        {"layer": "F.Cu", "h": 1, "name": "F.Cu / face",
         "means": "red — H=+1 top yarn"},
        {"layer": "B.Cu", "h": -1, "name": "B.Cu / reverse",
         "means": "blue — H=−1 bottom yarn"},
    ],
    "feeds": [
        {"name": "○ F# / B#", "swatch": "fg",
         "means": "electrode feed — snapped to a cell of that net, not extra copper"},
    ],
}


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
            "polarity": p.get("polarity"),
            "role": p.get("role", "pad"),
        })
    for v in layout.get("vias") or []:
        prims.append({
            "kind": "circle", "c": [v["x"], v["y"]],
            "r": float(v["d"]) / 2.0, "w": 0.15, "layer": "F.Cu",
        })
    return {"prims": prims, "bbox": layout["bbox"]}


def _sheet(H: np.ndarray, P: np.ndarray, pitch: float, gap_frac: float):
    """n×n cells: copper layer from H, fill polarity from flux P.

    Edge cells (P=0, W=½) are omitted — unfilled dielectric, not copper.
    """
    n = H.shape[0]
    cell = pitch * (1.0 - float(gap_frac))
    origin = -0.5 * (n - 1) * pitch
    rects = []
    for i in range(n):
        for j in range(n):
            p = int(P[i, j])
            if p == 0:
                continue
            h = int(H[i, j])
            rects.append({
                "x": origin + j * pitch,
                "y": origin + (n - 1 - i) * pitch,
                "w": cell, "h": cell,
                "layer": _layer_of_H(h),
                "polarity": p,
                "h_sign": h,
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
    """2-layer cloth: copper from H, flux tile as the visible map."""
    H = np.asarray(H, dtype=np.int8)
    P = flux_polarity(H)
    n = H.shape[0]
    pitch = float(pitch_mm)
    rects, origin, cell = _sheet(H, P, pitch, gap_frac)
    face_lab, n_face = _components((H == 1) & (P != 0))
    rev_lab, n_rev = _components((H == -1) & (P != 0))
    st = _polarity_stats(P)
    pads = []
    for name, mask, layer in (
        ("F", (H[:, 0] == 1) & (P[:, 0] != 0), "F.Cu"),
        ("B", (H[:, 0] == -1) & (P[:, 0] != 0), "B.Cu"),
    ):
        hits = np.argwhere(mask)
        if len(hits):
            i = int(hits[0, 0])
            pads.append({"name": name, "x": origin - pitch,
                         "y": origin + (n - 1 - i) * pitch,
                         "w": cell, "h": cell, "layer": layer,
                         "role": "feed", "polarity": 1 if name == "F" else -1})
    layout = {"rects": rects, "pads": pads, "vias": [],
              "bbox": _bbox_of(rects, pads), "kind": "cloth"}
    stats = {
        "kind": "cloth", "n": n, "pitch_mm": pitch,
        "n_face": int(n_face), "n_reverse": int(n_rev),
        "sites_face": int(np.sum(H == 1)),
        "sites_reverse": int(np.sum(H == -1)),
        "sites_plus": st["sites_plus"],
        "sites_zero": st["sites_zero"],
        "sites_minus": st["sites_minus"],
        "warp_runs": sum(len(_runs(H[:, j])) for j in range(n)),
        "weft_runs": sum(len(_runs(H[i])) for i in range(n)),
        "wall_bonds": int(np.sum(H[:, :-1] != H[:, 1:]))
                      + int(np.sum(H[:-1, :] != H[1:, :])),
        "mean_w": float(flux_map(H).mean()),
        "fill": 1.0 - gap_frac,
        "field": "copper=H  fill=flux P=2W-1  0=gap",
        "stack": "2-layer",
    }
    return {"layout": layout, "stats": stats, "preview": preview_from_layout(layout)}


# --- touchpad ----------------------------------------------------------------


def touchpad(H, pitch_mm: float = 2.0, gap_frac: float = 0.2) -> dict:
    """2-layer mutual-cap pad: electrodes from H, walls from flux."""
    H = np.asarray(H, dtype=np.int8)
    P = flux_polarity(H)
    n = H.shape[0]
    pitch = float(pitch_mm)
    rects, origin, cell = _sheet(H, P, pitch, gap_frac)
    face_lab, n_face = _components((H == 1) & (P != 0))
    rev_lab, n_rev = _components((H == -1) & (P != 0))
    st = _polarity_stats(P)
    W = flux_map(H)
    caps = []
    for i in range(n):
        for j in range(n - 1):
            if H[i, j] == H[i, j + 1] or P[i, j] == 0 or P[i, j + 1] == 0:
                continue
            a, b = (int(face_lab[i, j]), int(rev_lab[i, j + 1])) if H[i, j] == 1 \
                else (int(face_lab[i, j + 1]), int(rev_lab[i, j]))
            if a < 0 or b < 0:
                continue
            caps.append({"face": a, "reverse": b, "dir": "h",
                         "w": [float(W[i, j]), float(W[i, j + 1])]})
    for i in range(n - 1):
        for j in range(n):
            if H[i, j] == H[i + 1, j] or P[i, j] == 0 or P[i + 1, j] == 0:
                continue
            a, b = (int(face_lab[i, j]), int(rev_lab[i + 1, j])) if H[i, j] == 1 \
                else (int(face_lab[i + 1, j]), int(rev_lab[i, j]))
            if a < 0 or b < 0:
                continue
            caps.append({"face": a, "reverse": b, "dir": "v",
                         "w": [float(W[i, j]), float(W[i + 1, j])]})
    pads = []
    for lab, n_el, prefix, layer, hsign in (
        (face_lab, n_face, "F", "F.Cu", 1),
        (rev_lab, n_rev, "B", "B.Cu", -1),
    ):
        for eid in range(n_el):
            cells = np.argwhere(lab == eid)
            # prefer a copper cell (P≠0); W=½ cells are unfilled gaps
            copper = np.array([c for c in cells if int(P[int(c[0]), int(c[1])]) != 0])
            pick = copper if len(copper) else cells
            cy, cx = pick.mean(axis=0)
            k = int(np.argmin((pick[:, 0] - cy) ** 2 + (pick[:, 1] - cx) ** 2))
            i, j = int(pick[k, 0]), int(pick[k, 1])
            pads.append({
                "name": f"{prefix}{eid}",
                "x": origin + j * pitch,
                "y": origin + (n - 1 - i) * pitch,
                "w": cell * 0.28, "h": cell * 0.28,
                "layer": layer,
                "polarity": int(P[i, j]),
                "role": "feed",
            })
    layout = {"rects": rects, "pads": pads, "vias": [],
              "bbox": _bbox_of(rects, pads), "kind": "touchpad"}
    stats = {
        "kind": "touchpad", "n": n, "pitch_mm": pitch,
        "n_electrodes": int(n_face + n_rev),
        "n_face": int(n_face), "n_reverse": int(n_rev),
        "sites_plus": st["sites_plus"],
        "sites_zero": st["sites_zero"],
        "sites_minus": st["sites_minus"],
        "n_caps": len(caps),
        "caps_per_cell": len(caps) / max(n * n, 1),
        "mean_w": float(flux_map(H).mean()),
        "field": "copper=H  fill=flux P=2W-1  0=gap",
        "stack": "2-layer",
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
    # main view: flux-coloured sheet on 2-layer H copper
    rects, origin, cell = _sheet(H, P, pitch, gap_frac)
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
        "field": "copper=H  fill=flux P=2W-1  0=gap",
        "stack": "2-layer",
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
    out["key"] = MAP_KEY
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
    assert "2-layer" in c["stats"]["stack"]
    assert c["stats"]["n_face"] >= 1 and c["stats"]["n_reverse"] >= 1
    assert c["preview"]["prims"]
    assert c["stats"]["warp_runs"] > 8
    print(f"PASS  cloth H8 2-layer: face={c['stats']['n_face']} "
          f"rev={c['stats']['n_reverse']}  flux +/0/− "
          f"{c['stats']['sites_plus']}/{c['stats']['sites_zero']}/{c['stats']['sites_minus']}")

    t = touchpad(H8, pitch_mm=2.0)
    assert t["stats"]["n_caps"] > 0
    assert t["stats"]["n_electrodes"] == t["stats"]["n_face"] + t["stats"]["n_reverse"]
    feeds = [p for p in t["layout"]["pads"] if p.get("role") == "feed"]
    assert len(feeds) == t["stats"]["n_electrodes"]
    cell_xy = {(round(r["x"], 6), round(r["y"], 6)) for r in t["layout"]["rects"]}
    for p in feeds:
        assert (round(p["x"], 6), round(p["y"], 6)) in cell_xy, p
    print(f"PASS  touchpad H8: {t['stats']['n_electrodes']} 2-layer electrodes, "
          f"{t['stats']['n_caps']} caps, {len(feeds)} snapped feeds")

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
