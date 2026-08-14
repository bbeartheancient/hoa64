"""KiCad 7 PCB-antenna generator — physics dimensions → fabrication files.

Turns the analytic antenna designs of `em_physics.ANTENNA_TYPES` into
KiCad 7 (S-expression) footprint (``.kicad_mod``) and board
(``.kicad_pcb``) files for PCB antenna fabrication.  Everything is
procedural from the physics — no template numbers anywhere.

**Patch** (`footprint_patch`, `board_patch`).  Rectangular microstrip
patch, transmission-line/cavity model (Balanis ch. 14, see
`em_physics.build_patch`): W = (c/2f)√(2/(εᵣ+1)), L = c/(2f√ε_eff) −
2ΔL with the Hammerstad fringing ΔL.  The patch is one smd rect pad
(W×L, centered at origin, feed edge at −L/2).  The resonant edge
resistance R_in = 1/(2G₁) is typically 150–500 Ω; the inset feed moves
the tap point a depth y₀ into the patch so that

    R_in(y₀) = R_edge·cos²(π y₀/L) = 50 Ω   →   y₀ = (L/π)·acos(√(50/R_edge)),

cut as two slot keepouts flanking the feed tongue (slot width 1 mm,
gap 0.5 mm).  The 50 Ω microstrip feedline width comes from the
Wheeler/Hammerstad synthesis (`microstrip_width_50ohm`): with
A = (Z₀/60)√[(εᵣ+1)/2] + [(εᵣ−1)/(εᵣ+1)](0.23 + 0.11/εᵣ),

    W/h = 8e^A / (e^{2A} − 2)                    (W/h ≤ 2),

and the B-branch (Pozar 3.197) for W/h > 2.  FR4 (εᵣ = 4.4,
h = 1.6 mm) gives W/h ≈ 1.91 → W ≈ 3.06 mm.  The board ground plane
extends g = 6h beyond the patch edges (Balanis 14.3 finite-ground rule
of thumb) and sits as a B.Cu zone, net GND.

**Meander IFA** (`footprint_meander_ifa`).  Quarter-wave radiator
meandered into a compact bbox; the electrical length is the PIFA
estimate L_q = λ₀/(4√ε_eff), ε_eff = (εᵣ+1)/2 (em_physics.build_pifa).
The trace is a 4-arm serpentine of 0.5 mm fp_line segments on F.Cu
(arm spacing s = L_q/8 → arm = (L_q − 3s)/4, total path exactly L_q),
with a feed pad and a shorting-stub pad (IFA ground tap).

**Loop** (`footprint_loop`).  One-wavelength printed loop
(C = λ_medium, ka = 1 — em_physics.build_loop mode="resonant") as an
fp_circle on F.Cu with two feed pads at the bottom gap.

**Validity.**  `parse_sexpr` is a real tokenizer/parser (nested
lists, quoted strings), not a paren counter; the self-check parses
every generated file and checks KiCad-7 markers.  All coordinates are
mm with 6-decimal formatting, geometry centered at the origin.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
import uuid

from .em_physics import ANTENNA_TYPES, ETA0, medium_params

KICAD_VERSION = 20240108
GENERATOR = "hoa64.kicad_gen"

# --- formatting / s-expression helpers --------------------------------------


def _fmt(x: float) -> str:
    """mm coordinate with 6 decimals (KiCad canonical precision)."""
    s = f"{x:.6f}"
    return "0.000000" if s == "-0.000000" else s


def _sexpr(head: str, *items: str) -> str:
    """Single-line s-expression: _sexpr('at', '1', '2') → '(at 1 2)'."""
    body = " ".join((head, *items)).strip()
    return f"({body})"


def _xy(p) -> str:
    return _sexpr("xy", _fmt(p[0]), _fmt(p[1]))


def _tstamp() -> str:
    return str(uuid.uuid4())


# --- a real (small) s-expression parser for the validity gate ----------------

_TOKEN_RE = re.compile(r'\(|\)|"(?:\\.|[^"\\])*"|[^\s()"]+')


def parse_sexpr(text: str) -> list:
    """Parse one KiCad s-expression into nested lists of token strings."""
    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        raise ValueError("empty s-expression")
    pos = 0

    def _parse():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            out = []
            while True:
                if pos >= len(tokens):
                    raise ValueError("unbalanced parentheses: missing ')'")
                if tokens[pos] == ")":
                    pos += 1
                    return out
                out.append(_parse())
        if tok == ")":
            raise ValueError("unbalanced parentheses: unexpected ')'")
        return tok

    tree = _parse()
    if pos != len(tokens):
        raise ValueError(f"trailing tokens after top-level form: {tokens[pos]}")
    if not isinstance(tree, list):
        raise ValueError("top level must be a list")
    return tree


def _find_all(tree: list, head: str) -> list:
    """All sublists whose first element is `head` (recursive)."""
    out = []

    def walk(node):
        if isinstance(node, list):
            if node and node[0] == head:
                out.append(node)
            for ch in node:
                walk(ch)

    walk(tree)
    return out


# --- microstrip synthesis ----------------------------------------------------


def microstrip_width_50ohm(eps_r: float, h_m: float, z0: float = 50.0) -> float:
    """Wheeler/Hammerstad synthesis: conductor width (m) for Z₀ = `z0` Ω
    microstrip on a substrate εᵣ, height h_m.

    A = (Z₀/60)√[(εᵣ+1)/2] + [(εᵣ−1)/(εᵣ+1)](0.23 + 0.11/εᵣ);
    W/h = 8e^A/(e^{2A}−2) for W/h ≤ 2, else the wide-strip B-branch.
    FR4 (4.4, 1.6 mm) → ≈ 3.06 mm.
    """
    er = float(eps_r)
    a = (z0 / 60.0) * math.sqrt((er + 1.0) / 2.0) \
        + (er - 1.0) / (er + 1.0) * (0.23 + 0.11 / er)
    wh = 8.0 * math.exp(a) / (math.exp(2.0 * a) - 2.0)
    if wh > 2.0:  # wide strip: B = η₀π/(2Z₀√εᵣ), Pozar 3.197
        b = ETA0 * math.pi / (2.0 * z0 * math.sqrt(er))
        wh = (2.0 / math.pi) * (
            b - 1.0 - math.log(2.0 * b - 1.0)
            + (er - 1.0) / (2.0 * er)
            * (math.log(b - 1.0) + 0.39 - 0.61 / er))
    return wh * float(h_m)


# --- geometry → KiCad primitive emitters (coordinates in mm) -----------------


def _fp_line(p0, p1, layer: str, width: float) -> str:
    return ("  (fp_line " + _sexpr("start", _fmt(p0[0]), _fmt(p0[1])) + " "
            + _sexpr("end", _fmt(p1[0]), _fmt(p1[1])) + " "
            + _sexpr("stroke", _sexpr("width", _fmt(width)),
                     _sexpr("type", "solid"))
            + f' (layer "{layer}"))')


def _gr_line(p0, p1, layer: str, width: float) -> str:
    return _fp_line(p0, p1, layer, width).replace("(fp_line", "(gr_line", 1)


def _fp_rect(p0, p1, layer: str, width: float = 0.1, fill: str = "none",
             gr: bool = False) -> str:
    tag = "gr_rect" if gr else "fp_rect"
    return (f"  ({tag} " + _sexpr("start", _fmt(p0[0]), _fmt(p0[1])) + " "
            + _sexpr("end", _fmt(p1[0]), _fmt(p1[1])) + " "
            + _sexpr("stroke", _sexpr("width", _fmt(width)),
                     _sexpr("type", "solid"))
            + f' (fill {fill}) (layer "{layer}"))')


def _pad_rect(name: str, center, size, layers=('"F.Cu"',), rot: float = 0.0,
              extra: str = "") -> str:
    """smd rect pad; `center`/`size` in mm."""
    lay = " ".join(layers)
    s = (f'  (pad "{name}" smd rect '
         + _sexpr("at", _fmt(center[0]), _fmt(center[1]), _fmt(rot)) + " "
         + _sexpr("size", _fmt(size[0]), _fmt(size[1]))
         + f" (layers {lay})")
    if extra:
        s += " " + extra
    return s + ")"


def _gr_poly(points, layer: str, fill: str = "solid", width: float = 0.0,
             indent: str = "  ") -> str:
    pts = " ".join(_xy(p) for p in points)
    return (f"{indent}(gr_poly " + _sexpr("pts", *[_xy(p) for p in points])
            + " " + _sexpr("stroke", _sexpr("width", _fmt(width)),
                           _sexpr("type", "solid"))
            + f' (fill {fill}) (layer "{layer}"))')


def _rect_pts(cx: float, cy: float, sx: float, sy: float) -> list:
    """Axis-aligned rect polygon points (closed CCW) centered at (cx, cy)."""
    x0, x1 = cx - sx / 2.0, cx + sx / 2.0
    y0, y1 = cy - sy / 2.0, cy + sy / 2.0
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _keepout_zone_rect(p0, p1, layer: str) -> str:
    """Copper-keepout rule area (footprint-level inset-feed slot)."""
    pts = _rect_pts((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0,
                    abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))
    poly = " ".join(_xy(p) for p in pts)
    return (f'  (zone (net 0) (net_name "") (layers "{layer}") '
            f"(tstamp {_tstamp()})\n"
            "    (hatch edge 0.5)\n"
            "    (keepout (tracks allowed) (vias allowed) (pads allowed) "
            "(copperpour not_allowed) (footprints allowed))\n"
            f"    (polygon (pts {poly})))")


# --- footprint scaffolding ----------------------------------------------------


def _mhz(f_hz: float) -> int:
    return int(round(f_hz / 1e6))


def _footprint(name: str, descr: str, comment: str, body: list,
               text_dy: float = 5.0) -> str:
    lines = [
        f'(footprint "{name}"',
        f"  (version {KICAD_VERSION})",
        f'  (generator "{GENERATOR}")',
        '  (layer "F.Cu")',
        f'  (descr "{descr}")',
        "  (attr smd)",
        f'  (fp_text reference "REF**" (at 0 {_fmt(-text_dy)} 0) '
        '(layer "F.SilkS")',
        "    (effects (font (size 1 1) (thickness 0.15))))",
        f'  (fp_text value "{name}" (at 0 {_fmt(text_dy)} 0) '
        '(layer "F.SilkS")',
        "    (effects (font (size 1 1) (thickness 0.15))))",
        f'  (property "Comment" "{comment}" (at 0 0 0) (layer "F.SilkS") '
        "(hide yes)",
        "    (effects (font (size 0.5 0.5) (thickness 0.1))))",
        *body,
        ")",
    ]
    return "\n".join(lines) + "\n"


# --- patch --------------------------------------------------------------------


def _patch_geometry(f_hz: float, eps_r: float, h_m: float, feed: str) -> dict:
    """Shared patch geometry (all mm): W, L, inset depth y0, feed width."""
    if feed not in ("inset", "edge"):
        raise ValueError(f"feed must be 'inset' or 'edge', got {feed!r}")
    ant = ANTENNA_TYPES["patch"](f_hz, "air", eps_r=eps_r, h=h_m)
    d = ant["dimensions_m"]
    w_mm, l_mm = d["W"] * 1e3, d["L"] * 1e3
    r_edge = float(ant["z_in_ohm"].real)          # 1/(2G₁), Balanis 14-18
    w50_mm = microstrip_width_50ohm(eps_r, h_m) * 1e3
    feed_len_mm = 6.0
    y0 = 0.0
    slot_w, slot_gap = 0.0, 0.0
    if feed == "inset":
        # R_in(y0) = R_edge·cos²(π y0/L) = 50 Ω → inset depth
        y0 = l_mm / math.pi * math.acos(
            min(1.0, math.sqrt(50.0 / r_edge)))
        y0 = min(y0, 0.4 * l_mm)
        slot_w = min(1.0, w_mm / 20.0)
        slot_gap = 0.5
    return {"ant": ant, "W": w_mm, "L": l_mm, "w50": w50_mm,
            "feed_len": feed_len_mm, "y0": y0, "slot_w": slot_w,
            "slot_gap": slot_gap, "feed": feed}


def footprint_patch(f_hz: float, eps_r: float = 4.4, h_m: float = 1.6e-3,
                    feed: str = "inset") -> str:
    """KiCad 7 footprint (.kicad_mod) of an inset/edge-fed microstrip patch.

    Patch copper = one smd rect pad W×L on F.Cu (centered at origin,
    feed edge at −L/2); inset slots as copper keepouts; 50 Ω feedline
    (Wheeler/Hammerstad width) as a pad on the same number; an "ANT"
    smd pad marks the feed point at the inset vertex.
    """
    g = _patch_geometry(f_hz, eps_r, h_m, feed)
    w, l, w50, fl, y0 = g["W"], g["L"], g["w50"], g["feed_len"], g["y0"]
    mhz = _mhz(f_hz)

    body = [
        _pad_rect("1", (0.0, 0.0), (w, l)),          # the patch itself
        # 50 Ω feedline: patch edge −L/2 (or inset vertex) outward
        _pad_rect("1", (0.0, -l / 2.0 + (y0 - fl) / 2.0), (w50, fl + y0)),
        # feed-point marker at the inset vertex
        _pad_rect("ANT", (0.0, -l / 2.0 + y0), (0.6, 0.6)),
        # silkscreen outline
        _fp_rect((-w / 2.0 - 0.5, -l / 2.0 - fl - 0.5),
                 (w / 2.0 + 0.5, l / 2.0 + 0.5), "F.SilkS"),
    ]
    if feed == "inset":                              # two slot notches
        for sgn in (-1.0, 1.0):
            xi = sgn * (w50 / 2.0 + g["slot_gap"])
            xo = xi + sgn * g["slot_w"]
            body.append(_keepout_zone_rect(
                (min(xi, xo), -l / 2.0), (max(xi, xo), -l / 2.0 + y0),
                "F.Cu"))

    comment = (f"microstrip patch @ {mhz} MHz, eps_r={eps_r}, "
               f"h={h_m * 1e3:.3g} mm, feed={feed}; "
               "W=(c/2f)sqrt(2/(er+1)), L=c/(2f*sqrt(e_eff))-2*dL "
               "(Balanis 14-6/14-1/14-2/14-7, Hammerstad), "
               "inset y0: R_in*cos^2(pi*y0/L)=50 ohm, "
               "feedline w50 from Wheeler/Hammerstad 50-ohm synthesis "
               "(hoa64.em_physics.build_patch)")
    return _footprint(
        f"hoa64_patch_{mhz}",
        f"Rectangular microstrip patch antenna, {mhz} MHz, {feed} feed "
        "(hoa64.kicad_gen)", comment, body,
        text_dy=max(l / 2.0 + 2.0, 5.0))


# --- meander IFA ---------------------------------------------------------------


def _meander_path(f_hz: float, eps_r: float, h_m: float) -> tuple:
    """Serpentine path (mm) for a quarter-wave meander IFA.

    Electrical length L_q = λ₀/(4√ε_eff), ε_eff = (εᵣ+1)/2 (the PIFA
    model in em_physics).  4 arms spaced s = L_q/8 → path = 4·arm + 3s
    = L_q exactly.  Returns (points, l_q_mm, antenna dict).
    """
    ant = ANTENNA_TYPES["pifa"](f_hz, "air", eps_r=eps_r, h=h_m)
    lq = ant["dimensions_m"]["length"] * 1e3          # mm
    n_arms = 4
    s = lq / 8.0
    arm = (lq - (n_arms - 1) * s) / n_arms
    if arm < 2.0:
        raise ValueError(f"meander arm {arm:.3f} mm too short at {f_hz:.4g} Hz")
    pts = [(0.0, 0.0)]
    for i in range(n_arms):
        x = arm if i % 2 == 0 else 0.0
        pts.append((x, i * s))                        # horizontal arm
        if i < n_arms - 1:
            pts.append((x, (i + 1) * s))              # vertical connector
    return pts, lq, ant


def footprint_meander_ifa(f_hz: float, eps_r: float = 4.4,
                          h_m: float = 1.6e-3) -> str:
    """KiCad 7 footprint of a meandered quarter-wave inverted-F antenna.

    Trace = fp_line polyline (0.5 mm) on F.Cu with total path length
    L_q = λ₀/(4√ε_eff); feed pad "1" at the path start, shorting-stub
    pad "2" (ground tap) below it.
    """
    pts, lq, ant = _meander_path(f_hz, eps_r, h_m)
    mhz = _mhz(f_hz)
    trace_w = 0.5
    stub = 2.0

    body = [
        _pad_rect("1", pts[0], (1.0, 1.0)),                    # feed
        _pad_rect("2", (0.0, -stub), (1.0, 1.0)),              # short → GND
        _fp_line(pts[0], (0.0, -stub), "F.Cu", trace_w),       # shorting stub
    ]
    body += [_fp_line(a, b, "F.Cu", trace_w) for a, b in zip(pts, pts[1:])]
    arm = max(p[0] for p in pts)
    top = max(p[1] for p in pts)
    body.append(_fp_rect((-1.0, -stub - 1.0), (arm + 1.0, top + 1.0),
                         "F.SilkS"))

    e_eff = 0.5 * (eps_r + 1.0)
    comment = (f"meander IFA @ {mhz} MHz: quarter-wave radiator, "
               f"L_q = lambda0/(4*sqrt(e_eff)) = {lq:.3f} mm with "
               f"e_eff = (er+1)/2 = {e_eff:.2f} (em_physics.build_pifa); "
               "4-arm serpentine, 0.5 mm trace, pad 1 = feed, "
               "pad 2 = shorting stub to ground")
    return _footprint(
        f"hoa64_meander_ifa_{mhz}",
        f"Meandered quarter-wave IFA PCB antenna, {mhz} MHz "
        "(hoa64.kicad_gen)", comment, body,
        text_dy=max(top / 2.0 + 2.0, 5.0))


# --- loop ---------------------------------------------------------------------


def footprint_loop(f_hz: float, medium: str = "air") -> str:
    """KiCad 7 footprint of a one-wavelength printed loop (C = λ_medium,
    ka = 1 — em_physics.build_loop mode='resonant'): fp_circle on F.Cu
    with feed pads "1"/"2" at the bottom gap.
    """
    ant = ANTENNA_TYPES["loop"](f_hz, medium, mode="resonant")
    d = ant["dimensions_m"]
    r_mm = d["radius"] * 1e3
    mhz = _mhz(f_hz)

    body = [
        "  (fp_circle " + _sexpr("center", "0.000000", "0.000000") + " "
        + _sexpr("end", _fmt(r_mm), "0.000000") + " "
        + _sexpr("stroke", _sexpr("width", "0.500000"),
                 _sexpr("type", "solid"))
        + ' (fill none) (layer "F.Cu"))',
        _pad_rect("1", (-0.4, -r_mm), (0.8, 0.8)),
        _pad_rect("2", (0.4, -r_mm), (0.8, 0.8)),
        _fp_rect((-r_mm - 0.5, -r_mm - 0.5), (r_mm + 0.5, r_mm + 0.5),
                 "F.SilkS"),
    ]
    comment = (f"1-wavelength printed loop @ {mhz} MHz in {medium}: "
               f"C = lambda_medium = {d['circumference'] * 1e3:.3f} mm, "
               f"ka = 1, radius = {r_mm:.3f} mm "
               "(em_physics.build_loop mode=resonant; Balanis ch. 5 "
               "uniform-current loop)")
    return _footprint(
        f"hoa64_loop_{mhz}",
        f"One-wavelength printed loop antenna, {mhz} MHz, {medium} "
        "(hoa64.kicad_gen)", comment, body,
        text_dy=r_mm + 3.0)


# --- board (.kicad_pcb) ---------------------------------------------------------

_LAYERS_BLOCK = """  (layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
    (32 "B.Adhes" user "B.Adhesive")
    (33 "F.Adhes" user "F.Adhesive")
    (34 "B.Paste" user)
    (35 "F.Paste" user)
    (36 "B.SilkS" user "B.Silkscreen")
    (37 "F.SilkS" user "F.Silkscreen")
    (38 "B.Mask" user)
    (39 "F.Mask" user)
    (40 "Dwgs.User" user "User.Drawings")
    (41 "Cmts.User" user "User.Comments")
    (42 "Eco1.User" user "User.Eco1")
    (43 "Eco2.User" user "User.Eco2")
    (44 "Edge.Cuts" user)
    (45 "Margin" user)
    (46 "B.CrtYd" user "B.Courtyard")
    (47 "F.CrtYd" user "F.Courtyard")
    (48 "B.Fab" user)
    (49 "F.Fab" user)
  )"""


def _patch_outline_pts(g: dict) -> list:
    """Patch copper outline (mm) with the inset-feed slot cutouts."""
    w, l, y0 = g["W"], g["L"], g["y0"]
    pts = [(-w / 2.0, -l / 2.0)]
    if g["feed"] == "inset":
        xi = g["w50"] / 2.0 + g["slot_gap"]
        xo = xi + g["slot_w"]
        yv = -l / 2.0 + y0
        pts += [(-xo, -l / 2.0), (-xo, yv), (-xi, yv), (-xi, -l / 2.0),
                (xi, -l / 2.0), (xi, yv), (xo, yv), (xo, -l / 2.0)]
    pts += [(w / 2.0, -l / 2.0), (w / 2.0, l / 2.0), (-w / 2.0, l / 2.0)]
    return pts


def board_patch(f_hz: float, eps_r: float = 4.4, h_m: float = 1.6e-3,
                feed: str = "inset", board_margin_m: float = 5e-3) -> str:
    """KiCad 7 board (.kicad_pcb): patch + 50 Ω feedline as gr_poly copper
    on F.Cu, full ground pour zone on B.Cu (net GND), Edge.Cuts outline.

    The ground plane extends g = 6h beyond the patch edges (Balanis
    finite-ground rule) and beyond the feedline end; the board outline
    adds `board_margin_m` around the ground extent.
    """
    g = _patch_geometry(f_hz, eps_r, h_m, feed)
    w, l, w50, fl, y0 = g["W"], g["L"], g["w50"], g["feed_len"], g["y0"]
    mhz = _mhz(f_hz)
    gnd_ext = 6.0 * h_m * 1e3                          # Balanis: ≥ ~6h
    margin = board_margin_m * 1e3

    gx = w / 2.0 + gnd_ext                             # ground half-width
    gy0 = -(l / 2.0 + fl + gnd_ext)                    # ground bottom
    gy1 = l / 2.0 + gnd_ext                            # ground top
    bx0, bx1 = -gx - margin, gx + margin               # board outline
    by0, by1 = gy0 - margin, gy1 + margin

    feed_cy = -l / 2.0 + (y0 - fl) / 2.0
    parts = [
        "(kicad_pcb",
        f"  (version {KICAD_VERSION})",
        f'  (generator "{GENERATOR}")',
        "  (general (thickness " + _fmt(h_m * 1e3) + "))",
        '  (paper "A4")',
        f'  (title_block (title "hoa64 patch {mhz} MHz ({feed} feed)") '
        '(comment 1 "W/L per Balanis 14-6/14-7, feedline per '
        'Wheeler/Hammerstad; hoa64.em_physics.build_patch"))',
        _LAYERS_BLOCK,
        "  (setup (pad_to_mask_clearance 0))",
        '  (net 0 "")',
        '  (net 1 "GND")',
        '  (net 2 "ANT")',
        # patch copper (with inset slots) on F.Cu, net ANT
        _gr_poly(_patch_outline_pts(g), "F.Cu"),
        # 50 Ω feedline on F.Cu
        _gr_poly(_rect_pts(0.0, feed_cy, w50, fl + y0), "F.Cu"),
        # board outline
        _fp_rect((bx0, by0), (bx1, by1), "Edge.Cuts", width=0.05, gr=True),
        # ground pour on B.Cu
        '  (zone (net 1) (net_name "GND") (layer "B.Cu") '
        f"(tstamp {_tstamp()})\n"
        "    (hatch edge 0.5)\n"
        "    (connect_pads (clearance 0.5))\n"
        "    (min_thickness 0.25)\n"
        "    (fill yes (thermal_gap 0.5) (thermal_bridge_width 0.5))\n"
        "    (polygon (pts "
        + " ".join(_xy(p) for p in [(-gx, gy0), (gx, gy0),
                                    (gx, gy1), (-gx, gy1)])
        + ")))",
        ")",
    ]
    return "\n".join(parts) + "\n"


# --- dispatcher -----------------------------------------------------------------


def kicad_files(design_type: str, f_hz: float, **opts) -> dict:
    """{filename: content} for design_type ∈ {patch, meander_ifa, loop}.

    Always includes the .kicad_mod footprint; "patch" also gets the
    .kicad_pcb board (ground pour + Edge.Cuts).
    """
    mhz = _mhz(f_hz)
    if design_type == "patch":
        return {
            f"hoa64_patch_{mhz}.kicad_mod": footprint_patch(f_hz, **{
                k: v for k, v in opts.items() if k in ("eps_r", "h_m", "feed")}),
            f"hoa64_patch_{mhz}.kicad_pcb": board_patch(f_hz, **{
                k: v for k, v in opts.items()
                if k in ("eps_r", "h_m", "feed", "board_margin_m")}),
        }
    if design_type == "meander_ifa":
        return {
            f"hoa64_meander_ifa_{mhz}.kicad_mod": footprint_meander_ifa(
                f_hz, **{k: v for k, v in opts.items()
                         if k in ("eps_r", "h_m")}),
        }
    if design_type == "loop":
        return {
            f"hoa64_loop_{mhz}.kicad_mod": footprint_loop(f_hz, **{
                k: v for k, v in opts.items() if k in ("medium",)}),
        }
    raise ValueError(
        f"unknown design_type {design_type!r}; "
        "expected one of 'patch', 'meander_ifa', 'loop'")


# --- self-check -----------------------------------------------------------------

if __name__ == "__main__":
    import os

    F0 = 2.45e9
    EPS_R, H = 4.4, 1.6e-3

    # --- microstrip 50 Ω synthesis: FR4 1.6 mm → ≈ 3.06 mm ---
    w50 = microstrip_width_50ohm(EPS_R, H) * 1e3
    assert 3.0 <= w50 <= 3.1, w50
    print(f"PASS  microstrip_width_50ohm(4.4, 1.6 mm) = {w50:.4f} mm "
          "∈ [3.0, 3.1]")

    # --- patch footprint: parses, pad size = W×L from em_physics, feedline ---
    fp = footprint_patch(F0)
    tree = parse_sexpr(fp)
    assert tree[0] == "footprint"
    assert '(layer "F.Cu")' in fp and "(pad " in fp
    ant = ANTENNA_TYPES["patch"](F0, "air", eps_r=EPS_R, h=H)
    w_mm, l_mm = ant["dimensions_m"]["W"] * 1e3, ant["dimensions_m"]["L"] * 1e3
    size_str = _sexpr("size", _fmt(w_mm), _fmt(l_mm))
    assert size_str in fp, size_str
    feed_str = _sexpr("size", _fmt(w50), _fmt(6.0 + _patch_geometry(
        F0, EPS_R, H, "inset")["y0"]))
    assert feed_str in fp, feed_str
    assert abs(w50 - 3.0) / 3.0 < 0.10, w50
    print(f"PASS  footprint_patch(2.45 GHz): parses; patch pad {size_str} "
          f"(W = {w_mm:.3f} mm, L = {l_mm:.3f} mm); feedline {w50:.3f} mm "
          "≈ 3.0 mm (±10 %)")

    # --- patch board: zone on B.Cu, Edge.Cuts, ground ≥ 6h beyond patch ---
    bp = board_patch(F0)
    btree = parse_sexpr(bp)
    assert btree[0] == "kicad_pcb"
    zones = _find_all(btree, "zone")
    assert zones and '(layer "B.Cu")' in bp and '(net_name "GND")' in bp
    assert '"Edge.Cuts"' in bp
    gnd_half = w_mm / 2.0 + 6.0 * H * 1e3
    assert _fmt(gnd_half) in bp                      # zone x-extent = W/2 + 6h
    margin_mm = 5e-3 * 1e3
    assert _fmt(gnd_half + margin_mm) in bp          # Edge.Cuts x-extent
    print(f"PASS  board_patch(2.45 GHz): parses; GND zone on B.Cu extends "
          f"{6.0 * H * 1e3:.1f} mm = 6h beyond patch edges, Edge.Cuts rect "
          f"+{margin_mm:.1f} mm margin")

    # --- meander IFA: parses, total path = λ_eff/4 (±5 %) ---
    fm = footprint_meander_ifa(F0)
    mtree = parse_sexpr(fm)
    assert mtree[0] == "footprint" and "(fp_line" in fm
    pts, lq, _ = _meander_path(F0, EPS_R, H)
    path_len = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                   for a, b in zip(pts, pts[1:]))
    lam_eff_4 = medium_params(F0, "air")["lambda0"] * 1e3 \
        / (4.0 * math.sqrt(0.5 * (EPS_R + 1.0)))
    assert abs(path_len - lam_eff_4) / lam_eff_4 < 0.05, (path_len, lam_eff_4)
    assert abs(path_len - lq) < 1e-9
    print(f"PASS  footprint_meander_ifa(2.45 GHz): parses; meander path "
          f"{path_len:.3f} mm ≈ λ_eff/4 = {lam_eff_4:.3f} mm (±5 %)")

    # --- loop: parses, circumference = λ_medium (±2 %) ---
    fl_ = footprint_loop(F0)
    ltree = parse_sexpr(fl_)
    assert ltree[0] == "footprint" and "(fp_circle" in fl_
    loop_ant = ANTENNA_TYPES["loop"](F0, "air", mode="resonant")
    circ = loop_ant["dimensions_m"]["circumference"]
    lam_med = medium_params(F0, "air")["wavelength"]
    assert abs(circ - lam_med) / lam_med < 0.02, (circ, lam_med)
    print(f"PASS  footprint_loop(2.45 GHz): parses; circumference "
          f"{circ * 1e3:.3f} mm ≈ λ_medium = {lam_med * 1e3:.3f} mm (±2 %)")

    # --- kicad_files round-trip for all three design types ---
    files = {}
    for dt in ("patch", "meander_ifa", "loop"):
        files.update(kicad_files(dt, F0))
    assert len(files) == 4, sorted(files)
    assert any(k.endswith(".kicad_pcb") for k in files)
    for name, content in files.items():
        t = parse_sexpr(content)
        assert t[0] in ("footprint", "kicad_pcb"), name
    print(f"PASS  kicad_files round-trip: {sorted(files)} all parse")

    # --- optional kicad-cli validation ---
    cli = shutil.which("kicad-cli")
    if cli:
        with tempfile.TemporaryDirectory() as td:
            ok = True
            for name, content in files.items():
                src = os.path.join(td, name)
                with open(src, "w") as fh:
                    fh.write(content)
                if name.endswith(".kicad_pcb"):
                    # kicad-cli 10: `pcb upgrade` rewrites INPUT_FILE in
                    # place (no -o) — a full load+save round-trip parse.
                    r = subprocess.run(
                        [cli, "pcb", "upgrade", "--force", src],
                        capture_output=True, text=True)
                    ok = ok and r.returncode == 0
            assert ok, "kicad-cli pcb upgrade failed"
        print("PASS  kicad-cli: pcb upgrade validated the generated board")
    else:
        print("NOTE  kicad-cli not on PATH — relying on the s-expression "
              "parser gate only")

    print("kicad_gen self-check: all checks passed")
