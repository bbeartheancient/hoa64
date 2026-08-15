"""Virtual site survey — terrain-aware RF link analysis on open elevation data.

No API key, no local DEM: elevations come from the AWS Open Data Terrain
Tiles (Mapzen Terrarium encoding), plain PNGs served at

    https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png

Terrarium encoding
------------------
Each pixel packs the elevation in metres into its RGB channels as a
fixed-point number with 1/256 m resolution:

    elevation_m = R·256 + G + B/256 − 32768

i.e. a 24-bit unsigned integer (R·65536 + G·256 + B) scaled by 1/256 and
biased by −32768 m, covering −32768 … +65535.996 m.

Slippy-map tiling (Web Mercator, EPSG:3857)
-------------------------------------------
At zoom z the world is a 2^z × 2^z grid of 256 px tiles.  With n = 2^z
and λ = longitude in degrees:

    x = floor((λ + 180)/360 · n)
    y = floor((1 − asinh(tan φ)/π)/2 · n)        φ = latitude (rad)

the y formula being the standard spherical-Mercator projection
y ∝ 1 − (1/π)·ln(tan φ + sec φ), valid for |φ| ≤ 85.0511°.  Within a
tile, the global fractional pixel coordinate gives the sub-pixel sample
position; `elevation` bilinearly interpolates between pixel centres
(centre of pixel (i, j) at i + 0.5) so profiles are C⁰ across tile
boundaries up to quantization.

Propagation model
-----------------
The link geometry uses the classic 4/3-earth-radius effective model:
atmospheric refraction bends rays toward the ground, which to first
order is absorbed by pretending the earth has radius R_eff = (4/3)R⊕
and rays are straight.  On that flat-ray picture the earth bulge
relative to the chord between the two antenna base points at path
position d (total path D) is

    b(d) = d·(D − d) / (2·R_eff)

and is *added* to the terrain elevation.  The LOS reference is the
straight line between the two antenna phase centres (terrain height +
mast height at each end).

First Fresnel zone
------------------
At path position with distances d₁, d₂ to the two ends the first
Fresnel zone radius is

    r₁ = √(λ·d₁·d₂ / (d₁ + d₂))

with λ the in-medium wavelength from `em_physics.medium_params`.  The
report normalizes terrain clearance by r₁; the engineering rule of
thumb is ≥ 60 % of r₁ clear for (near-)free-space loss.

Diffraction — Deygout multiple knife-edge
-----------------------------------------
Each interior terrain point above the LOS line is treated as a
knife-edge with Fresnel diffraction parameter

    v = h·√(2·(d₁ + d₂) / (λ·d₁·d₂)),   h = obstruction height above LOS

and single-edge loss (ITU-R P.526 approximation)

    J(v) = 6.9 + 20·log₁₀(√((v − 0.1)² + 1) + v − 0.1)   for v > −0.78
    J(v) = 0                                                otherwise

Deygout's method: the dominant edge (max v) splits the path; secondary
edges are sought in each sub-path with obstruction heights measured
above the line joining that sub-path's endpoints (TX→main edge,
main edge→RX).  Simplification: only one level of recursion, and a
sub-edge contributes only if its own v exceeds −0.78.  Total
diffraction loss is the sum of the edge losses in dB.  This is the
standard link-budget simplification — it errs on the pessimistic side
for few, well-separated edges.

The full link budget (Friis + medium attenuation) comes from
`em_physics.link_budget`; the diffraction loss is then subtracted to
give the expected received power.
"""

from __future__ import annotations

import math
import os
import urllib.request

import numpy as np

from .em_physics import link_budget, medium_params
from .webapp._png import decode_png, write_png

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TILE_PX = 256
EARTH_R_M = 6_371_000.0          # mean earth radius
K_EARTH = 4.0 / 3.0              # effective-earth-radius factor
MAX_PATH_M = 200_000.0           # beyond this the tile count explodes
MIN_PATH_M = 50.0                # coincident TX/RX → 1 km east hop
MAX_LAT = 85.05112878            # Web-Mercator latitude limit
FRESNEL_CLEAR_FRAC = 0.6         # ≥60 % of r1 clear ≈ free-space loss
V_THRESHOLD = -0.78              # knife-edge contributes only above this
RADIO_HORIZON_K = 4.12           # km / √(h_m) — 4/3-earth geometric horizon
IMAGERY_URLS = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "https://services.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}",
)
_TILE_UA = {"User-Agent": "hoa64-lab/0.5 (site survey)"}


# ------------------------------------------------------------ tiles

def _tile_pixel(lat: float, lon: float, z: int) -> tuple[int, int, float, float]:
    """Slippy-map tile (x, y) at zoom z plus fractional pixel (fx, fy).

    fx, fy ∈ [0, 256) are the pixel coordinates *within* the tile of the
    query point (pixel centres at i + 0.5).
    """
    lat = max(-MAX_LAT, min(MAX_LAT, lat))
    n = 1 << z
    px = (lon + 180.0) / 360.0 * n * TILE_PX
    phi = math.radians(lat)
    py = (1.0 - math.asinh(math.tan(phi)) / math.pi) / 2.0 * n * TILE_PX
    x = min(int(px // TILE_PX), n - 1)
    y = min(int(py // TILE_PX), n - 1)
    return x, y, px - x * TILE_PX, py - y * TILE_PX


def fetch_tile(z: int, x: int, y: int, cache_dir: str | None = None) -> np.ndarray:
    """Download (or load from cache) one Terrarium tile → (256, 256, 3) uint8.

    Cache layout: {cache_dir}/{z}/{x}/{y}.png, defaulting to
    ~/.cache/hoa64/terrain.  10 s network timeout.
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/hoa64/terrain")
    path = os.path.join(cache_dir, str(z), str(x), f"{y}.png")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return decode_png(f.read())
    url = TILE_URL.format(z=z, x=x, y=y)
    req = urllib.request.Request(url, headers=_TILE_UA)
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = resp.read()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return decode_png(data)


def _terrarium_elev(tile: np.ndarray, i: int, j: int) -> float:
    """Decode one Terrarium pixel: elevation_m = R·256 + G + B/256 − 32768."""
    r, g, b = (int(v) for v in tile[j, i])
    return r * 256.0 + g + b / 256.0 - 32768.0


def elevation(lat: float, lon: float, zoom: int = 12) -> float:
    """Terrain elevation (m) at (lat, lon), bilinearly interpolated.

    The four pixels whose centres bracket the query point are blended
    with the standard bilinear weights; indices are clamped at tile
    edges, so points on a tile boundary interpolate across *two* tiles
    only through their shared edge pixels (good enough at 1/256 m
    quantization, and tile fetches are cached anyway).
    """
    x, y, fx, fy = _tile_pixel(lat, lon, zoom)
    tile = fetch_tile(zoom, x, y)
    # pixel centres at i+0.5 → local coordinate relative to centres
    u, v = fx - 0.5, fy - 0.5
    i0, j0 = math.floor(u), math.floor(v)
    tx, ty = u - i0, v - j0
    i0 = max(0, min(TILE_PX - 1, i0))
    j0 = max(0, min(TILE_PX - 1, j0))
    i1 = min(TILE_PX - 1, i0 + 1)
    j1 = min(TILE_PX - 1, j0 + 1)
    e00 = _terrarium_elev(tile, i0, j0)
    e10 = _terrarium_elev(tile, i1, j0)
    e01 = _terrarium_elev(tile, i0, j1)
    e11 = _terrarium_elev(tile, i1, j1)
    return ((1 - tx) * (1 - ty) * e00 + tx * (1 - ty) * e10
            + (1 - tx) * ty * e01 + tx * ty * e11)


# ------------------------------------------------------------ profiles

def _radio_horizon_m(h_tx: float, h_rx: float) -> float:
    """Geometric radio horizon (m) on a 4/3-earth: 4.12 km · (√h_tx + √h_rx)."""
    return 1000.0 * RADIO_HORIZON_K * (
        math.sqrt(max(float(h_tx), 0.0)) + math.sqrt(max(float(h_rx), 0.0))
    )


def _offset_east(lat: float, lon: float, dist_m: float = 1000.0) -> tuple[float, float]:
    """Displace (lat, lon) due east by dist_m. Used for coincident TX/RX."""
    return _offset_bearing(lat, lon, dist_m, 90.0)


def _offset_bearing(lat: float, lon: float, dist_m: float,
                    bearing_deg: float) -> tuple[float, float]:
    """Displace (lat, lon) by dist_m along a compass bearing (0° = N, 90° = E)."""
    br = math.radians(bearing_deg)
    dlat = (dist_m * math.cos(br)) / 111_320.0
    dlon = (dist_m * math.sin(br)) / (
        111_320.0 * max(0.2, math.cos(math.radians(lat)))
    )
    return float(lat) + dlat, float(lon) + dlon


def _haversine_m(lat0: float, lon0: float, lat1: float, lon1: float) -> float:
    p0, p1 = math.radians(lat0), math.radians(lat1)
    dp, dl = p1 - p0, math.radians(lon1 - lon0)
    a = math.sin(dp / 2) ** 2 + math.cos(p0) * math.cos(p1) * math.sin(dl / 2) ** 2
    return 2.0 * EARTH_R_M * math.asin(min(1.0, math.sqrt(a)))


def _to_vec(lat: float, lon: float) -> np.ndarray:
    p, l = math.radians(lat), math.radians(lon)
    return np.array([math.cos(p) * math.cos(l), math.cos(p) * math.sin(l),
                     math.sin(p)])


def path_profile(lat0: float, lon0: float, lat1: float, lon1: float,
                 n: int = 200, zoom: int = 12) -> dict:
    """Sample terrain along the great circle between two endpoints.

    Positions come from spherical linear interpolation of the endpoint
    unit vectors; distances are cumulative haversine (≡ arc fractions on
    a great circle).  Paths longer than ~200 km are rejected — the tile
    count (and fetch time) explodes.
    """
    total = _haversine_m(lat0, lon0, lat1, lon1)
    if total > MAX_PATH_M:
        raise ValueError(f"path {total / 1000:.0f} km exceeds the "
                         f"{MAX_PATH_M / 1000:.0f} km limit")
    a, b = _to_vec(lat0, lon0), _to_vec(lat1, lon1)
    omega = math.acos(float(np.clip(a @ b, -1.0, 1.0)))
    dist = np.linspace(0.0, total, n)
    elev = np.empty(n)
    for k in range(n):
        t = k / (n - 1) if n > 1 else 0.0
        if omega < 1e-12:
            v = a
        else:
            v = (math.sin((1 - t) * omega) * a
                 + math.sin(t * omega) * b) / math.sin(omega)
        lat = math.degrees(math.asin(float(np.clip(v[2], -1.0, 1.0))))
        lon = math.degrees(math.atan2(v[1], v[0]))
        elev[k] = elevation(lat, lon, zoom)
    return {"dist_m": dist, "elev_m": elev}


# ------------------------------------------------------------ geometry

def _fresnel_r1(lam: float, d1: float, d2: float) -> float:
    """First Fresnel zone radius √(λ·d₁·d₂/(d₁+d₂))."""
    s = d1 + d2
    return math.sqrt(lam * d1 * d2 / s) if s > 0 else 0.0


def _deygout_j(v: float) -> float:
    """Single knife-edge loss J(v) in dB (ITU-R P.526 approximation)."""
    if v <= V_THRESHOLD:
        return 0.0
    return 6.9 + 20.0 * math.log10(math.sqrt((v - 0.1) ** 2 + 1.0) + v - 0.1)


def _edge_scan(dist: np.ndarray, ground: np.ndarray, line: np.ndarray,
               lam: float, i0: int, i1: int) -> dict | None:
    """Dominant knife edge in the sub-path between sample indices i0, i1.

    Obstruction heights are measured above `line` (already the straight
    reference between the sub-path endpoints); returns the max-v edge or
    None when nothing sticks out / the sub-path has no interior points.
    """
    if i1 - i0 < 2:
        return None
    best = None
    for i in range(i0 + 1, i1):
        d1 = dist[i] - dist[i0]
        d2 = dist[i1] - dist[i]
        if d1 <= 0 or d2 <= 0:
            continue
        h_obs = ground[i] - line[i]
        v = h_obs * math.sqrt(2.0 * (d1 + d2) / (lam * d1 * d2))
        if best is None or v > best["v"]:
            best = {"dist_m": float(dist[i]), "h_m": float(h_obs),
                    "v": float(v), "loss_db": _deygout_j(v)}
    return best


def _analyze(dist_m: np.ndarray, elev_m: np.ndarray,
             tx_h: float, rx_h: float, f_hz: float,
             medium: str = "air") -> dict:
    """Pure-geometry link analysis on a sampled terrain profile.

    `dist_m`/`elev_m` are the along-path distance/terrain arrays,
    `tx_h`/`rx_h` the antenna heights above ground at the two ends.
    No network, no tile access — unit-testable on synthetic terrain.
    """
    dist = np.asarray(dist_m, dtype=np.float64)
    elev = np.asarray(elev_m, dtype=np.float64)
    n = dist.size
    lam = medium_params(f_hz, medium)["wavelength"]
    D = float(dist[-1] - dist[0])
    if D <= 0 or n < 3:
        raise ValueError("need a non-degenerate path with ≥ 3 samples")
    d = dist - dist[0]

    # 4/3-earth bulge added to terrain; LOS = chord between phase centres
    r_eff = K_EARTH * EARTH_R_M
    bulge = d * (D - d) / (2.0 * r_eff)
    ground = elev + bulge
    tx_abs = elev[0] + tx_h
    rx_abs = elev[-1] + rx_h
    line = tx_abs + (rx_abs - tx_abs) * d / D

    clearance = line - ground
    inner = clearance[1:-1]
    i_worst = int(np.argmin(inner)) + 1
    cmin = float(clearance[i_worst])

    d1 = d.copy()
    d2 = D - d
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = np.sqrt(np.where(d1 * d2 > 0, lam * d1 * d2 / (d1 + d2), 0.0))
    clear_frac = np.where(r1 > 0, clearance / np.maximum(r1, 1e-12), np.inf)
    min_frac = float(np.min(clear_frac[1:-1]))

    los = cmin > 0.0

    # ---- Deygout: dominant edge, then one secondary per sub-path ----
    main = _edge_scan(d, ground, line, lam, 0, n - 1)
    loss = 0.0
    edges = []
    if main is not None and main["v"] > V_THRESHOLD:
        loss += main["loss_db"]
        edges.append({**main, "kind": "main"})
        m = int(np.argmin(np.abs(d - main["dist_m"])))
        # left sub-path: TX → top of the main edge
        sub_line = tx_abs + (ground[m] - tx_abs) * (d[:m + 1] - d[0]) / max(d[m] - d[0], 1e-12)
        left = _edge_scan(d, ground, np.concatenate([sub_line, line[m + 1:]]),
                          lam, 0, m)
        # right sub-path: top of the main edge → RX
        denom = max(d[-1] - d[m], 1e-12)
        sub_line = ground[m] + (rx_abs - ground[m]) * (d[m:] - d[m]) / denom
        right = _edge_scan(d, ground, np.concatenate([line[:m], sub_line]),
                           lam, m, n - 1)
        for sub in (left, right):
            if sub is not None and sub["v"] > V_THRESHOLD:
                loss += sub["loss_db"]
                edges.append({**sub, "kind": "secondary"})

    if not los:
        verdict = "obstructed"
    elif min_frac < FRESNEL_CLEAR_FRAC:
        verdict = "Fresnel encroached"
    else:
        verdict = "LOS clear"

    # extra mast height to restore 0.6·r₁ clearance at every interior point
    need = FRESNEL_CLEAR_FRAC * r1
    deficit = need - clearance
    t = d / D
    dh_tx = 0.0
    dh_rx = 0.0
    for i in range(1, n - 1):
        if deficit[i] <= 0:
            continue
        if (1.0 - t[i]) > 0.05:
            dh_tx = max(dh_tx, float(deficit[i] / (1.0 - t[i])))
        if t[i] > 0.05:
            dh_rx = max(dh_rx, float(deficit[i] / t[i]))

    return {
        "los": bool(los),
        "verdict": verdict,
        "clearance_m": cmin,
        "min_fresnel_clearance": min_frac,
        "worst_point_dist_m": float(d[i_worst]),
        "diffraction_loss_db": float(loss),
        "edges": edges,
        "path_m": D,
        "wavelength_m": float(lam),
        "dist_m": [float(v) for v in dist],
        "elev_m": [float(v) for v in elev],
        "bulge_m": [float(v) for v in bulge],
        "los_line_m": [float(v) for v in line],
        "fresnel_r1_m": [float(v) for v in r1],
        "suggest_tx_h_m": float(tx_h + dh_tx),
        "suggest_rx_h_m": float(rx_h + dh_rx),
        "radio_horizon_m": _radio_horizon_m(tx_h, rx_h),
        "tx_horizon_m": _radio_horizon_m(tx_h, 0.0),
    }


# ------------------------------------------------------------ survey

def survey(tx: dict, rx: dict, f_mhz: float, p_tx_dbw: float = 0.0,
           g_tx_dbi: float = 2.15, g_rx_dbi: float = 2.15,
           medium: str = "air", n: int = 200, zoom: int = 12) -> dict:
    """Full virtual site survey between two sites.

    tx/rx: {"lat", "lon", "h_m"} — h_m is the antenna height above
    ground.  Returns a JSON-safe dict: terrain profile + bulge + LOS
    line + Fresnel radii, clearance metrics, Deygout diffraction loss,
    the `em_physics.link_budget` Friis budget, the diffraction-corrected
    received power, and a verdict ("LOS clear" / "Fresnel encroached" /
    "obstructed").
    """
    f_hz = f_mhz * 1e6
    tx = {"lat": float(tx["lat"]), "lon": float(tx["lon"]),
          "h_m": float(tx.get("h_m", 15.0))}
    rx = {"lat": float(rx["lat"]), "lon": float(rx["lon"]),
          "h_m": float(rx.get("h_m", tx["h_m"]))}
    hop = _haversine_m(tx["lat"], tx["lon"], rx["lat"], rx["lon"])
    offset_m = 0.0
    site_only = hop < MIN_PATH_M
    bearing_deg = None
    if site_only:
        # Don't fire a 1 km due-east hop into the nearest hillside and
        # call the site "obstructed".  Probe 8 short (400 m) bearings and
        # keep the clearest — that's the local placement suggestion.
        hop_m = 400.0
        best = None
        for az in range(0, 360, 45):
            rlat, rlon = _offset_bearing(tx["lat"], tx["lon"], hop_m, az)
            prof = path_profile(tx["lat"], tx["lon"], rlat, rlon, max(32, n // 4), zoom)
            try:
                geom = _analyze(prof["dist_m"], prof["elev_m"],
                                tx["h_m"], rx["h_m"], f_hz, medium)
            except ValueError:
                continue
            if best is None or geom["clearance_m"] > best[0]["clearance_m"]:
                best = (geom, rlat, rlon, az, hop_m)
        if best is None:
            rlat, rlon = _offset_bearing(tx["lat"], tx["lon"], hop_m, 90.0)
            rx = {"lat": rlat, "lon": rlon, "h_m": rx["h_m"]}
            offset_m = hop_m
            bearing_deg = 90.0
            prof = path_profile(tx["lat"], tx["lon"], rx["lat"], rx["lon"], n, zoom)
            geom = _analyze(prof["dist_m"], prof["elev_m"],
                            tx["h_m"], rx["h_m"], f_hz, medium)
        else:
            geom, rlat, rlon, az, hop_m = best
            rx = {"lat": rlat, "lon": rlon, "h_m": rx["h_m"]}
            offset_m = hop_m
            bearing_deg = float(az)
    else:
        prof = path_profile(tx["lat"], tx["lon"], rx["lat"], rx["lon"], n, zoom)
        geom = _analyze(prof["dist_m"], prof["elev_m"],
                        tx["h_m"], rx["h_m"], f_hz, medium)
    lb = link_budget(p_tx_dbw, g_tx_dbi, g_rx_dbi, geom["path_m"], f_hz, medium)
    received = lb["received_dbw"] - geom["diffraction_loss_db"]
    return {
        "tx": tx,
        "rx": rx,
        "rx_offset_m": offset_m,
        "site_only": site_only,
        "bearing_deg": bearing_deg,
        "f_mhz": float(f_mhz),
        "medium": medium,
        "zoom": int(zoom),
        **geom,
        "link_budget": {k: (float(v) if isinstance(v, (int, float)) else v)
                        for k, v in lb.items()},
        "received_dbw": float(received),
    }


def terrain_map(lat0: float, lon0: float, lat1: float, lon1: float,
                zoom: int = 12, size: int = 64,
                cache_dir: str | None = None) -> dict:
    """Local DEM patch around the two sites → {"elev", bbox, zoom, span_m}.

    Older versions returned a whole slippy-tile mosaic (often 10+ km on
    a side for a 1 km hop) and the UI min-max stretched it, so a flat
    field looked alpine.  This crops a *tight* geographic window — the
    axis-aligned bbox of the two points, padded by 15 % of the path
    (floor 200 m).  Tiles covering that window are fetched once and
    bilinearly resampled; row 0 is north.  `span_m` is the window's
    longer side — the 3-D view uses it as the metric horizontal scale.
    """
    path = _haversine_m(lat0, lon0, lat1, lon1)
    span = max(path, 400.0)
    pad = max(200.0, 0.15 * span)
    half = span / 2.0 + pad
    lat_c = 0.5 * (lat0 + lat1)
    lon_c = 0.5 * (lon0 + lon1)
    m_per_lat = 111_320.0
    m_per_lon = 111_320.0 * max(0.2, math.cos(math.radians(lat_c)))
    dlat = half / m_per_lat
    dlon = half / m_per_lon
    lat_lo, lat_hi = lat_c - dlat, lat_c + dlat
    lon_lo, lon_hi = lon_c - dlon, lon_c + dlon
    z = zoom
    if half > 20_000:
        z = min(z, 10)
    if half > 50_000:
        z = min(z, 8)
    if half > 100_000:
        z = min(z, 6)
    size = int(max(16, min(128, size)))

    x0, y0, _, _ = _tile_pixel(lat_hi, lon_lo, z)  # NW
    x1, y1, _, _ = _tile_pixel(lat_lo, lon_hi, z)  # SE
    xa, xb = min(x0, x1), max(x0, x1)
    ya, yb = min(y0, y1), max(y0, y1)
    rows, cols = (yb - ya + 1) * TILE_PX, (xb - xa + 1) * TILE_PX
    mosaic = np.empty((rows, cols), dtype=np.float64)
    for ty in range(ya, yb + 1):
        for tx_ in range(xa, xb + 1):
            t = fetch_tile(z, tx_, ty, cache_dir).astype(np.float64)
            e = t[:, :, 0] * 256.0 + t[:, :, 1] + t[:, :, 2] / 256.0 - 32768.0
            mosaic[(ty - ya) * TILE_PX:(ty - ya + 1) * TILE_PX,
                   (tx_ - xa) * TILE_PX:(tx_ - xa + 1) * TILE_PX] = e

    # fractional mosaic coordinates of each output sample
    n = 1 << z
    # global pixel of (lon, lat): same convention as _tile_pixel
    def _gpix(lat: float, lon: float) -> tuple[float, float]:
        lat = max(-MAX_LAT, min(MAX_LAT, lat))
        px = (lon + 180.0) / 360.0 * n * TILE_PX
        phi = math.radians(lat)
        py = (1.0 - math.asinh(math.tan(phi)) / math.pi) / 2.0 * n * TILE_PX
        return px - xa * TILE_PX, py - ya * TILE_PX

    lats = np.linspace(lat_hi, lat_lo, size)
    lons = np.linspace(lon_lo, lon_hi, size)
    elev = np.empty((size, size), dtype=np.float64)
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            fx, fy = _gpix(float(lat), float(lon))
            # bilinear in the mosaic (pixel centres at i+0.5)
            u, v = fx - 0.5, fy - 0.5
            i0 = int(math.floor(u)); j0 = int(math.floor(v))
            txw, tyw = u - i0, v - j0
            i0 = max(0, min(cols - 1, i0)); j0 = max(0, min(rows - 1, j0))
            i1 = min(cols - 1, i0 + 1); j1 = min(rows - 1, j0 + 1)
            elev[i, j] = ((1 - txw) * (1 - tyw) * mosaic[j0, i0]
                          + txw * (1 - tyw) * mosaic[j0, i1]
                          + (1 - txw) * tyw * mosaic[j1, i0]
                          + txw * tyw * mosaic[j1, i1])
    ji = int(np.argmax(elev))
    bi, bj = divmod(ji, size)
    return {
        "elev": elev,
        "lat_lo": float(lat_lo),
        "lat_hi": float(lat_hi),
        "lon_lo": float(lon_lo),
        "lon_hi": float(lon_hi),
        "zoom": int(z),
        "span_m": float(2.0 * half),
        "best_site": {
            "lat": float(lats[bi]),
            "lon": float(lons[bj]),
            "elev_m": float(elev[bi, bj]),
            "delta_m": float(elev[bi, bj] - elev[size // 2, size // 2]),
        },
    }


def _fetch_imagery_tile(z: int, x: int, y: int, cache_dir: str | None = None) -> np.ndarray:
    """Esri World Imagery slippy tile → (256, 256, 3) uint8. Cached."""
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/hoa64/imagery")
    path = os.path.join(cache_dir, str(z), str(x), f"{y}.png")
    if os.path.exists(path):
        with open(path, "rb") as f:
            blob = f.read()
        if blob[:8] == b"\x89PNG\r\n\x1a\n":
            return decode_png(blob)[:, :, :3]
        # leftover JPEG from an earlier cache write — fall through and rewrite
    data = None
    last_err = None
    for tmpl in IMAGERY_URLS:
        url = tmpl.format(z=z, x=x, y=y)
        req = urllib.request.Request(url, headers=_TILE_UA)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    if data is None:
        raise RuntimeError(f"imagery tile z={z} x={x} y={y}: {last_err}")
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        rgb = decode_png(data)[:, :, :3]
    else:
        # Esri World Imagery serves JPEG
        from io import BytesIO
        from PIL import Image
        rgb = np.asarray(Image.open(BytesIO(data)).convert("RGB"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(write_png(rgb))
    return rgb


def imagery_map(lat_lo: float, lat_hi: float, lon_lo: float, lon_hi: float,
                zoom: int = 15, size: int = 256,
                cache_dir: str | None = None) -> np.ndarray:
    """Full-colour satellite mosaic of a lat/lon window → (size, size, 3) uint8.

    Same slippy-map convention as `terrain_map` (row 0 = north).  Zoom is
    independent of the DEM zoom — imagery is fetched at a higher z so
    the 3-D texture has visible detail on a 1 km field.
    """
    z = int(max(3, min(17, zoom)))
    x0, y0, _, _ = _tile_pixel(lat_hi, lon_lo, z)
    x1, y1, _, _ = _tile_pixel(lat_lo, lon_hi, z)
    xa, xb = min(x0, x1), max(x0, x1)
    ya, yb = min(y0, y1), max(y0, y1)
    # cap tile count so a long hop doesn't pull 100 images
    while (xb - xa + 1) * (yb - ya + 1) > 16 and z > 3:
        z -= 1
        x0, y0, _, _ = _tile_pixel(lat_hi, lon_lo, z)
        x1, y1, _, _ = _tile_pixel(lat_lo, lon_hi, z)
        xa, xb = min(x0, x1), max(x0, x1)
        ya, yb = min(y0, y1), max(y0, y1)
    rows, cols = (yb - ya + 1) * TILE_PX, (xb - xa + 1) * TILE_PX
    mosaic = np.empty((rows, cols, 3), dtype=np.uint8)
    for ty in range(ya, yb + 1):
        for tx_ in range(xa, xb + 1):
            t = _fetch_imagery_tile(z, tx_, ty, cache_dir)
            mosaic[(ty - ya) * TILE_PX:(ty - ya + 1) * TILE_PX,
                   (tx_ - xa) * TILE_PX:(tx_ - xa + 1) * TILE_PX] = t
    n = 1 << z
    size = int(max(16, min(512, size)))
    rgb = np.empty((size, size, 3), dtype=np.uint8)
    for i, lat in enumerate(np.linspace(lat_hi, lat_lo, size)):
        for j, lon in enumerate(np.linspace(lon_lo, lon_hi, size)):
            lat = max(-MAX_LAT, min(MAX_LAT, float(lat)))
            px = (float(lon) + 180.0) / 360.0 * n * TILE_PX - xa * TILE_PX
            phi = math.radians(lat)
            py = ((1.0 - math.asinh(math.tan(phi)) / math.pi) / 2.0
                  * n * TILE_PX - ya * TILE_PX)
            ii = max(0, min(cols - 1, int(round(px))))
            jj = max(0, min(rows - 1, int(round(py))))
            rgb[i, j] = mosaic[jj, ii]
    return rgb


# ------------------------------------------------------------ self-check

if __name__ == "__main__":
    from .em_physics import medium_params as _mp

    F = 2.45e9
    lam = _mp(F)["wavelength"]

    # (a0) helpers — no network
    assert abs(_radio_horizon_m(15.0, 15.0) - 2 * 4120.0 * math.sqrt(15.0)) < 1.0
    elat, elon = _offset_east(52.445472, -2.597833, 1000.0)
    assert abs(_haversine_m(52.445472, -2.597833, elat, elon) - 1000.0) < 15.0
    print(f"(a0) horizon 15+15 m = {_radio_horizon_m(15, 15)/1000:.2f} km; "
          f"1 km east hop ok")

    # (a) flat 5 km path, 10 m masts → clear LOS, no diffraction
    npts = 200
    dist = np.linspace(0.0, 5000.0, npts)
    flat = np.zeros(npts)
    ra = _analyze(dist, flat, 10.0, 10.0, F)
    assert ra["los"], "flat path should be LOS"
    assert abs(ra["diffraction_loss_db"]) < 0.1, ra["diffraction_loss_db"]
    print(f"(a) flat 5 km @2.45 GHz h=10 m: los={ra['los']} "
          f"clearance={ra['clearance_m']:.2f} m "
          f"({ra['min_fresnel_clearance']:.2f}·r1) "
          f"L_diff={ra['diffraction_loss_db']:.2f} dB → {ra['verdict']}")

    # (b) same path with a 40 m Gaussian ridge at the midpoint
    ridge = 40.0 * np.exp(-((dist - 2500.0) / 200.0) ** 2)
    rb = _analyze(dist, ridge, 10.0, 10.0, F)
    assert not rb["los"], "ridge path should be obstructed"
    assert rb["diffraction_loss_db"] > 10.0, rb["diffraction_loss_db"]
    print(f"(b) +40 m ridge mid-path: los={rb['los']} "
          f"clearance={rb['clearance_m']:.2f} m "
          f"L_diff={rb['diffraction_loss_db']:.2f} dB → {rb['verdict']}")

    # (c) first Fresnel radius at the midpoint of a 10 km path:
    #     r1 = √(λ·d1·d2/(d1+d2)) with d1 = d2 = 5000 m → √(λ·2500) ≈ 17.5 m
    r1 = _fresnel_r1(lam, 5000.0, 5000.0)
    r1_ref = math.sqrt(lam * 2500.0)
    assert abs(r1 - r1_ref) / r1_ref < 0.01
    assert abs(r1 - 17.5) / 17.5 < 0.01
    print(f"(c) r1 midpoint 10 km @2.45 GHz: {r1:.3f} m "
          f"(λ={lam:.4f} m, √(λ·2500)={r1_ref:.3f} m)")

    # (d) NETWORK-gated: one real Terrarium tile over Europe/Atlantic
    try:
        tile = fetch_tile(3, 4, 5)
        elevs = np.array([_terrarium_elev(tile, i, j)
                          for j in range(0, TILE_PX, 16)
                          for i in range(0, TILE_PX, 16)])
        print(f"(d) tile z=3 x=4 y=5: {tile.shape} {tile.dtype}, "
              f"elevation {elevs.min():.1f} … {elevs.max():.1f} m "
              f"(1/16-stride sample)")
    except Exception as exc:  # noqa: BLE001 — offline is fine
        print(f"(d) SKIP network tile fetch: {exc}")

    print("site_survey self-check OK")
