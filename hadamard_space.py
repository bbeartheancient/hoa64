"""Hadamard space — transmuting Hadamard matrices into hyperbolic 3-space.

A *Hadamard manifold* is a complete, simply-connected Riemannian manifold
of non-positive sectional curvature (the CAT(0) models); the canonical
3-dimensional one is hyperbolic space ℍ³.  The pun this module cashes in:
the rows of a Hadamard matrix H are mutually orthogonal ±1 vectors, so
every pair of distinct rows is exactly ‖rᵢ − rⱼ‖ = √(2n) apart — they are
the vertices of a **regular (n−1)-simplex**.  "Transmutation" embeds that
simplex (or the n×n entry grid) into ℍ³ and renders it in the two
classical models.

Models and curvature conventions
--------------------------------
We work in the **unit Poincaré ball** {|y| < 1} ⊂ ℝ³ throughout.  The
curvature −κ metric on it is the conformally-scaled ball metric

    ds² = (4/κ) |dy|² / (1 − |y|²)²,

i.e. the textbook metric 4|dx|²/(1−κ|x|²)² on the ball of radius 1/√κ
pulled back by y = √κ·x.  Three consequences we exploit:

* geodesics as *point sets* are κ-independent (constant metric rescaling
  changes only arc length, not the geodesic equations) — they are the
  Euclidean circles orthogonal to the boundary sphere, plus diameters;
* arc length at fixed ball coordinates scales by κ^(−1/2);
* κ → 0 flattens the metric toward the Euclidean one — `to_poincare`
  exposes a κ = 0 "flat display mode" (no radial warp) as the honest
  Euclidean limit, and `hyperbolic_dist` there returns the plain chord.

Distances.  `hyperbolic_dist(p, q, kappa)` implements the standard
Poincaré formula verbatim,

    d(p, q) = arcosh( 1 + 2κ|p−q|² / ((1−κ|p|²)(1−κ|q|²)) ),

for points in the κ-ball (radius 1/√κ).  Our pipeline keeps unit-ball
points, so it is called with κ = 1; the curvature-−κ distance is then
d/√κ (transmute's stats do this).  In the unit ball: d(p, q) ≥ |p−q|,
with equality only at the origin — the CAT(0) "stretch" the UI reports.

Geodesics.  For p, q not collinear with the origin, the geodesic is the
arc through p, q of the Euclidean circle orthogonal to the unit sphere.
Orthogonality forces the centre c to satisfy |c|² = R² + 1; membership of
p and q then gives the linear conditions

    c·p = (1 + |p|²)/2,      c·q = (1 + |q|²)/2,

solved in span(p, q) as a 2×2 system c = αp + βq, with radius
R = √(|c|² − 1).  The sample arc is the unique p→q arc staying inside
the ball (the other arc of the same circle exits through the boundary;
we detect and flip).  Collinear p, q (or an endpoint at the origin) give
the diameter chord.

Embeddings
----------
`pca_embed` centers the rows and takes the top-3 SVD scores.  The raw
singular spectrum of H is perfectly flat — HHᵀ = nI ⇒ all σᵢ = √n — so
there is no distinguished 3-frame: the top-3 projection is an
arbitrary-but-canonical 3-shadow of the simplex (LAPACK's deterministic
choice, sign-canonicalized per component; centering perturbs the flat
spectrum only through the row sums, e.g. Sylvester's first row).

`to_poincare` normalizes radii to [0, r_max] and warps radially,

    |y| = tanh(√κ·ρ/2),   ρ ∈ [0, r_max],

which is exactly the ball radius of a point at hyperbolic distance ρ
from the origin (d = (2/√κ)·artanh(|y|)); κ = 0 skips the warp.

`lattice_embed` maps the n×n entry grid (scaled into the disc of radius
r_max ≤ 0.95) through the same warp and lifts the Poincaré disc to the
upper sheet of the two-sheeted hyperboloid x² + y² − z² = −1 — itself ℍ²,
a 2D Hadamard manifold — via x = 2y₁/(1−|y|²), y = 2y₂/(1−|y|²),
z = (1+|y|²)/(1−|y|²).  Raw coordinates are returned (z ≥ 1);
`transmute` rescales z into a unit-ish range for display.
"""

from __future__ import annotations

import math

import numpy as np

from .hadamard import hadamard_known

R_MAX = 0.9  # default display radius (normalized hyperbolic radius cap)


# ---------------------------------------------------------------- embeddings

def pca_embed(H, k: int = 3) -> np.ndarray:
    """Row coordinates from the top-k SVD scores of the row-centered H.

    Flat-spectrum caveat in the module docstring.  Deterministic: LAPACK
    SVD plus a per-component sign canonicalization (largest-|entry|
    positive).  Rows (not columns) are the simplex vertices.  Returns an
    (n, k) array; k is clamped to the matrix size with zero-padding.
    """
    H = np.asarray(H, dtype=np.float64)
    n = H.shape[0]
    kk = min(int(k), n)
    Hc = H - H.mean(axis=1, keepdims=True)
    U, S, _ = np.linalg.svd(Hc, full_matrices=False)
    pts = U[:, :kk] * S[:kk]
    for j in range(kk):  # sign canonicalization for cross-run stability
        i = int(np.argmax(np.abs(pts[:, j])))
        if pts[i, j] < 0:
            pts[:, j] = -pts[:, j]
    if kk < int(k):
        pts = np.hstack([pts, np.zeros((n, int(k) - kk))])
    return pts


def to_poincare(points, kappa: float = 1.0, r_max: float = R_MAX) -> np.ndarray:
    """Normalize radii to [0, r_max] and apply the hyperbolic radial warp.

    |y| = tanh(√κ·ρ/2) for κ > 0 — ball radius of hyperbolic radius ρ.
    κ = 0 is the flat display limit: no warp (|y| = ρ).  All outputs lie
    strictly inside the unit ball (tanh < 1).
    """
    pts = np.asarray(points, dtype=np.float64)
    r = np.linalg.norm(pts, axis=-1, keepdims=True)
    rmax = float(r.max())
    rho = r / rmax if rmax > 0 else r  # [0, 1]
    rho = rho * float(r_max)
    if kappa > 0:
        rad = np.tanh(math.sqrt(kappa) * rho / 2.0)
    else:
        rad = rho
    direction = pts / np.where(r > 0, r, 1.0)
    return direction * rad


def hyperbolic_dist(p, q, kappa: float = 1.0):
    """Poincaré distance arcosh(1 + 2κ|p−q|²/((1−κ|p|²)(1−κ|q|²))).

    Inputs are points of the κ-ball (|x| < 1/√κ); for this module's
    unit-ball pipeline call with κ = 1 (curvature-−κ arc length then
    scales by 1/√κ — see module docstring).  κ ≤ 0 returns the Euclidean
    chord (flat display mode).  Broadcasts over leading dimensions.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    d2 = np.sum((p - q) ** 2, axis=-1)
    if kappa <= 0:
        return np.sqrt(d2)
    p2 = np.sum(p * p, axis=-1)
    q2 = np.sum(q * q, axis=-1)
    arg = 1.0 + 2.0 * kappa * d2 / ((1.0 - kappa * p2) * (1.0 - kappa * q2))
    return np.arccosh(np.maximum(arg, 1.0))


def geodesic(p, q, kappa: float = 1.0, n_pts: int = 32) -> np.ndarray:
    """Poincaré-ball geodesic p → q as (n_pts, 3) points, endpoints exact.

    Orthogonal-circle arc (2×2 centre solve in span(p, q)); diameter
    chord when p, q, 0 are collinear.  κ is accepted for API symmetry but
    does not change the point set — constant metric rescaling moves no
    geodesic (module docstring).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    t = np.linspace(0.0, 1.0, int(n_pts))
    pn = float(np.linalg.norm(p))
    qn = float(np.linalg.norm(q))
    if pn < 1e-15 or qn < 1e-15 or np.linalg.norm(np.cross(p, q)) < 1e-12 * pn * qn:
        return p[None, :] + t[:, None] * (q - p)[None, :]

    pp, qq, pq = float(p @ p), float(q @ q), float(p @ q)
    A = np.array([[pp, pq], [pq, qq]])
    b = np.array([(1.0 + pp) / 2.0, (1.0 + qq) / 2.0])
    alpha, beta = np.linalg.solve(A, b)
    c = alpha * p + beta * q
    R = math.sqrt(max(float(c @ c) - 1.0, 0.0))

    e1 = p / pn
    e2 = q - (q @ e1) * e1
    e2 /= np.linalg.norm(e2)
    vp, vq = p - c, q - c
    tp = math.atan2(float(vp @ e2), float(vp @ e1))
    tq = math.atan2(float(vq @ e2), float(vq @ e1))
    sweep = (tq - tp + math.pi) % (2.0 * math.pi) - math.pi  # wrap to (−π, π]

    def arc(dth: float) -> np.ndarray:
        ang = tp + t * dth
        return c[None, :] + R * (
            np.cos(ang)[:, None] * e1[None, :] + np.sin(ang)[:, None] * e2[None, :]
        )

    pts = arc(sweep)
    if float(np.linalg.norm(pts, axis=1).max()) > 1.0 + 1e-9:
        # wrong way around: the inside arc is the one staying in the ball
        pts = arc(sweep - math.copysign(2.0 * math.pi, sweep))
    pts[0] = p
    pts[-1] = q
    return pts


def lattice_embed(H, kappa: float = 1.0, r_max: float = 0.95) -> dict:
    """The n×n entry grid lifted to the hyperboloid model of ℍ².

    Grid (i, j) → square coords (u, v) ∈ [−1, 1]² → disc radius
    r = √(u²+v²)/√2 · r_max → Poincaré warp |y| = tanh(√κ·r/2) →
    hyperboloid lift (2y₁, 2y₂, 1+|y|²)/(1−|y|²).  Returns dict(verts:
    (n, n, 3) RAW hyperboloid coordinates (z ≥ 1), values: (n, n) ±1).
    """
    H = np.asarray(H)
    n = H.shape[0]
    ax = np.linspace(-1.0, 1.0, n)
    U, V = np.meshgrid(ax, ax, indexing="ij")
    r = np.sqrt(U * U + V * V) / math.sqrt(2.0) * float(r_max)
    if kappa > 0:
        rad = np.tanh(math.sqrt(kappa) * r / 2.0)
    else:
        rad = r
    ang = np.arctan2(V, U)
    y1 = rad * np.cos(ang)
    y2 = rad * np.sin(ang)
    denom = 1.0 - rad * rad
    verts = np.stack(
        [2.0 * y1 / denom, 2.0 * y2 / denom, (1.0 + rad * rad) / denom], axis=-1
    )
    return {"verts": verts, "values": np.asarray(H, dtype=np.int8)}


# ---------------------------------------------------------------- orchestrator

def transmute(order_or_H, mode: str = "rows", kappa: float = 1.0,
              geodesics: bool = True, max_points: int = 256) -> dict:
    """Transmute a Hadamard matrix into ℍ³ display data.

    mode="rows": the regular simplex of rows as a Poincaré-ball point
    cloud; geodesics connect each point to its 4 nearest neighbours in
    the hyperbolic metric (cap 2·n_pts segments).  mode="lattice": the
    entry grid as a hyperboloid surface (z rescaled to unit range).
    Returns dict(mode, points|verts, colors, geodesics, stats).
    """
    if isinstance(order_or_H, (int, np.integer)):
        H = hadamard_known(int(order_or_H))
        if H is None:
            raise ValueError(f"no known Hadamard matrix of order {order_or_H}")
    else:
        H = np.asarray(order_or_H, dtype=np.int8)
    n = int(H.shape[0])
    if mode not in ("rows", "lattice"):
        raise ValueError(f"mode must be 'rows' or 'lattice', got {mode!r}")

    if mode == "lattice":
        lat = lattice_embed(H, kappa=kappa)
        verts = lat["verts"]
        zmax = float(verts[..., 2].max())
        verts = verts / zmax  # display: z ∈ [1/zmax, 1]
        values = lat["values"]
        return {
            "mode": mode,
            "order": n,
            "verts": verts,
            "colors": ((values + 1) // 2).astype(np.int8),
            "geodesics": [],
            "stats": {
                "kappa": float(kappa),
                "z_scale": zmax,
                "note": "hyperboloid lift of the entry grid; z rescaled by 1/z_max",
            },
        }

    # rows mode
    m = min(int(max_points), n)
    Hs = H[:m]  # any row subset is still a regular simplex
    pts = to_poincare(pca_embed(Hs, 3), kappa=kappa)
    colors = (Hs[:, 0] > 0).astype(np.int8)  # sign of first component (gauge)

    edges: list[tuple[int, int]] = []
    if geodesics and m >= 2:
        D = hyperbolic_dist(pts[:, None, :], pts[None, :, :], kappa=1.0)
        np.fill_diagonal(D, np.inf)
        k_near = min(4, m - 1)
        nbrs = np.argsort(D, axis=1)[:, :k_near]
        seen = set()
        for i in range(m):
            for j in nbrs[i]:
                e = (min(i, int(j)), max(i, int(j)))
                if e not in seen:
                    seen.add(e)
                    edges.append(e)
        edges = edges[: 2 * m]

    geos = [geodesic(pts[i], pts[j], n_pts=16) for i, j in edges]
    chords = [float(np.linalg.norm(pts[i] - pts[j])) for i, j in edges]
    hdists = [float(hyperbolic_dist(pts[i], pts[j], kappa=1.0)) for i, j in edges]
    sqk = math.sqrt(kappa) if kappa > 0 else 1.0
    return {
        "mode": mode,
        "order": n,
        "points": pts,
        "colors": colors,
        "geodesics": geos,
        "stats": {
            "kappa": float(kappa),
            "n_points": m,
            "n_geodesics": len(geos),
            "mean_hyperbolic_dist": float(np.mean(hdists)) / sqk if hdists else 0.0,
            "mean_euclidean_chord": float(np.mean(chords)) if chords else 0.0,
            "note": "row simplex in the Poincaré ball; d_ℍ = d_unit/√κ",
        },
    }


def selftest() -> int:
    def expect(cond: bool, msg: str) -> None:
        if not cond:
            raise AssertionError(msg)

    from .hadamard import sylvester

    H = sylvester(8)
    n = 8

    # the rows form a regular simplex: all pairwise distances √(2n)
    D = [np.linalg.norm(H[i] - H[j]) for i in range(n) for j in range(i + 1, n)]
    expect(all(abs(d - math.sqrt(2 * n)) < 1e-9 for d in D), "rows not a regular simplex")
    print(f"PASS regular simplex (28 pairs all at √(2n) = {math.sqrt(2 * n):.3f})")

    # pca determinism + shape
    P1, P2 = pca_embed(H), pca_embed(H)
    expect(P1.shape == (n, 3) and np.array_equal(P1, P2), "pca_embed not deterministic")
    print("PASS pca_embed deterministic (8, 3)")

    # to_poincare: strictly inside the ball; κ=0 is the flat identity warp
    Y = to_poincare(P1, kappa=1.0)
    rn = np.linalg.norm(Y, axis=1)
    expect(float(rn.max()) < 1.0, "poincaré points not inside the ball")
    Y0 = to_poincare(P1, kappa=0.0)
    rho = np.linalg.norm(P1, axis=1) / np.linalg.norm(P1, axis=1).max() * R_MAX
    expect(np.allclose(np.linalg.norm(Y0, axis=1), rho), "κ=0 warp not identity")
    expect(np.allclose(rn, np.tanh(rho / 2), atol=1e-12), "κ=1 warp formula off")
    print(f"PASS to_poincare (max |y| {rn.max():.4f} < 1, κ=0 identity, tanh warp)")

    # geodesics: exact endpoints, stay in the ball, inside-arc norm bound
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        p = Y[rng.integers(0, n)]
        q = Y[rng.integers(0, n)]
        g = geodesic(p, q, n_pts=32)
        expect(np.allclose(g[0], p) and np.allclose(g[-1], q), "geodesic endpoints off")
        gn = np.linalg.norm(g, axis=1)
        expect(float(gn.max()) <= 1.0 + 1e-9, "geodesic left the ball")
        # inside arc bulges toward the origin: never past the endpoint norms
        expect(
            float(gn.max()) <= max(np.linalg.norm(p), np.linalg.norm(q)) + 1e-9,
            "geodesic arc overshoots endpoint radius",
        )
        worst = max(worst, float(gn.max()) / max(np.linalg.norm(p), np.linalg.norm(q), 1e-12))
    # diameters too
    g = geodesic(Y[0], -Y[0], n_pts=32)
    expect(np.allclose(np.cross(g[10], Y[0]), 0.0, atol=1e-9), "diameter not straight")
    print(f"PASS geodesics (200 arcs in-ball; max |x|/max(|p|,|q|) = {worst:.4f})")

    # distances: symmetric, ≥ euclidean chord (CAT(0) stretch), flat mode
    p, q = Y[0], Y[3]
    d1 = float(hyperbolic_dist(p, q, kappa=1.0))
    d2 = float(hyperbolic_dist(q, p, kappa=1.0))
    ch = float(np.linalg.norm(p - q))
    expect(abs(d1 - d2) < 1e-12, "hyperbolic_dist not symmetric")
    expect(d1 >= ch - 1e-12, "hyperbolic_dist < chord")
    expect(abs(float(hyperbolic_dist(p, q, kappa=0.0)) - ch) < 1e-12, "κ=0 not chord")
    print(f"PASS hyperbolic_dist (symmetric; d_ℍ {d1:.3f} ≥ chord {ch:.3f}; κ=0 chord)")

    # lattice: finite, raw z ≥ 1, values ±1
    lat = lattice_embed(sylvester(16), kappa=1.0)
    v = lat["verts"]
    expect(v.shape == (16, 16, 3) and np.isfinite(v).all(), "lattice verts bad")
    expect(float(v[..., 2].min()) >= 1.0, "hyperboloid z < 1")
    expect(set(np.unique(lat["values"]).tolist()) <= {-1, 1}, "lattice values not ±1")
    print(f"PASS lattice_embed (16² verts, z ∈ [{v[..., 2].min():.2f}, {v[..., 2].max():.2f}])")

    # orchestrator: both modes
    t = transmute(16, mode="rows", kappa=1.0, geodesics=True)
    expect(t["points"].shape == (16, 3) and len(t["geodesics"]) > 0, "transmute rows bad")
    t2 = transmute(16, mode="lattice", kappa=1.0)
    expect(t2["verts"].shape == (16, 16, 3) and t2["verts"][..., 2].max() <= 1.0 + 1e-9,
           "transmute lattice bad")
    try:
        transmute(3)
    except ValueError:
        pass
    else:
        raise AssertionError("transmute(3) should raise ValueError")
    print("PASS transmute rows/lattice + bad order raises")
    print("hadamard_space selftest: all checks passed")
    return 0


if __name__ == "__main__":
    import time

    t0 = time.monotonic()
    t = transmute(64, mode="rows", kappa=1.0)
    s = t["stats"]
    print(
        f"transmute(64, rows) in {time.monotonic() - t0:.2f}s — "
        f"{s['n_geodesics']} geodesics, mean d_ℍ {s['mean_hyperbolic_dist']:.3f} "
        f"vs chord {s['mean_euclidean_chord']:.3f}"
    )
    selftest()
