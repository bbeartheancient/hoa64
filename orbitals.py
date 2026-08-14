"""Hydrogenic real orbitals on the package's SN3D spherical harmonics.

The hydrogen atom's stationary states factor into a radial and an
angular part,

    ψ_nlm(r, Ω) = R_nl(r) · Y_l^m(Ω),

with the radial wavefunction (Bohr radius a₀ = 1, nuclear charge Z)

    R_nl(r) = √[(2Z/n)³ (n−l−1)! / (2n (n+l)!)] · e^{−ρ/2} ρˡ
              · L^{2l+1}_{n−l−1}(ρ),          ρ = 2Zr/n,

L the associated Laguerre polynomials (evaluated here by their three-term
recurrence — no scipy).  Closed forms used as anchors:

    R_10 = 2 e^{−r},
    R_20 = (1/(2√2))(2 − r) e^{−r/2},
    R_21 = (1/(2√6)) r e^{−r/2},

and every R_nl satisfies ∫₀^∞ |R_nl|² r² dr = 1.

The angular part uses the package's real-spherical-harmonic convention
(Ambix ACN/SN3D, `hoa64.basis`): (l, m) maps to ACN channel
l(l+1)+m (`basis.acn_index`), i.e. the order-l block spans channels
l²..l²+2l.  Evaluation here is `_y_real_sn3d`, a vectorized replica of
`basis._sn3d_one` (same no-Condon–Shortley associated-Legendre
recurrence, same SN3D norm, same cos/sin azimuth split) — `basis`
itself loops per point and always builds all 64 channels, which is far
too slow for Monte-Carlo batches; the replica is pinned against
`basis.sh_sn3d_batch` in `selftest()`.  Two normalization caveats, both
immaterial here:

* SN3D harmonics are orthonormal under (1/4π)∫ dΩ, so Y_sn3d =
  √(4π)·Y_physics.  The overall scale of |ψ|² cancels in rejection
  sampling and in any display normalization.
* the basis omits the Condon–Shortley phase; for real orbitals that is a
  per-(l,m) sign at most, and |ψ|² doesn't see signs.

Axis conventions are the Ambix ones (+X front, +Y left, +Z up), so
channel (1, 0) ∝ z is the physics p_z, (1, 1) ∝ x is p_x, (1, −1) ∝ y.

`sample_orbital` rejection-samples |ψ|² inside the ball r ≤ extent with
extent defaulting to 2.5 n² bohr — comfortably past the mean radius
⟨r⟩ = (3n² − l(l+1))/2; the exponential tail beyond the ball is
truncated, which is invisible for visualization.  The proposal is a
uniform direction on S² times a uniform radius in [0, extent]: its 3D
density is ∝ 1/r², so the acceptance ratio is ∝ r²|ψ|² (the radial
probability density times the angular factor), which stays bounded and
keeps acceptance high even at n = 7, where a uniform-cube proposal
collapses like 1/extent³.  The bound C on r²|ψ|² is measured on a probe
batch with 25 % headroom; rare overshoots accept with probability 1.
"""

from __future__ import annotations

import math

import numpy as np

from .basis import MAX_ORDER


def _y_real_sn3d(dirs: np.ndarray, l: int, m: int) -> np.ndarray:
    """Vectorized SN3D real spherical harmonic for one (l, m).

    Same math as `basis._sn3d_one`: P_l^|m|(z) without Condon–Shortley
    phase (identical three-term recurrence), SN3D norm
    √(2(l−|m|)!/(l+|m|)!), m > 0 → cos(m·az), m < 0 → sin(|m|·az) with
    az = atan2(y, x).  `dirs` rows need not be unit (z is renormalized).
    Cross-checked against `basis.sh_sn3d_batch` in selftest().
    """
    x = dirs[..., 0]
    y = dirs[..., 1]
    z = dirs[..., 2]
    r = np.sqrt(x * x + y * y + z * z)
    r = np.where(r == 0.0, 1.0, r)
    z = z / r
    am = abs(m)
    st = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    if am == l:
        # P_l^l: P_n^n = (2n−1)·√(1−z²)·P_{n−1}^{n−1}
        P = np.ones_like(z)
        for n in range(1, l + 1):
            P = (2 * n - 1) * st * P
    else:
        Pmm = np.ones_like(z)
        for n in range(1, am + 1):
            Pmm = (2 * n - 1) * st * Pmm
        if am + 1 == l:
            P = (2 * am + 1) * z * Pmm  # P_{am+1}^{am}
        else:
            P_prev = Pmm
            P_cur = (2 * am + 1) * z * Pmm
            for n in range(am + 2, l + 1):
                P_prev, P_cur = (
                    P_cur,
                    ((2 * n - 1) * z * P_cur - (n + am - 1) * P_prev) / (n - am),
                )
            P = P_cur
    if m == 0:
        return P
    norm = math.sqrt(2.0 * math.factorial(l - am) / math.factorial(l + am))
    az = np.arctan2(y, x)
    return norm * P * (np.cos(am * az) if m > 0 else np.sin(am * az))


def _validate(n, l, m) -> tuple[int, int, int]:
    n, l, m = int(n), int(l), int(m)
    if not 1 <= n <= MAX_ORDER:
        raise ValueError(f"n must be in 1..{MAX_ORDER}, got {n}")
    if not 0 <= l < n:
        raise ValueError(f"l must be in 0..{n - 1} for n={n}, got {l}")
    if abs(m) > l:
        raise ValueError(f"|m| must be <= l={l}, got {m}")
    return n, l, m


def _assoc_laguerre(k: int, j: int, x: np.ndarray) -> np.ndarray:
    """Associated Laguerre L_j^k(x) by the three-term recurrence.

    (j+1) L_{j+1}^k = (2j + 1 + k − x) L_j^k − (j + k) L_{j−1}^k.
    """
    if j < 0:
        raise ValueError(f"j must be >= 0, got {j}")
    L0 = np.ones_like(x)
    if j == 0:
        return L0
    L1 = 1.0 + k - x
    if j == 1:
        return L1
    for i in range(2, j + 1):
        L0, L1 = L1, ((2 * i - 1 + k - x) * L1 - (i - 1 + k) * L0) / i
    return L1


def radial_wavefunction(r, n, l, Z=1) -> np.ndarray:
    """Hydrogenic radial wavefunction R_nl(r) (a₀ = 1); vectorized in r.

    Formula in the module docstring.  Unlike the orbital functions this
    is not capped at n ≤ 7 — it needs no spherical-harmonic tables.
    """
    if n < 1 or not 0 <= l < n:
        raise ValueError(f"need 0 <= l < n, got (n, l) = ({n}, {l})")
    r = np.asarray(r, dtype=np.float64)
    rho = 2.0 * Z * r / n
    pref = math.sqrt(
        (2.0 * Z / n) ** 3
        * math.factorial(n - l - 1)
        / (2.0 * n * math.factorial(n + l))
    )
    lag = _assoc_laguerre(2 * l + 1, n - l - 1, rho)
    return pref * np.exp(-0.5 * rho) * rho**l * lag


def default_extent(n: int) -> float:
    """Sampling/visualization half-side in bohr: 2.5 n² (module docstring)."""
    return 2.5 * n * n


def _psi2(n: int, l: int, m: int, xyz: np.ndarray) -> np.ndarray:
    """|ψ|² = (R_nl(r) · Y_l^m(Ω))² at (…, 3) points."""
    xyz = np.asarray(xyz, dtype=np.float64)
    r = np.linalg.norm(xyz, axis=-1)
    Y = _y_real_sn3d(xyz, l, m)
    R = radial_wavefunction(r, n, l)
    return (R * Y) ** 2


def orbital_grid(n, l, m, n_r=64, extent=None) -> dict:
    """|ψ|² on a cubic grid spanning [−extent, extent]³.

    Returns dict(density: (n_r, n_r, n_r) float64, extent: float).
    extent defaults to `default_extent(n)` = 2.5 n² bohr.
    """
    n, l, m = _validate(n, l, m)
    extent = float(extent) if extent is not None else default_extent(n)
    n_r = int(n_r)
    ax = np.linspace(-extent, extent, n_r)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    dens = _psi2(n, l, m, pts).reshape(n_r, n_r, n_r)
    return {"density": dens, "extent": extent}


def sample_orbital(n, l, m, n_samples=20000, seed=None) -> dict:
    """Rejection-sample |ψ|² in the ball r ≤ extent (scheme in docstring).

    Returns dict(points: (N, 3) float64 — Cartesian bohr coordinates,
    weights: (N,) — the 3D density |ψ|² at each point, normalized to
    [0, 1] for coloring, extent: float).
    """
    n, l, m = _validate(n, l, m)
    extent = default_extent(n)
    rng = np.random.default_rng(seed)
    n_samples = int(n_samples)
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    def draw(k: int):
        z = rng.uniform(-1.0, 1.0, size=k)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=k)
        st = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        dirs = np.stack([st * np.cos(phi), st * np.sin(phi), z], axis=1)
        r = rng.uniform(0.0, extent, size=k)
        return r, dirs

    def target(r: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        return r * r * _psi2(n, l, m, r[:, None] * dirs)

    pr, pd = draw(8192)
    C = float(target(pr, pd).max()) * 1.25
    if C <= 0.0:
        raise RuntimeError("probe found no density — cannot sample")

    out: list[np.ndarray] = []
    total = 0
    tries = 0
    while total < n_samples and tries < 500:
        tries += 1
        k = max(8192, 2 * (n_samples - total))
        r, dirs = draw(k)
        acc = rng.uniform(0.0, C, size=k) <= target(r, dirs)
        pts = (r[:, None] * dirs)[acc]
        out.append(pts)
        total += len(pts)
    if total < n_samples:
        raise RuntimeError(f"acceptance too low: {total}/{n_samples} after {tries} batches")
    pts = np.concatenate(out)[:n_samples]
    w = _psi2(n, l, m, pts)
    wmax = float(w.max())
    return {
        "points": pts,
        "weights": w / wmax if wmax > 0.0 else w,
        "extent": extent,
    }


def selftest() -> int:
    def expect(cond: bool, msg: str) -> None:
        if not cond:
            raise AssertionError(msg)

    # closed forms
    r = np.linspace(0.0, 30.0, 3001)
    expect(
        np.allclose(radial_wavefunction(r, 1, 0), 2.0 * np.exp(-r), atol=1e-12),
        "R_10 != 2 e^{-r}",
    )
    expect(
        np.allclose(
            radial_wavefunction(r, 2, 0),
            (2.0 - r) * np.exp(-r / 2.0) / (2.0 * math.sqrt(2.0)),
            atol=1e-12,
        ),
        "R_20 != (1/(2√2))(2−r) e^{−r/2}",
    )
    expect(
        np.allclose(
            radial_wavefunction(r, 2, 1),
            r * np.exp(-r / 2.0) / (2.0 * math.sqrt(6.0)),
            atol=1e-12,
        ),
        "R_21 != (1/(2√6)) r e^{−r/2}",
    )
    print("PASS radial R_10/R_20/R_21 closed forms")

    # angular replica pinned to the package SH implementation
    from .basis import acn_index, sh_sn3d_batch, unit_vector

    rng0 = np.random.default_rng(0)
    dirs = unit_vector(rng0.uniform(-180.0, 180.0, 64), rng0.uniform(-90.0, 90.0, 64))
    for ll, mm in [(0, 0), (1, -1), (1, 0), (1, 1), (2, -2), (2, 1), (3, -3),
                   (4, 3), (5, 0), (7, -7), (7, 6)]:
        ref = sh_sn3d_batch(dirs, max_order=ll)[..., acn_index(ll, mm)]
        got = _y_real_sn3d(dirs, ll, mm)
        expect(np.allclose(ref, got, atol=1e-10), f"Y_{ll}^{mm} vs basis mismatch")
    print("PASS _y_real_sn3d == basis.sh_sn3d_batch (11 channels)")

    # radial normalization ∫|R_nl|² r² dr = 1
    for nn, ll in [(1, 0), (2, 0), (2, 1), (3, 2), (4, 3)]:
        rr = np.linspace(0.0, 12.0 * nn * nn, 20001)
        R = radial_wavefunction(rr, nn, ll)
        norm = float(np.trapezoid(R * R * rr * rr, rr))
        expect(abs(norm - 1.0) < 1e-4, f"R_{nn}{ll} norm {norm} != 1")
    print("PASS radial normalization (5 states)")

    # grid shape / validation
    g = orbital_grid(2, 1, 0, n_r=16)
    expect(g["density"].shape == (16, 16, 16), "orbital_grid density not 16³")
    expect((g["density"] >= 0.0).all() and g["density"].max() > 0.0, "grid density bad")
    expect(g["extent"] == default_extent(2), "orbital_grid extent")
    for bad in [(2, 2, 0), (3, 1, 2), (8, 0, 0)]:
        try:
            orbital_grid(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"orbital_grid{bad} should raise ValueError")
    print("PASS orbital_grid shape/validation")

    # sampling: shapes, determinism, anisotropy
    s_a = sample_orbital(2, 1, 0, n_samples=4000, seed=1)
    s_b = sample_orbital(2, 1, 0, n_samples=4000, seed=1)
    expect(np.array_equal(s_a["points"], s_b["points"]), "seeded determinism")
    p = s_a["points"]
    expect(p.shape == (4000, 3), "sample points not (4000, 3)")
    expect(s_a["weights"].shape == (4000,) and s_a["extent"] > 0.0, "weights/extent")
    mz, mx = float(np.abs(p[:, 2]).mean()), float(np.abs(p[:, 0]).mean())
    expect(mz > 1.25 * mx, f"2p_z not z-concentrated: |z| {mz:.3f} vs |x| {mx:.3f}")
    s_x = sample_orbital(2, 1, 1, n_samples=4000, seed=1)
    q = s_x["points"]
    mx2, mz2 = float(np.abs(q[:, 0]).mean()), float(np.abs(q[:, 2]).mean())
    expect(mx2 > 1.25 * mz2, f"2p_x not x-concentrated: |x| {mx2:.3f} vs |z| {mz2:.3f}")
    print(
        f"PASS p-orbital anisotropy (2p_z |z|/|x| = {mz / mx:.2f}, "
        f"2p_x |x|/|z| = {mx2 / mz2:.2f})"
    )
    print("orbitals selftest: all checks passed")
    return 0


if __name__ == "__main__":
    import time

    t0 = time.monotonic()
    s = sample_orbital(3, 2, 0, n_samples=20000, seed=42)
    print(
        f"sampled 3d (n=3 l=2 m=0): 20000 pts in {time.monotonic() - t0:.2f}s "
        f"— extent {s['extent']:.0f} bohr"
    )
    selftest()
