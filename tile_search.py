"""Tile‑based micromag search — operates on 2×2 H2 cells, not individual bits.

In a normalized Hadamard matrix, every 2×2 submatrix (rows i,i+1; cols j,j+1)
must be an H2 cell up to equivalence: either [[+,+],[+,-]] or [[-,-],[-,+]].

Flipping an entire 2×2 tile (all 4 signs) preserves the H2‑cell structure,
making SA proposals far more efficient than random single‑bit flips.
"""

import numpy as np, time, math
from hoa64.hadamard import verify, random_seed

# The two H2 cell types (up to row/column sign flips)
H2_TYPE_A = np.array([[1, 1], [1, -1]], dtype=np.int8)
H2_TYPE_B = np.array([[-1, -1], [-1, 1]], dtype=np.int8)


def total_energy_fast(H):
    """Fast energy: just the demagnetization (F = sum dot²)."""
    G = H.astype(np.float64) @ H.astype(np.float64).T
    n = H.shape[0]
    E = 0.0
    for i in range(n):
        for k in range(i + 1, n):
            E += float(G[i, k] * G[i, k])
    return E


def _swap_random(H, rng):
    """Exchange a random +1 and a random -1 in the interior.
    Preserves the total count of each polarity exactly."""
    n = H.shape[0]
    interior = H[1:, 1:]
    pos = np.argwhere(interior == 1)
    neg = np.argwhere(interior == -1)
    if len(pos) == 0 or len(neg) == 0:
        return False
    pi = int(rng.integers(0, len(pos)))
    ni = int(rng.integers(0, len(neg)))
    rp, cp = pos[pi]; rn, cn = neg[ni]
    rp += 1; cp += 1; rn += 1; cn += 1  # back to full indices
    H[rp, cp] = -1; H[rn, cn] = 1
    return True


def tile_sa_swap(order, T_start=20.0, T_end=0.01, cooling=0.9995,
                 max_steps=20000, rng=None):
    """Tile‑based SA using swap moves (preserves polarity count).

    Each proposal: pick two random interior tiles, flip both simultaneously.
    This preserves the total +1/−1 count and the H2 cell structure.
    """
    rng = rng or np.random.default_rng()
    H = random_seed(order, rng).astype(np.int8)
    n = H.shape[0]
    E_cur = total_energy_fast(H)
    best_H = H.copy(); best_E = E_cur

    T = T_start; steps = 0; accepts = 0; t0 = time.monotonic()

    while steps < max_steps and T > T_end:
        # pick two different random interior tiles and flip both
        i1 = int(rng.integers(1, n - 1)); j1 = int(rng.integers(1, n - 1))
        i2 = int(rng.integers(1, n - 1)); j2 = int(rng.integers(1, n - 1))
        if i1 == i2 and j1 == j2:
            continue

        H[i1, j1] *= -1; H[i1, j1+1] *= -1; H[i1+1, j1] *= -1; H[i1+1, j1+1] *= -1
        H[i2, j2] *= -1; H[i2, j2+1] *= -1; H[i2+1, j2] *= -1; H[i2+1, j2+1] *= -1

        E_new = total_energy_fast(H)
        delta = E_new - E_cur

        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            E_cur = E_new; accepts += 1
            if E_cur < best_E:
                best_H = H.copy(); best_E = E_cur
        else:
            H[i1, j1] *= -1; H[i1, j1+1] *= -1; H[i1+1, j1] *= -1; H[i1+1, j1+1] *= -1
            H[i2, j2] *= -1; H[i2, j2+1] *= -1; H[i2+1, j2] *= -1; H[i2+1, j2+1] *= -1

        steps += 1
        T *= cooling

    return best_H, dict(steps=steps, accepts=accepts, best_E=best_E,
                         elapsed_s=time.monotonic()-t0, hadamard=(best_E < 1e-6))


def tile_ils(order, T_start=20.0, cooling=0.9995, sa_steps=15000,
             restarts=5, time_budget=None, rng=None):
    """ILS with tile‑based SA inner loop."""
    rng = rng or np.random.default_rng()
    best_H = None; best_f = None; it = 0; t0 = time.monotonic()
    while it < restarts:
        if time_budget and time.monotonic() - t0 > time_budget: break
        H, st = tile_sa_swap(order, T_start=T_start, cooling=cooling,
                        max_steps=sa_steps, rng=rng)
        G = H.astype(np.float64) @ H.astype(np.float64).T
        f = float(np.sum((G - order * np.eye(order)) ** 2)) / 2.0
        if best_H is None or f < best_f:
            best_H = H.copy(); best_f = f
        it += 1
        T_start = min(30.0, T_start * 1.3)
    return best_H, best_f, verify(best_H)
