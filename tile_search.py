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


def tile_sa(order, T_start=20.0, T_end=0.01, cooling=0.9995,
            max_steps=20000, rng=None):
    """Tile‑based simulated annealing.

    Each proposal: pick a random 2×2 interior tile and flip all 4 cells.
    This is equivalent to swapping an H2 type A ↔ type B locally.
    """
    rng = rng or np.random.default_rng()
    H = random_seed(order, rng).astype(np.int8)
    n = H.shape[0]
    E_cur = total_energy_fast(H)
    best_H = H.copy(); best_E = E_cur

    T = T_start; steps = 0; accepts = 0; t0 = time.monotonic()

    while steps < max_steps and T > T_end:
        # pick random 2×2 tile in interior (avoid row0/col0 border)
        i = int(rng.integers(1, n - 1))  # top row of tile
        j = int(rng.integers(1, n - 1))  # left col of tile

        # flip all 4 cells
        H[i, j] *= -1; H[i, j + 1] *= -1
        H[i + 1, j] *= -1; H[i + 1, j + 1] *= -1

        E_new = total_energy_fast(H)
        delta = E_new - E_cur

        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            E_cur = E_new; accepts += 1
            if E_cur < best_E:
                best_H = H.copy(); best_E = E_cur
        else:
            H[i, j] *= -1; H[i, j + 1] *= -1
            H[i + 1, j] *= -1; H[i + 1, j + 1] *= -1

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
        H, st = tile_sa(order, T_start=T_start, cooling=cooling,
                        max_steps=sa_steps, rng=rng)
        G = H.astype(np.float64) @ H.astype(np.float64).T
        f = float(np.sum((G - order * np.eye(order)) ** 2)) / 2.0
        if best_H is None or f < best_f:
            best_H = H.copy(); best_f = f
        it += 1
        T_start = min(30.0, T_start * 1.3)
    return best_H, best_f, verify(best_H)
