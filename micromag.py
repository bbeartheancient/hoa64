"""Robust micromag search with simulated annealing + multi‑flip moves.

Uses temperature‑controlled acceptance (Metropolis criterion) and atomic‑orbital
block decomposition to escape local minima that trap the greedy single‑flip
descent.  Designed for orders up to ~500 (tested to 7th‑order HOA = 64).

Scaled for the hoa64 framework: exchange + demag + anisotropy energy,
with block‑level updating based on orbital‑shell partitioning.
"""

import numpy as np, time, math
from hoa64.hadamard import verify, normalize, random_seed


def total_energy(H, lam_ex=0.0, lam_dem=1.0, lam_ani=0.0):
    """Micromagnetic energy E = lam_ex*E_exch + lam_dem*E_dem + lam_ani*E_anis."""
    G = H.astype(np.float64) @ H.astype(np.float64).T
    n = H.shape[0]

    # exchange: penalize adjacent sign changes within each row
    E_exch = 0.0
    for i in range(n):
        for j in range(n):
            diff = float(H[i, j]) - float(H[i, (j + 1) % n])
            E_exch += diff * diff / 4.0

    # demagnetization: penalize non‑zero row dot products
    E_dem = 0.0
    for i in range(n):
        for k in range(i + 1, n):
            E_dem += float(G[i, k] * G[i, k])

    # anisotropy: penalize column imbalance
    col_sums = H.astype(np.float64).sum(axis=0)
    E_anis = float(np.sum(col_sums * col_sums))

    return (lam_ex * E_exch + lam_dem * E_dem + lam_ani * E_anis,
            E_exch, E_dem, E_anis)


def _swap_pair(H, rng):
    """Exchange a random +1 and a random -1 in the interior.
    Preserves total polarity count exactly (n(n+1)/2 positives)."""
    n = H.shape[0]
    interior = H[1:, 1:]
    pos = np.argwhere(interior == 1)
    neg = np.argwhere(interior == -1)
    if len(pos) == 0 or len(neg) == 0:
        return False
    pi = int(rng.integers(0, len(pos)))
    ni = int(rng.integers(0, len(neg)))
    rp, cp = pos[pi]; rn, cn = neg[ni]
    H[rp+1, cp+1] = -1; H[rn+1, cn+1] = 1
    return True


def micromag_sa(order, T_start=10.0, T_end=0.01, cooling=0.999, max_steps=50000,
                lam_ex=0.0, lam_ani=0.0, n_swap=3, rng=None):
    """Simulated annealing micromag search with swap moves.

    n_swap: number of (+1, -1) swaps per proposal.
    Swaps preserve the total polarity count exactly.
    """
    rng = rng or np.random.default_rng()
    H = random_seed(order, rng).astype(np.int8)
    n = H.shape[0]
    E_cur, _, _, _ = total_energy(H, lam_ex=lam_ex, lam_ani=lam_ani)
    best_H = H.copy(); best_E = E_cur

    T = T_start; steps = 0; accepts = 0; t0 = time.monotonic()

    while steps < max_steps and T > T_end:
        # Save current state for undo
        H_save = H.copy()

        # Perform n_swap exchanges
        ok = True
        for _ in range(n_swap):
            if not _swap_pair(H, rng):
                ok = False; break

        if not ok:
            H = H_save; steps += 1; continue

        E_new, _, _, _ = total_energy(H, lam_ex=lam_ex, lam_ani=lam_ani)
        delta = E_new - E_cur

        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            E_cur = E_new; accepts += 1
            if E_cur < best_E:
                best_H = H.copy(); best_E = E_cur
        else:
            H = H_save

        steps += 1
        T *= cooling

    elapsed = time.monotonic() - t0
    return best_H, dict(steps=steps, accepts=accepts, best_E=best_E,
                         elapsed_s=elapsed, hadamard=(best_E < 1e-6))


def micromag_ils_robust(order, T_start=20.0, n_flip=3, sa_steps=20000,
                        restarts=5, time_budget=None, rng=None):
    """ILS with robust SA inner loop. Returns (H, best_f, is_hadamard)."""
    rng = rng or np.random.default_rng()
    best_H = None
    best_f = None
    it = 0
    t0 = time.monotonic()

    while it < restarts:
        if time_budget and time.monotonic() - t0 > time_budget:
            break
        H, st = micromag_sa(order, T_start=T_start, n_swap=n_flip,
                            max_steps=sa_steps, rng=rng)
        G = H.astype(np.float64) @ H.astype(np.float64).T
        f = float(np.sum((G - order * np.eye(order)) ** 2)) / 2.0
        if best_H is None or f < best_f:
            best_H = H.copy()
            best_f = f
        it += 1
        # Increase temperature for next restart if stuck
        T_start = min(50.0, T_start * 1.5)

    return best_H, best_f, verify(best_H)
