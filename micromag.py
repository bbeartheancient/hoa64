"""Micromagnetic Hadamard descent — vectorized with O(1) per-candidate deltas.

Energy:  E = lam_ex*E_exch + lam_dem*E_demag + lam_ani*E_anis

Single-flip delta at cell (i,j) from sign s to -s:
  dE_exch = (neighbor left change) + (neighbor right change)   [O(1)]
  dE_dem  = -4*s*M[i,j] + 4*(n-1)                              [O(1) via M]
  dE_anis = -4*s*col_sum[j] + 4                               [O(1)]

Same incremental Gram/M update as local_search in hadamard.py.
"""

import numpy as np, time
from hoa64.hadamard import verify, random_seed


def _make_M(H, G, n):
    return (G - n * np.eye(n, dtype=np.float64)) @ H.astype(np.float64)


def _apply_flip(H, HT, G, M, col_sum, i, j, n):
    s = float(H[i, j])
    oldrow = G[i].copy()
    H[i, j] = -1 if s > 0 else 1
    HT[j, i] = H[i, j]
    G[i] = H[i].astype(np.float64) @ HT
    G[:, i] = G[i]
    M[i] = G[i] @ H.astype(np.float64) - n * H[i].astype(np.float64)
    kk = np.arange(n) != i
    v = H[kk, j].astype(np.float64)
    M[kk, :] += np.outer(-2.0 * s * v, H[i].astype(np.float64))
    M[kk, j] += -2.0 * s * oldrow[kk] + 4.0 * v
    col_sum[j] += -2.0 * s
    return H, HT, G, M, col_sum


def micromag_search(H0, max_flips=100000, lam_ex=0.01, lam_ani=0.1,
                    min_gain=0.0):
    n = H0.shape[0]
    H = np.array(H0, dtype=np.int8)
    HT = H.T.copy()
    G = H.astype(np.float64) @ HT
    M = (G - n * np.eye(n, dtype=np.float64)) @ H.astype(np.float64)
    col_sum = H.astype(np.float64).sum(axis=0)
    flips = 0; t0 = time.monotonic()

    while flips < max_flips:
        # exchange delta: dE_exch = lam_ex * s * (H_left + H_right)
        Hleft = np.roll(H, 1, axis=1)   # shifted right: col j becomes old col j-1
        Hright = np.roll(H, -1, axis=1)  # shifted left
        D_exch = lam_ex * H.astype(np.float64) * (Hleft.astype(np.float64) + Hright.astype(np.float64))

        # demagnetization delta
        D_dem = -4.0 * H.astype(np.float64) * M + 4.0 * (n - 1)

        # anisotropy delta
        D_ani = lam_ani * (-4.0 * H.astype(np.float64) * col_sum[None, :] + 4.0)

        D = D_exch + D_dem + D_ani
        Dint = D[1:, 1:]
        idx = int(np.argmin(Dint))
        dmin = float(Dint.ravel()[idx])

        if dmin >= -min_gain:
            break
        i = 1 + idx // (n - 1)
        j = 1 + idx % (n - 1)
        H, HT, G, M, col_sum = _apply_flip(H, HT, G, M, col_sum, i, j, n)
        flips += 1
    return H, dict(flips=flips, elapsed_s=time.monotonic() - t0)


def micromag_ils(n, lam_ex=0.01, lam_ani=0.1, inner_flips=20000,
                 outer_iters=20, time_budget=None, rng=None):
    rng = rng or np.random.default_rng()
    best_H = None; best_f = None; it = 0; t0 = time.monotonic()
    while it < outer_iters:
        if time_budget and time.monotonic() - t0 > time_budget: break
        if best_H is not None and it > 0:
            H0 = best_H.copy()
            kp = max(1, int(0.05 * (n := H0.shape[0]) * (n - 1)))
            interior = (n - 1)
            cells = rng.choice(interior * interior, size=kp, replace=False)
            rows = cells // interior + 1
            cols = cells % interior + 1
            H0[rows, cols] *= -1
        else:
            H0 = random_seed(n, rng).astype(np.int8)
        H, st = micromag_search(H0, max_flips=inner_flips,
                                lam_ex=lam_ex, lam_ani=lam_ani)
        G = H.astype(np.float64) @ H.astype(np.float64).T
        f = float(np.sum((G - n * np.eye(n)) ** 2)) / 2.0
        if best_H is None or f < best_f:
            best_H = H.copy(); best_f = f
        it += 1
    return best_H, best_f, verify(best_H)
