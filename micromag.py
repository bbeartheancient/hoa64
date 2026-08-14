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
    Hf = H.astype(np.float64)
    G = Hf @ Hf.T
    n = H.shape[0]

    # exchange: penalize adjacent sign changes within each row.
    # diff²/4 is 1 on a broken bond, 0 otherwise — the sum is exact in f64.
    diff = Hf - np.roll(Hf, -1, axis=1)
    E_exch = float(np.sum(diff * diff)) / 4.0

    # demagnetization: penalize non‑zero row dot products.  G is symmetric
    # with G[i,i] = n, so Σ_{i<k} G[i,k]² = (‖G‖²_F − n³)/2 — again exact
    # (integer arithmetic below 2⁵³ for any n ≤ 1024).
    E_dem = float((np.sum(G * G) - n ** 3) / 2.0)

    # anisotropy: penalize column imbalance
    col_sums = Hf.sum(axis=0)
    E_anis = float(np.sum(col_sums * col_sums))

    return (lam_ex * E_exch + lam_dem * E_dem + lam_ani * E_anis,
            E_exch, E_dem, E_anis)


def site_energy(H, lam_ex=0.0, lam_dem=1.0, lam_anis=0.0):
    """Per‑site (i, j) energy density — the local contribution of each
    matrix entry to the micromagnetic energy, as an n×n float map.

    Demagnetization (the Hadamard term): with G = H Hᵀ and
    M = (G − nI) H, the entry‑wise density

        dem_map[i, j] = ½ · H[i, j] · M[i, j]

    sums exactly to the demag energy, because
    Σᵢⱼ H[i,j]·M[i,j] = tr(Hᵀ (G − nI) H) = ‖G‖²_F − n³ = 2·E_dem.
    It also ranks flip candidates: the Gram‑objective change for flipping
    entry (i, j) is dF(i, j) = −4·H[i,j]·M[i,j] + 4(n−1)
                       = −8·dem_map[i, j] + 4(n−1),
    so bright (large positive) sites are precisely the entries whose flip
    most reduces the energy — the same dF formula `local_search` uses.

    Exchange: each horizontal bond b(i, j) = (H[i,j] − H[i,j+1])²/4 (with
    wraparound) is split evenly between its two endpoint sites.
    Anisotropy: each column's col_sum² is distributed uniformly over its
    n entries.  All three maps sum to their respective total_energy part.
    """
    H = np.asarray(H, dtype=np.float64)
    n = H.shape[0]
    G = H @ H.T
    M = (G - n * np.eye(n)) @ H
    dem = 0.5 * H * M

    diff = H - np.roll(H, -1, axis=1)
    bonds = diff * diff / 4.0                      # bond (j, j+1) at [i, j]
    exch = 0.5 * (bonds + np.roll(bonds, 1, axis=1))

    col_sq = H.sum(axis=0) ** 2
    anis = np.broadcast_to(col_sq / n, (n, n))

    return lam_ex * exch + lam_dem * dem + lam_anis * anis


def energy_gradient(H):
    """Per‑entry |dF| magnitude map of the demag (Gram) objective.

    dF(i, j) = −4·H[i,j]·M[i,j] + 4(n−1) is the exact change in
    f = (‖G‖²_F − n³)/2 when entry (i, j) is flipped; negative dF means
    an improving flip.  Returns |dF| — the "gradient distillation" view:
    dark sites are already optimal, bright sites carry the remaining
    orthogonality defect.
    """
    H = np.asarray(H, dtype=np.float64)
    n = H.shape[0]
    G = H @ H.T
    M = (G - n * np.eye(n)) @ H
    dF = -4.0 * H * M + 4.0 * (n - 1)
    return np.abs(dF)


def flux_map(H) -> np.ndarray:
    """Polarity flux — Ising/micromagnetic domain‑wall density, as an n×n grid.

    Each nearest‑neighbour bond carries an exchange flux
    H[i,j]·H[i,j′] ∈ {+1, −1}: +1 means aligned spins (no wall), −1 a
    broken bond (a +1/−1 domain wall crosses the bond — flux flows
    across these walls, driven by the demag gradient).  The per‑site
    wall density combines the two outgoing bonds (horizontal and
    vertical, cyclic/toroidal boundary like `total_energy`'s E_exch):

        W[i, j] = (2 − H[i,j]·H[i,j+1] − H[i,j]·H[i+1,j]) / 4

    so W = 0 where both bonds are aligned (inside a domain), ½ at a
    single broken bond, and 1 where both are broken (wall corner).
    Summing over the grid counts walls: Σ W = (B_h + B_v)/2 with B_h,
    B_v the numbers of broken horizontal/vertical bonds.  In a perfect
    Hadamard matrix adjacent rows are orthogonal, so the wall pattern
    is the local fingerprint of the remaining exchange frustration.
    """
    H = np.asarray(H, dtype=np.int8)
    hb = H * np.roll(H, -1, axis=1)  # horizontal bond flux H[i,j]·H[i,j+1]
    vb = H * np.roll(H, -1, axis=0)  # vertical bond flux   H[i,j]·H[i+1,j]
    return (2.0 - hb.astype(np.float64) - vb.astype(np.float64)) / 4.0


def _e_goal(corr, n, lam_goal):
    """Goal-attraction energy: lam_goal per entry disagreeing with ±goal."""
    return lam_goal * (n * n - corr) / 2.0


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
                lam_ex=0.0, lam_ani=0.0, n_swap=3, rng=None, start=None,
                goal=None, lam_goal=0.5,
                callback=None, stop_flag=None, live_params=None):
    """Simulated annealing micromag search with swap moves.

    n_swap: number of (+1, -1) swaps per proposal.
    Swaps preserve the total polarity count exactly.
    start: optional initial ±1 matrix (default: `random_seed(order)`).

    Goal attraction: when `goal` (a ±1 matrix of the same shape) is given,
    an extra energy term

        E_goal = lam_goal · (n² − corr) / 2,   corr = Σ H ⊙ goal

    is added — i.e. lam_goal per entry disagreeing with the target.  Since
    H and −H are both Hadamard (global sign is gauge), the term targets
    whichever of ±goal the start matrix is nearer to; that sign is fixed
    once at the start.  E_goal = 0 exactly at H = ±goal, so the combined
    landscape keeps its global minima on the Hadamard set — this is a
    *guided* anneal (gradient distillation toward a known solution).
    Keep lam_goal in a weak-coupling regime (≲ the demag scale): a large
    lam_goal collapses the search onto the target and stops exploration.

    Optional streaming hooks (zero behavior change when omitted):
    - callback(dict): called every 500 steps with
      {"step", "T", "E", "best_E", "accepts", "H"} — H is the current
      best ±1 matrix (for live previews; never serialized).  When a goal
      is active the dict also carries "E_goal" (current goal term) and
      "goal_agree" (fraction of entries matching ±goal).
    - stop_flag (threading.Event): checked every 500 steps; break early
      when set, returning the current best.
    - live_params (dict): mid-run retuning — every 500 steps the keys
      "cooling" / "lam_ex" / "lam_ani" / "lam_goal" are read and applied
      when present and not None (a WebSocket peer can mutate the dict
      while SA runs).
    """
    rng = rng or np.random.default_rng()
    if start is None:
        H = random_seed(order, rng).astype(np.int8)
    else:
        H = np.array(start, dtype=np.int8)
    n = H.shape[0]
    if goal is not None:
        goal = np.asarray(goal, dtype=np.int8)
        if goal.shape != H.shape:
            raise ValueError(
                f"goal shape {goal.shape} does not match order {n}")
        corr0 = int((H.astype(np.int64) * goal.astype(np.int64)).sum())
        G = goal if corr0 >= 0 else -goal   # target the nearer of ±goal
    else:
        G = None
    E_cur, _, _, _ = total_energy(H, lam_ex=lam_ex, lam_ani=lam_ani)
    corr_cur = int((H.astype(np.int64) * G.astype(np.int64)).sum()) \
        if G is not None else 0
    best_H = H.copy(); best_E = E_cur
    best_tot = E_cur + _e_goal(corr_cur, n, lam_goal)

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
            H = H_save; steps += 1
            # no proposal happened — T holds, but the stop flag must still
            # be observed or a persistently-failing swap loop runs forever
            if (stop_flag is not None and steps % 500 == 0
                    and stop_flag.is_set()):
                break
            continue

        E_new, _, _, _ = total_energy(H, lam_ex=lam_ex, lam_ani=lam_ani)
        delta = E_new - E_cur
        if G is not None:
            corr_new = int((H.astype(np.int64) * G.astype(np.int64)).sum())
            delta += _e_goal(corr_new, n, lam_goal) \
                   - _e_goal(corr_cur, n, lam_goal)

        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            E_cur = E_new; accepts += 1
            if G is not None:
                corr_cur = corr_new
            tot = E_cur + _e_goal(corr_cur, n, lam_goal)
            if tot < best_tot:
                best_H = H.copy(); best_E = E_cur; best_tot = tot
        else:
            H = H_save

        steps += 1
        T *= cooling

        if steps % 500 == 0:
            if live_params is not None:
                c = live_params.get("cooling")
                if c is not None:
                    cooling = float(c)
                lx = live_params.get("lam_ex")
                if lx is not None:
                    lam_ex = float(lx)
                la = live_params.get("lam_ani")
                if la is not None:
                    lam_ani = float(la)
                lg = live_params.get("lam_goal")
                if lg is not None:
                    lam_goal = float(lg)
                    if G is not None:
                        # re-score the incumbent under the new coupling,
                        # else a pre-retune best stays stuck
                        corr_best = int(
                            (best_H.astype(np.int64)
                             * G.astype(np.int64)).sum())
                        best_tot = best_E + _e_goal(corr_best, n, lam_goal)
            if callback is not None:
                d = {"step": steps, "T": T, "E": E_cur,
                     "best_E": best_E, "accepts": accepts, "H": best_H,
                     "E_goal": _e_goal(corr_cur, n, lam_goal)
                     if G is not None else 0.0}
                if G is not None:
                    d["goal_agree"] = (n * n + corr_cur) / (2.0 * n * n)
                callback(d)
            if stop_flag is not None and stop_flag.is_set():
                break

    elapsed = time.monotonic() - t0
    info = dict(steps=steps, accepts=accepts, best_E=best_E,
                elapsed_s=elapsed, hadamard=(best_E < 1e-6))
    if G is not None:
        corr_best = int((best_H.astype(np.int64) * G.astype(np.int64)).sum())
        info["E_goal"] = _e_goal(corr_best, n, lam_goal)
        info["goal_agree"] = (n * n + corr_best) / (2.0 * n * n)
    return best_H, info


def micromag_ils_robust(order, T_start=20.0, n_flip=3, sa_steps=20000,
                        restarts=5, time_budget=None, rng=None, stop_flag=None,
                        progress_callback=None):
    """ILS with robust SA inner loop. Returns (H, best_f, is_hadamard).

    stop_flag (threading.Event, optional): checked at each restart
    boundary — break out early and return the current best when set.
    progress_callback (optional): called once per restart with
    {"iter", "f", "best_f", "elapsed_s"} and forwarded to the inner SA
    (per-500-step frames) so long descents stream live progress.
    """
    rng = rng or np.random.default_rng()
    best_H = None
    best_f = None
    it = 0
    t0 = time.monotonic()

    while it < restarts:
        if stop_flag is not None and stop_flag.is_set():
            break
        if time_budget and time.monotonic() - t0 > time_budget:
            break
        H, st = micromag_sa(order, T_start=T_start, n_swap=n_flip,
                            max_steps=sa_steps, rng=rng,
                            callback=progress_callback, stop_flag=stop_flag)
        G = H.astype(np.float64) @ H.astype(np.float64).T
        f = float(np.sum((G - order * np.eye(order)) ** 2)) / 2.0
        if best_H is None or f < best_f:
            best_H = H.copy()
            best_f = f
        if progress_callback is not None:
            progress_callback({"iter": it, "f": f, "best_f": best_f,
                               "elapsed_s": time.monotonic() - t0})
        it += 1
        # Increase temperature for next restart if stuck
        T_start = min(50.0, T_start * 1.5)

    return best_H, best_f, (best_H is not None and verify(best_H))
