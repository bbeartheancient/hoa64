#!/usr/bin/env python3
"""Aggressive Hadamard search daemon — tries ALL search engines on every order.

Unlike evolve.py (which only tries known constructions), this daemon
actively SEARCHES for new matrices using every available engine:
  - micromag descent (exchange + demag + anisotropy)
  - GS circulant FFT search
  - Williamson FFT search
  - Gerzon AB cell SA (H₂ prior in WXYZ)
  - Sudoku-style row solvers (backtrack / overlay / CSP / DLX / residuals)
  - pure max‑det descent

Runs with longer time budgets per order, designed for overnight operation.
Any matrix found is saved, verified, and auto‑cascaded.
"""

import os, sys, time, math
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hoa64.hadamard import verify, normalize, hadamard_known, hadamard_product
from hoa64.rh import rh_check

OUT = Path.home() / "open_hadamard"
OUT.mkdir(parents=True, exist_ok=True)


def export_csv(order, H):
    path = OUT / f"hadamard_{order}.csv"
    np.savetxt(str(path), H.astype(np.int8), delimiter=",", fmt="%d")
    return path


def known_orders():
    known = set()
    for p in OUT.glob("hadamard_*.csv"):
        try:
            known.add(int(p.stem.replace("hadamard_", "")))
        except ValueError:
            continue
    return known


def try_micromag(order, budget=60):
    from hoa64.micromag import micromag_ils_robust
    H, _, ok = micromag_ils_robust(order, T_start=15.0, n_flip=3,
                                    sa_steps=5000, restarts=5,
                                    time_budget=budget,
                                    rng=np.random.default_rng())
    if ok and H is not None and verify(H):
        return normalize(H), "micromag"
    return None, None


def try_gs_circulant(order, budget=60):
    if order % 4 != 0:
        return None, None
    k = order // 4
    from hoa64.williamson import gs_circulant_ils
    H, _, _, ok = gs_circulant_ils(k, inner_flips=5000, outer_iters=10,
                                    time_budget=budget,
                                    rng=np.random.default_rng())
    if ok and H is not None:
        return normalize(H), "gs_circulant"
    return None, None


def try_williamson(order, budget=60):
    if order % 4 != 0:
        return None, None
    k = order // 4
    from hoa64.williamson import williamson_ils
    H, _, _, ok = williamson_ils(k, inner_flips=5000, outer_iters=10,
                                  time_budget=budget,
                                  rng=np.random.default_rng())
    if ok and H is not None:
        return normalize(H), "williamson"
    return None, None


def try_maxdet(order, budget=60):
    from hoa64.hadamard import local_search, random_seed
    H = random_seed(order).astype(np.int8)
    Hf, st = local_search(H, max_flips=50000, min_gain=4)
    if st['f'] == 0 and verify(Hf):
        return normalize(Hf), "maxdet"
    return None, None


def try_tile(order, budget=60):
    from hoa64.tile_search import tile_ils
    H, _, ok = tile_ils(order, T_start=20.0, sa_steps=5000,
                         restarts=5, time_budget=budget,
                         rng=np.random.default_rng())
    if ok and H is not None:
        from hoa64.hadamard import normalize
        return normalize(H), "tile_swap"
    return None, None


def try_gerzon(order, budget=60):
    from hoa64.gerzon import gerzon_ils
    from hoa64.hadamard import normalize
    H, _, ok = gerzon_ils(order, T_start=20.0, sa_steps=5000,
                          restarts=5, time_budget=budget,
                          rng=np.random.default_rng())
    if ok and H is not None:
        return normalize(H), "gerzon"
    return None, None


def try_holographic(order, budget=60):
    from hoa64.holographic import holo_ils
    from hoa64.hadamard import normalize
    H, _, ok = holo_ils(order, T_start=20.0, sa_steps=5000,
                        restarts=5, time_budget=budget,
                        rng=np.random.default_rng())
    if ok and H is not None:
        return normalize(H), "holographic"
    return None, None


def try_crown(order, budget=60):
    from hoa64.crown import crown_ils
    from hoa64.hadamard import normalize
    H, _, ok = crown_ils(order, T_start=20.0, sa_steps=5000,
                         restarts=5, time_budget=budget,
                         rng=np.random.default_rng())
    if ok and H is not None:
        return normalize(H), "crown"
    return None, None


def try_brillouin(order, budget=60):
    from hoa64.brillouin import bzf_ils
    from hoa64.hadamard import normalize
    H, _, ok = bzf_ils(order, T_start=20.0, sa_steps=5000,
                       restarts=5, time_budget=budget,
                       rng=np.random.default_rng())
    if ok and H is not None:
        return normalize(H), "brillouin"
    return None, None


def try_sudoku(order, budget=60):
    from hoa64.sudoku import sudoku_ils
    from hoa64.hadamard import normalize
    H, _, ok = sudoku_ils(order, T_start=20.0, sa_steps=5000,
                          restarts=5, time_budget=budget,
                          rng=np.random.default_rng())
    if ok and H is not None:
        return normalize(H), "sudoku"
    return None, None


ENGINES = [
    ("micromag", try_micromag),
    ("tile_swap", try_tile),
    ("gerzon", try_gerzon),
    ("holographic", try_holographic),
    ("crown", try_crown),
    ("brillouin", try_brillouin),
    ("sudoku", try_sudoku),
    ("gs_circulant", try_gs_circulant),
    ("williamson", try_williamson),
    ("maxdet", try_maxdet),
]


def cascade(order, known):
    """Generate Kronecker multiples of a newly-found order."""
    found = []
    for mult in [2, 4, 8]:
        c = order * mult
        if c in known:
            continue
        H = hadamard_known(c)
        if H is not None and verify(H):
            path = export_csv(c, H)
            known.add(c)
            found.append(c)
            print(f"  cascade H({c}) = {mult}x{order} → {path}", flush=True)
    return found


def main():
    os.system("clear" if sys.platform != "win32" else "cls")
    print("Hadamard Search Daemon — aggressive search mode")
    print("Press Ctrl-C to stop\n", flush=True)

    known = known_orders()
    print(f"Starting from {len(known)} known orders, max H({max(known)})", flush=True)
    rh = rh_check(N=10)
    print(f"RH bound={rh['bound']:.0f}\n", flush=True)

    cycle = 0
    total_found = 0

    # Search targets: all multiples of 4 from the first gap upward
    search_from = min(n for n in range(4, 10000, 4) if n not in known)

    try:
        while True:
            cycle += 1
            # Pick the next few untried orders (skip those already in known)
            targets = [n for n in range(search_from, search_from + 200, 4)
                       if n not in known]

            if not targets:
                search_from += 200
                continue

            cycle_found = 0
            budget_per = max(30, min(120, 30 + cycle // 10))

            for order in targets[:3]:  # search 3 orders per cycle
                if order in known:
                    continue

                for name, engine in ENGINES:
                    t0 = time.monotonic()
                    H, method = engine(order, budget=budget_per)
                    dt = time.monotonic() - t0

                    if H is not None and order not in known:
                        path = export_csv(order, H)
                        known.add(order)
                        cycle_found += 1
                        total_found += 1
                        print(f"  [{cycle:04d}] H({order:5d}) via {method:>15s} "
                              f"→ {path} ({dt:.1f}s)", flush=True)

                        # cascade
                        cascade_n = cascade(order, known)
                        cycle_found += len(cascade_n)
                        total_found += len(cascade_n)
                        search_from = max(search_from, order + 4)
                        break  # move to next order after finding one

                if order in known:
                    continue

            if cycle_found == 0 and cycle % 20 == 0:
                gaps_below = sum(1 for n in range(4, min(max(known)+1, 4000), 4)
                                 if n not in known)
                print(f"  [{cycle:04d}] no finds. known={len(known)} "
                      f"max=H({max(known)}) searching from {search_from} "
                      f"gaps≤4000={gaps_below}", flush=True)

    except KeyboardInterrupt:
        print(f"\nStopped. {total_found} found this session. "
              f"Total known: {len(known)}", flush=True)


if __name__ == "__main__":
    main()
