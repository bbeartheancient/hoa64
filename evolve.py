#!/usr/bin/env python3
"""Hadamard gap‑filling daemon — targets only unknown orders.

Loads all known CSV matrices from ~/open_hadamard, plus caches the toolchain's
constructible set.  Each cycle scans for new gaps and attempts to fill them
using available constructions (Paley, Paley‑pp, Sylvester, Kronecker, Miyamoto,
CSV import).  Exports any newly‑found matrices to CSV.
"""

import os, sys, time, math, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hoa64.hadamard import (hadamard_known, hadamard_orders, normalize, verify,
                             _is_prime_power, hadamard_product, paley, sylvester)
from hoa64.rh import rh_check

OUT = Path.home() / "open_hadamard"


def load_known(max_n=4000):
    """Return set of orders for which we have a verified matrix."""
    known = set()
    for p in sorted(OUT.glob("hadamard_*.csv")):
        try:
            order = int(p.stem.replace("hadamard_", ""))
            if order <= max_n:
                known.add(order)
        except ValueError:
            continue
    return known


def export_csv(order, H):
    path = OUT / f"hadamard_{order}.csv"
    np.savetxt(str(path), H.astype(np.int8), delimiter=",", fmt="%d")
    return path


def try_build(order, known_set):
    H = hadamard_known(order)
    if H is not None and verify(H):
        return normalize(H), "cached"

    q = order - 1
    if q >= 3 and q % 4 == 3 and _is_prime_power(q):
        from hoa64.hadamard import _paley_from_q
        H = _paley_from_q(q, is_type_ii=False)
        if H is not None and verify(H):
            return normalize(H), "PaleyI"

    q2 = order // 2 - 1
    if order % 2 == 0 and q2 >= 5 and q2 % 4 == 1 and _is_prime_power(q2):
        from hoa64.hadamard import _paley_from_q
        H = _paley_from_q(q2, is_type_ii=True)
        if H is not None and verify(H):
            return normalize(H), "PaleyII"

    for d in range(2, int(math.isqrt(order)) + 1):
        if order % d: continue
        e = order // d
        A = hadamard_known(d)
        B = hadamard_known(e)
        if A is not None and B is not None:
            H = hadamard_product(A, B)
            if verify(H):
                return normalize(H), f"kron({d},{e})"

    p = OUT / f"hadamard_{order}.csv"
    if p.is_file():
        H = np.loadtxt(str(p), delimiter=",", dtype=np.int8)
        Hn = normalize(H)
        if verify(Hn):
            return Hn, "CSV"

    return None, None


def find_gaps(known, max_n):
    return sorted(n for n in range(4, max_n + 1, 4) if n not in known)


def main():
    os.system("clear" if sys.platform != "win32" else "cls")
    print("Hadamard Gap‑Filling Daemon", flush=True)
    print("Press Ctrl-C to stop\n", flush=True)

    # Load RNN model for guided search
    rnn_model = None
    try:
        import torch
        from hoa64.rnn_hadamard import HadamardRNN
        FEAT_DIM = 110
        rnn_model = HadamardRNN(FEAT_DIM, hidden_dim=128, num_layers=2)
        rnn_pt = OUT / "rnn_hadamard.pt"
        if rnn_pt.is_file():
            rnn_model.load_state_dict(torch.load(str(rnn_pt), map_location="cpu"))
            rnn_model.eval()
            print("RNN model loaded for guided search", flush=True)
    except Exception as e:
        print(f"RNN not available: {e}", flush=True)

    max_scan = 4000
    known = load_known(max_scan)
    print(f"Loaded {len(known)} known orders from CSV", flush=True)

    # seed cache
    _ = hadamard_orders(200)

    cycle = 0
    try:
        while True:
            cycle += 1
            gaps = find_gaps(known, max_scan)
            if not gaps:
                print(f"\n  [{cycle:03d}] No gaps ≤ {max_scan}. Advancing.", flush=True)
                max_scan += 200
                continue

            hi_known = max(known)
            filled = 0

            # Scan a few gaps per cycle (ascending)
            print(f"  [{cycle:03d}] scanning {len(gaps[:5])} gaps...", end=" ", flush=True)
            for gap in gaps[:5]:
                H, method = try_build(gap, known)
                if H is not None and gap not in known:
                    path = export_csv(gap, H)
                    known.add(gap)
                    filled += 1
                    print(f"  [{cycle:03d}] H({gap:5d}) via {method:>20s} → {path}",
                          flush=True)

            if filled > 0:
                continue

            # RNN‑guided micromag search on hardest gaps
            if rnn_model is not None and cycle % 3 == 0:
                for gap in gaps[:3]:
                    if gap > 500: continue  # RNN search too slow for large orders
                    from hoa64.rnn_hadamard import rnn_guided_search
                    H, best_f = rnn_guided_search(
                        gap, rnn_model, n_trials=5, search_flips=2000)
                    from hoa64.hadamard import verify
                    if H is not None and verify(H):
                        path = export_csv(gap, H)
                        known.add(gap)
                        filled += 1
                        print(f"  [{cycle:03d}] H({gap:5d}) via RNN+micromag → {path}",
                              flush=True)
                        break

            if filled > 0:
                continue

            # Nothing found: report
            print(f"  [{cycle:03d}] No fills. Remaining gaps: {len(gaps)} "
                  f" max-known={hi_known}. "
                  f"First gaps: {gaps[:8]}", flush=True)
            rh = rh_check(N=10)
            print(f"         RH bound={rh['bound']:.0f}", flush=True)
            time.sleep(5)

    except KeyboardInterrupt:
        print(f"\nStopped. {len(known)} known, max H({max(known)}), "
              f"gaps={len(find_gaps(known, max_scan))}", flush=True)


if __name__ == "__main__":
    main()
