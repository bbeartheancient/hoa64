#!/usr/bin/env python3
"""Matrix‑pixel visualizer — animate Hadamard growth with green/black cells.

Selectable algorithm, evolving from H.1 to the limit of what the algorithm
can produce.  Each matrix is rendered as a grid of text‑mode pixels:
  █ = +1 (green)    ░ = -1 (black)

Algorithms: Sylvester, PaleyI, PaleyII, Miyamoto, Williamson, CW,
            GS‑circulant, All
"""

import os, sys, time, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

ALGORITHMS = ["sylvester", "paleyI", "paleyII", "miyamoto", "williamson",
              "cw", "micromag", "gs_circulant", "all"]


def build_sylvester(n):
    if (n & (n - 1)) != 0 or n < 1:
        return None
    H = np.array([[1]], dtype=np.int8)
    while H.shape[0] < n:
        top = np.concatenate([H, H], axis=1)
        bot = np.concatenate([H, -H], axis=1)
        H = np.concatenate([top, bot], axis=0)
    return H


def build_paleyI(order):
    from hoa64.hadamard import paley, verify
    if order < 4 or order % 4 != 0:
        return None
    H = paley(order)
    if H is not None and verify(H):
        return normalize(H)
    return None


def build_paleyII(order):
    return build_paleyI(order)


def build_miyamoto(order):
    from hoa64.miyamoto import miyamoto_from_cache
    H = miyamoto_from_cache(order)
    if H is not None:
        from hoa64.hadamard import verify
        if verify(H):
            return normalize(H)
    return None


def build_williamson(order):
    if order % 4 != 0:
        return None
    k = order // 4
    from hoa64.williamson import williamson_type_quadruples_smallcases
    # Use the SageMath Williamson DB data we ported
    will_db = {1:('+','+','+','+'),3:('+++','+--','+--','+--'),
               5:('+-++-','++--+','+----','+----'),
               7:('+--++--','+-+--+-','++----+','+------'),
               9:('+---++---','+--+--+--','+-+----+-','++------+'),
               11:('++--------+','++-+-++-+-+','++-++--++-+','+-++----++-'),
               13:('++++-+--+-+++','+---+-++-+---','++---+--+---+','++---+--+---+'),
               15:('+-+---++++---+-','++-++------++-+','++-++++--++++-+','++-++-+--+-++-+'),
               }
    if k not in will_db:
        return None
    a,b,c,d = will_db[k]
    def pm(s): return np.array([1 if c=='+' else -1 for c in s], dtype=np.int64)
    def circ(s):
        m = len(s); M=np.empty((m,m),dtype=np.int64)
        for i in range(m): M[i]=np.roll(s,i)
        return M
    A=circ(pm(a));B=circ(pm(b));C=circ(pm(c));D=circ(pm(d))
    H=np.block([[A,B,C,D],[-B,A,-D,C],[-C,D,A,-B],[-D,-C,B,A]]).astype(np.int8)
    from hoa64.hadamard import verify
    if verify(H): return normalize(H)
    return None


def build_cw(order):
    from hoa64.cw_construction import cw_build
    from hoa64.hadamard import verify, normalize
    H = cw_build(order)
    if H is not None and verify(H):
        return normalize(H)
    return None


def build_micromag(order):
    if order < 4 or order % 4 != 0:
        return None
    from hoa64.micromag import micromag_ils
    H, _, ok = micromag_ils(order, lam_ex=0.01, lam_ani=0.1,
                            inner_flips=2000, outer_iters=3, time_budget=10,
                            rng=np.random.default_rng())
    if ok and H is not None:
        from hoa64.hadamard import verify
        if verify(H):
            return normalize(H)
    return None


def build_gs_circ(order):
    if order % 4 != 0:
        return None
    from hoa64.williamson import gs_circulant_ils
    H, _, _, ok = gs_circulant_ils(order // 4, inner_flips=3000, outer_iters=3,
                                    time_budget=5, rng=np.random.default_rng())
    if ok:
        from hoa64.hadamard import normalize
        return normalize(H)
    return None


def build_all(order):
    for fn in [build_sylvester, build_paleyI, build_miyamoto,
               build_williamson, build_cw, build_micromag, build_gs_circ]:
        H = fn(order)
        if H is not None:
            return H
    return None


BUILDERS = {
    "sylvester": build_sylvester,
    "paleyI": build_paleyI,
    "paleyII": build_paleyII,
    "miyamoto": build_miyamoto,
    "williamson": build_williamson,
    "cw": build_cw,
    "micromag": build_micromag,
    "gs_circulant": build_gs_circ,
    "all": build_all,
}


def normalize(H):
    n = H.shape[0]
    Hn = np.array(H, dtype=np.int8)
    for j in range(n):
        if Hn[0, j] == -1:
            Hn[:, j] *= -1
    for i in range(1, n):
        if Hn[i, 0] == -1:
            Hn[i, :] *= -1
    return Hn


def render(H, term_width=80):
    """Render H as green/black terminal pixels.  Returns a string."""
    n = H.shape[0]
    # scale to fit terminal
    scale = max(1, n // (term_width - 2))
    out = []
    GREEN = "\033[42m  \033[0m"
    BLACK = "\033[40m  \033[0m"
    for i in range(0, n, scale):
        row_str = ""
        for j in range(0, n, scale):
            cell = H[i, j]
            row_str += GREEN if cell == 1 else BLACK
        out.append(row_str)
    return "\n".join(out)


def evolve(algo_name, max_order=128, fps=3):
    builder = BUILDERS[algo_name]
    built = []
    free_orders = []

    os.system("clear" if sys.platform != "win32" else "cls")
    print(f"Algorithm: {algo_name}    Evolving H.1 → ...\n", flush=True)

    for order in range(1, max_order + 1):
        if order > 2 and order % 4 != 0:
            if order in (1, 2):
                pass
            else:
                free_orders.append(order)
                continue
        H = builder(order)
        if H is not None:
            built.append(order)
            os.system("clear" if sys.platform != "win32" else "cls")
            frame = render(H)
            bar_h = len(built)
            bar = "█" * max(1, bar_h // 5)
            print(f"  {algo_name} ─ H({order})  [{len(built)}/{order//4*len(built) if order>4 else '?'}]\n")
            print(frame)
            print(f"\n  Built: {built}")
            if free_orders:
                print(f"  Skipped (not 4k): {free_orders[:10]}{'...' if len(free_orders)>10 else ''}")
            time.sleep(1.0 / fps)
        else:
            free_orders.append(order)

    os.system("clear" if sys.platform != "win32" else "cls")
    print(f"Algorithm: {algo_name} — Final State")
    print(f"  Built: {len(built)} orders")
    print(f"  Max: H({max(built)})")
    print(f"  First gaps: {[o for o in free_orders if o%4==0 and o>2][:10]}")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Pixel‑art Hadamard visualizer")
    p.add_argument("-a", "--algo", choices=ALGORITHMS, default="all",
                   help="construction algorithm")
    p.add_argument("-n", "--max", type=int, default=128,
                   help="max order to attempt")
    p.add_argument("-f", "--fps", type=float, default=4,
                   help="frames per second")
    args = p.parse_args()
    evolve(args.algo, args.max, args.fps)


if __name__ == "__main__":
    main()
