#!/usr/bin/env python3
"""Game of Hadamard — construction-DAG animation distinguishing built vs claimed.

Orders up to max_n come from three tiers:
  BUILT  — actually constructed & verified by our toolchain (Sylvester,
           Paley, Paley‑pp, Kronecker, CSV‑ingested)
  CLAIM  — known to exist via Milestone/CWT/SDS but NOT yet built by us
  GAP    — no known construction

Display: uppercase glyph = built, lowercase = claimed, dot = gap.
Sidebar shows built count, claimed count, highest built, first gap, RH.
"""

import os, sys, time, math
from hoa64.hadamard import hadamard_orders, _is_prime_power
from hoa64.rh import rh_check

GLYPH = ["S","1","2","=","W","M","T","D","V"]
GAP   = "."
COL_N = len(GLYPH)

def _pp_check(q):
    return _is_prime_power(q) is not None


def _classify(max_n):
    built = set(hadamard_orders(max_n))

    # Miyamoto candidates (H(q-1) must be BUILT, q≡1mod4 prime power)
    miyamoto = set()
    for q in range(5, max_n // 4 + 1):
        if q % 4 != 1 or not _pp_check(q):
            continue
        o = 4 * q
        if o <= max_n and (q - 1) in built and o not in built:
            miyamoto.add(o)

    # CW (Cooper-Wallis with T-sequences, from Cati-Pasechnik Table 2)
    cw_ts = {47,59,65,67,81,89,93,107,119,133,153,183,189,
             209,213,235,245,247,249,253,259,267,275,287,295,299}
    cw = {4*t for t in cw_ts if 4*t <= max_n and 4*t not in built}

    # SDS (supplementary difference sets)
    sds_n = {103,127,151,163,191,219,239,251}
    sds = {4*n for n in sds_n if 4*n <= max_n and 4*n not in built}

    claimed = miyamoto | cw | sds
    return built, claimed


def _method_label(n, built, claimed):
    m = set()
    if n in built:
        if n <= 2:
            m.add("S")
        elif (n & (n - 1)) == 0 and n > 2:
            m.add("=")
        q1 = n - 1
        if q1 >= 3 and q1 % 4 == 3 and _pp_check(q1):
            m.add("1")
        q2 = n // 2 - 1 if n % 2 == 0 else None
        if q2 and q2 >= 5 and q2 % 4 == 1 and _pp_check(q2):
            m.add("2")
        if 668 <= n <= 1964 and n % 4 == 0:
            m.add("V")
        if n % 4 == 0:
            q = n // 4
            if q in (23, 29, 39, 43): m.add("W")
            if q in (103, 127, 151, 163, 191, 219, 239, 251): m.add("D")
            if q in (47,59,65,67,81,89,93,107,119,133,153,183,189,
                     209,213,235,245,247,249,253,259,267,275,287,295,299): m.add("T")
    if (n in claimed and not m) or (n in claimed):
        if n in claimed and not m:
            miy = any(q % 4 == 1 and _pp_check(q) and (q - 1) in built and 4 * q == n
                      for q in range(5, n // 4 + 1))
            if miy: m.add("M")
            if n in {4*t for t in (47,59,65,67,81,89,93,107,119,133,153,183,189,
                                   209,213,235,245,247,249,253,259,267,275,287,295,299)}:
                m.add("T")
            if n in {4*n for n in (103,127,151,163,191,219,239,251)}:
                m.add("D")
    return m or {"S"}


def _depths(built, claimed):
    d = {}
    d[1] = d[2] = 0
    all_n = sorted(built | claimed)
    for _ in range(20):
        for n in all_n:
            if n in d:
                continue
            if n == 4:
                d[n] = 1
            elif n % 2 == 0 and n // 2 in d:
                d[n] = d[n // 2] + 1
            elif n % 4 == 0 and n // 4 in d:
                d[n] = d[n // 4] + 2
    for n in all_n:
        if n not in d:
            d[n] = 2
    return d


def evolve(max_n=128, pause=0.3, max_gen=None):
    built, claimed = _classify(max_n)
    all_n = sorted(built | claimed)
    cst = {n: _method_label(n, built, claimed) for n in all_n}
    dpt = _depths(built, claimed)
    max_d = max(dpt.values())
    if max_gen is None:
        max_gen = max_d

    first_gap = None
    for n in range(4, max_n + 1, 4):
        if n not in built and n not in claimed:
            first_gap = n
            break

    rows = max_n // 4 + 1

    try:
        for gen in range(1, max_gen + 1):
            os.system("clear" if sys.platform != "win32" else "cls")
            alive = [n for n in all_n if dpt.get(n, 0) <= gen]
            hi = max(alive)
            built_alive = sum(1 for n in alive if n in built)
            claimed_alive = len(alive) - built_alive
            rh = rh_check(N=10)

            hdr = " " * 8
            for a in GLYPH:
                hdr += f" {a}  "
            print(hdr + " d")
            print("─" * (8 + COL_N * 4) + "──")

            for row in range(rows):
                n = row * 4
                label = "H1,H2" if row == 0 else f"  {n:>5}"
                line = label
                for ci, abv in enumerate(GLYPH):
                    g = GAP
                    if row == 0 and abv == "S":
                        g = "S"
                    elif abv in cst.get(n, set()):
                        in_built = n in built
                        mthods = cst[n]
                        g = abv if in_built else abv.lower()
                    line += f" {g}  "
                d = dpt.get(n, 99)
                mark = "#" if n in built else ("c" if n in claimed else "·")
                line += f" {d:>2}"
                if row == 0:
                    line += "  #=built c=claimed ·=gap"
                print(line)

            print()
            print(f"  Gen {gen}/{max_d}   built: {built_alive}  claimed: {claimed_alive}  "
                  f"highest: H({hi})")
            print(f"  bound-log10 = {0.5 * hi * math.log10(hi):.1f}   "
                  f"1st gap: {first_gap}" if first_gap else "  no gaps")
            print(f"  RH: bound={rh['bound']:.0f}  ({rh['verdict'][:30]})")
            print(f"  # = built & verified   c = claimed, not built   · = unknown")

            time.sleep(pause)
    except KeyboardInterrupt:
        pass

    os.system("clear" if sys.platform != "win32" else "cls")
    print(f"Final:  built={len(built)}  claimed={len(claimed)}  "
          f"top=H({max(built | claimed)})  gap={first_gap}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Game of Hadamard")
    p.add_argument("-n", "--max", type=int, default=128)
    p.add_argument("-p", "--pause", type=float, default=0.3)
    p.add_argument("-g", "--gens", type=int, default=None)
    args = p.parse_args()
    evolve(args.max, args.pause, args.gens)
