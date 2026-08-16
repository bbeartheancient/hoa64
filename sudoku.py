"""Sudoku-solving algorithms with the goal remapped to Hadamard rows.

Source
------
https://en.wikipedia.org/wiki/Sudoku_solving_algorithms

Wikipedia's solvers fill a 9×9 grid so each row, column, and box holds
the digits 1–9.  The same algorithms fill an n×n ±1 matrix so the *rows*
are pairwise orthogonal of weight √n — i.e. H Hᵀ = n I, a Hadamard
matrix.  Digits become signs; boxes disappear; the "clue" is the
gauge-fixed first row and first column (all +1 after ``normalize``).

Technique → Hadamard
--------------------
Backtracking / brute force
    DFS on cells, or on whole admissible rows.  A partial row is
    rejected as soon as a pairwise inner product cannot still hit 0.
Pattern overlay / templates
    A template is one admissible ±1 row (first bit +1, exactly n/2
    pluses).  Overlay places mutually orthogonal templates into the
    remaining row-slots.
Stochastic search
    Random balanced fill + shuffle of residual errors to zero:
    simulated annealing, tabu tenure, and a small genetic row-crossover.
Constraint programming
    Variables = remaining rows; domain = admissible templates.
    MRV + AC-3-style wipe of templates that are not orthogonal to an
    assignment.
Exact cover / Algorithm X / dancing links
    Knuth's dancing-candidates search for a clique of n−1 pairwise
    orthogonal templates.  Full dancing-links exact cover (slots ×
    templates, secondary conflict items) for n ≤ 12.
Relations and residuals
    Residual of pair (i, j) is (Hᵢ · Hⱼ)².  Nonzero residual is a
    prohibited arrangement; repair flips the agreeing columns of the
    worst pair.

Public solvers
--------------
``sudoku_sa`` / ``sudoku_ils`` match the Gerzon/tile contract
(``step, T, E, best_E, accepts, H`` every 500; ILS ``while True`` until
``time_budget`` / ``stop_flag``).  ILS cycles the Wikipedia set;
``method`` pins one technique.
"""
from __future__ import annotations

import math
import time
from itertools import combinations

import numpy as np

from .hadamard import (
    normalize,
    pack_row,
    random_seed,
    unpack_row,
    verify,
)


#: Wikipedia techniques, remapped.  ILS cycles this order — constructive
#: solvers first (backtrack finds H16 in milliseconds), then overlay/CSP,
#: then dancing-links exact cover, then residual / stochastic repair.
METHODS = (
    "backtrack",
    "overlay",
    "csp",
    "exact",
    "residual",
    "stochastic",
)

#: Enumerate every admissible template only while C(n−1, n/2−1) ≤ this.
_ENUM_MAX = 20_000

#: Full DLX (slots × templates + conflict secondaries) only this small.
_DLX_MAX_N = 12


def _all_plus(n: int) -> int:
    return (1 << n) - 1


def _n_templates(n: int) -> int:
    if n < 2:
        return 1
    return math.comb(n - 1, n // 2 - 1)


def enumerate_templates(n: int) -> list[int]:
    """Every gauge-fixed admissible row: bit 0 set, popcount = n/2."""
    if n < 2:
        return [1]
    need = n // 2 - 1
    out = []
    for combo in combinations(range(1, n), need):
        bits = 1
        for j in combo:
            bits |= 1 << j
        out.append(bits)
    return out


def sample_templates(n: int, k: int, rng: np.random.Generator) -> list[int]:
    """k random admissible rows (with replacement-collision retry)."""
    if n < 2:
        return [1]
    need = n // 2 - 1
    seen: set[int] = set()
    out: list[int] = []
    guard = 0
    while len(out) < k and guard < k * 20:
        guard += 1
        pos = rng.choice(n - 1, size=need, replace=False) + 1
        bits = 1
        for j in pos:
            bits |= 1 << int(j)
        if bits not in seen:
            seen.add(bits)
            out.append(bits)
    return out


def templates_for(n: int, rng: np.random.Generator, cap: int = 4000) -> list[int]:
    if _n_templates(n) <= _ENUM_MAX:
        return enumerate_templates(n)
    return sample_templates(n, min(cap, 4000), rng)


def _ortho(a: int, b: int, n: int) -> bool:
    return (a ^ b).bit_count() == n // 2


def _dots_ok(bits: int, accepted: list[int], n: int) -> bool:
    half = n // 2
    return all((bits ^ a).bit_count() == half for a in accepted)


def residual_energy(H: np.ndarray) -> float:
    """F = Σ_{i<j} (Hᵢ · Hⱼ)² — 0 iff the rows are pairwise orthogonal."""
    A = np.asarray(H, dtype=np.float64)
    G = A @ A.T
    n = A.shape[0]
    off = G - n * np.eye(n)
    return float(np.sum(off * off) / 2.0)


def residual_energy_rows(rows: list[int], n: int) -> float:
    e = 0
    for i in range(len(rows)):
        ai = rows[i]
        for j in range(i):
            d = n - 2 * (ai ^ rows[j]).bit_count()
            e += d * d
    return float(e)


def gram_energy(H: np.ndarray) -> float:
    return residual_energy(H)


def _matrix_from_rows(rows: list[int], n: int) -> np.ndarray:
    H = np.empty((n, n), dtype=np.int8)
    for i, bits in enumerate(rows):
        H[i] = unpack_row(bits, n)
    if len(rows) < n:
        # pad with copies of row 0 so callers always get an n×n ±1 grid
        plus = unpack_row(_all_plus(n), n)
        for i in range(len(rows), n):
            H[i] = plus
    return H


def _accepted_prefix(H: np.ndarray) -> list[int]:
    """Longest gauge-fixed orthogonal prefix of the rows of ``H``."""
    A = normalize(np.asarray(H, dtype=np.int8))
    n = int(A.shape[0])
    accepted = [_all_plus(n)]
    half = n // 2
    for i in range(1, n):
        bits = pack_row(A[i])
        if bits.bit_count() != half:
            continue
        if _dots_ok(bits, accepted, n):
            accepted.append(bits)
    return accepted


def analyze(H: np.ndarray) -> dict:
    """Row-residual stats (no search).  Safe to attach to a job result."""
    A = np.asarray(H, dtype=np.int8)
    n = int(A.shape[0])
    rows = [pack_row(A[i]) for i in range(n)]
    n_pairs = n * (n - 1) // 2
    n_ortho = 0
    worst = 0
    for i in range(n):
        for j in range(i):
            d = abs(n - 2 * (rows[i] ^ rows[j]).bit_count())
            if d == 0:
                n_ortho += 1
            if d > worst:
                worst = d
    return {
        "order": n,
        "F": residual_energy_rows(rows, n),
        "E_res": residual_energy_rows(rows, n),
        "n_ortho_pairs": n_ortho,
        "n_pairs": n_pairs,
        "worst_dot": int(worst),
        "gauge_ok": bool(np.all(A[0] == 1) and np.all(A[:, 0] == 1)),
        "balance_ok": all(r.bit_count() == n // 2 for r in rows[1:]) if n >= 2 else True,
        "n_prefix": len(_accepted_prefix(A)),
        "methods": list(METHODS),
    }


# ---------------------------------------------------------------------------
# Exact cover / Algorithm X / dancing links
# ---------------------------------------------------------------------------


class _Dance:
    """Dancing list of candidate templates (Knuth cover/uncover)."""

    __slots__ = ("L", "R", "n")

    def __init__(self, n: int):
        # nodes 0..n-1;  circular.  n == 0 is empty.
        self.n = n
        self.L = [(i - 1) % n for i in range(n)] if n else []
        self.R = [(i + 1) % n for i in range(n)] if n else []

    def cover(self, i: int) -> None:
        self.R[self.L[i]] = self.R[i]
        self.L[self.R[i]] = self.L[i]

    def uncover(self, i: int) -> None:
        self.R[self.L[i]] = i
        self.L[self.R[i]] = i


def _algorithm_x(cands: list[int], n: int, need: int,
                 rng, deadline, stop_flag, node_limit: int):
    """Dancing-candidates Algorithm X: clique of ``need`` orthogonal rows."""
    m = len(cands)
    if m == 0 or need <= 0:
        return None
    # compatibility as sorted arrays for fast walk
    half = n // 2
    compat = [[] for _ in range(m)]
    for i, a in enumerate(cands):
        ai = cands[i]
        for j in range(i + 1, m):
            if (ai ^ cands[j]).bit_count() == half:
                compat[i].append(j)
                compat[j].append(i)
    # start from a random permutation of the candidate order
    order = rng.permutation(m).tolist()
    dance = _Dance(m)
    # hide nothing yet; pick in ``order`` but skip covered
    covered = bytearray(m)
    chosen: list[int] = []
    nodes = [0]
    found: list[int] | None = None

    def expired() -> bool:
        if node_limit and nodes[0] >= node_limit:
            return True
        if deadline and time.monotonic() > deadline:
            return True
        return stop_flag is not None and stop_flag.is_set()

    def search(live_head: int) -> bool:
        nonlocal found
        if len(chosen) >= need:
            found = list(chosen)
            return True
        if expired() or dance.n == 0:
            return False
        # remaining live candidates
        live = []
        i = live_head
        start = i
        # find a live head if this one is covered
        guard = 0
        while covered[i] and guard < m:
            i = dance.R[i]
            guard += 1
            if i == start and covered[i]:
                return False
        start = i
        while True:
            if not covered[i]:
                live.append(i)
            i = dance.R[i]
            if i == start:
                break
            if len(live) > m:
                break
        remain = need - len(chosen)
        if len(live) < remain:
            return False
        # MRV: pick the live cand with fewest still-live neighbours
        def live_deg(v):
            return sum(1 for w in compat[v] if not covered[w])

        # try in the shuffled ``order`` among live, but prefer low degree
        live.sort(key=lambda v: (live_deg(v), order[v] if v < len(order) else v))
        for v in live:
            nodes[0] += 1
            if expired():
                return False
            # cover v and everyone not compatible with v
            hidden = []
            chosen.append(v)
            covered[v] = 1
            dance.cover(v)
            hidden.append(v)
            # hide live cands that are not neighbours of v
            neigh = set(compat[v])
            i = dance.R[v] if dance.n else v
            # after covering v, R[v] is the next live (v is unlinked but
            # its R still points at the old neighbour)
            start = dance.R[v]
            i = start
            to_hide = []
            if start != v or m == 1:
                # walk current circle; v is already unlinked so start ≠ v
                # unless m==1
                seen = 0
                while seen < m:
                    if i != v and not covered[i] and i not in neigh:
                        to_hide.append(i)
                    i = dance.R[i]
                    seen += 1
                    if i == start:
                        break
            for u in to_hide:
                covered[u] = 1
                dance.cover(u)
                hidden.append(u)
            # next live head
            nxt = dance.R[v]
            if search(nxt):
                return True
            chosen.pop()
            for u in reversed(hidden):
                dance.uncover(u)
                covered[u] = 0
        return False

    search(int(order[0]) if order else 0)
    if found is None:
        return None
    return [cands[i] for i in found]


class _DLX:
    """Knuth dancing links over a 0-based item list (primary + secondary).

    Secondary items are columns ``>= n_primary``: they may stay uncovered.
    """

    def __init__(self, n_primary: int, n_secondary: int = 0):
        n_items = n_primary + n_secondary
        self.n_primary = n_primary
        self.n_items = n_items
        # node 0 is the root; 1..n_items are column headers
        N = n_items + 1
        self.L = list(range(N))
        self.R = list(range(N))
        self.U = list(range(N))
        self.D = list(range(N))
        self.C = list(range(N))
        self.S = [0] * N
        self.opt_of = [-1] * N  # which option a node belongs to
        # link primary headers into the root circle; secondaries hang off
        self.L[0] = n_primary if n_primary else 0
        self.R[0] = 1 if n_primary else 0
        for c in range(1, n_primary + 1):
            self.L[c] = c - 1
            self.R[c] = c + 1 if c < n_primary else 0
        if n_primary:
            self.R[n_primary] = 0
            self.L[1] = 0
        # secondaries are not in the root circle
        for c in range(n_primary + 1, N):
            self.L[c] = self.R[c] = c
        self._option_items: list[list[int]] = []

    def add_option(self, items: list[int]) -> int:
        """``items`` are 0-based item indices.  Returns option id."""
        opt = len(self._option_items)
        self._option_items.append(list(items))
        first = None
        for it in items:
            c = it + 1  # header index
            node = len(self.L)
            self.L.append(node)
            self.R.append(node)
            self.U.append(self.U[c])
            self.D.append(c)
            self.C.append(c)
            self.opt_of.append(opt)
            self.D[self.U[c]] = node
            self.U[c] = node
            self.S[c] += 1
            if first is None:
                first = node
            else:
                self.L[node] = self.L[first]
                self.R[node] = first
                self.R[self.L[first]] = node
                self.L[first] = node
        return opt

    def _cover(self, c: int) -> None:
        self.R[self.L[c]] = self.R[c]
        self.L[self.R[c]] = self.L[c]
        i = self.D[c]
        while i != c:
            j = self.R[i]
            while j != i:
                self.D[self.U[j]] = self.D[j]
                self.U[self.D[j]] = self.U[j]
                self.S[self.C[j]] -= 1
                j = self.R[j]
            i = self.D[i]

    def _uncover(self, c: int) -> None:
        i = self.U[c]
        while i != c:
            j = self.L[i]
            while j != i:
                self.S[self.C[j]] += 1
                self.D[self.U[j]] = j
                self.U[self.D[j]] = j
                j = self.L[j]
            i = self.U[i]
        self.R[self.L[c]] = c
        self.L[self.R[c]] = c

    def search(self, deadline=None, stop_flag=None, node_limit=200_000):
        sol: list[int] = []
        found: list[int] | None = None
        nodes = [0]

        def expired() -> bool:
            if node_limit and nodes[0] >= node_limit:
                return True
            if deadline and time.monotonic() > deadline:
                return True
            return stop_flag is not None and stop_flag.is_set()

        def rec() -> bool:
            nonlocal found
            if self.R[0] == 0:
                found = list(sol)
                return True
            if expired():
                return False
            # choose primary column with smallest S
            c = self.R[0]
            best = c
            best_s = self.S[c]
            while c != 0:
                if self.S[c] < best_s:
                    best, best_s = c, self.S[c]
                    if best_s == 0:
                        break
                c = self.R[c]
            if best_s == 0:
                return False
            c = best
            self._cover(c)
            r = self.D[c]
            while r != c:
                nodes[0] += 1
                sol.append(self.opt_of[r])
                j = self.R[r]
                while j != r:
                    self._cover(self.C[j])
                    j = self.R[j]
                if rec():
                    return True
                j = self.L[r]
                while j != r:
                    self._uncover(self.C[j])
                    j = self.L[j]
                sol.pop()
                r = self.D[r]
            self._uncover(c)
            return False

        rec()
        return found


def _dlx_exact(cands: list[int], n: int, need: int,
               rng, deadline, stop_flag, node_limit: int):
    """Exact cover: ``need`` slots, each template used at most once,
    incompatible pair = secondary item covered at most once."""
    m = len(cands)
    if m < need or need <= 0:
        return None
    half = n // 2
    # conflict pairs as secondary items
    conflicts: list[tuple[int, int]] = []
    inc_of = [[] for _ in range(m)]  # template -> conflict-item indices
    for i in range(m):
        for j in range(i + 1, m):
            if (cands[i] ^ cands[j]).bit_count() != half:
                idx = len(conflicts)
                conflicts.append((i, j))
                inc_of[i].append(idx)
                inc_of[j].append(idx)
    n_primary = need
    n_secondary = m + len(conflicts)  # template-used + conflicts
    dlx = _DLX(n_primary, n_secondary)
    # option (slot s, template t) covers: slot s, template-secondary t,
    # and every conflict involving t
    opt_map: list[tuple[int, int]] = []
    # cut the (need)! by forcing template index to increase with slot:
    # only emit options where we don't care — still emit all, DLX will
    # find one matching.  For n=12 that's 11*462 options; OK.
    # Shuffle template order so restarts differ.
    t_order = rng.permutation(m).tolist()
    for s in range(need):
        for t in t_order:
            items = [s, n_primary + t]
            for ci in inc_of[t]:
                items.append(n_primary + m + ci)
            dlx.add_option(items)
            opt_map.append((s, t))
    picked = dlx.search(deadline=deadline, stop_flag=stop_flag,
                        node_limit=node_limit)
    if not picked:
        return None
    used = [cands[opt_map[o][1]] for o in picked]
    return used


def _exact_rows(n: int, accepted: list[int], rng, deadline, stop_flag,
                node_limit: int = 200_000) -> list[int] | None:
    need = n - len(accepted)
    if need <= 0:
        return list(accepted)
    pool = templates_for(n, rng)
    # drop templates that clash with the already-accepted prefix
    pool = [t for t in pool if _dots_ok(t, accepted, n)]
    if len(pool) < need:
        return None
    extra = _algorithm_x(pool, n, need, rng, deadline, stop_flag, node_limit)
    if extra is None and n <= _DLX_MAX_N and _n_templates(n) <= _ENUM_MAX:
        extra = _dlx_exact(pool, n, need, rng, deadline, stop_flag, node_limit)
    if extra is None:
        return None
    return list(accepted) + extra


# ---------------------------------------------------------------------------
# Backtracking (cell / row DFS)
# ---------------------------------------------------------------------------


def _build_row(n: int, accepted: list[int], rng, deadline, stop_flag,
               node_limit: int, nodes: list[int]) -> int | None:
    """DFS-fill one admissible row orthogonal to ``accepted``."""
    half = n // 2
    plus_need = half - 1  # bit 0 is already +1
    n_acc = len(accepted)
    acc = list(accepted)
    # disagreement counts so far (bit 0: row is +1, accepted bit0 is +1 → agree)
    dsg = [0] * n_acc
    bits0 = 1

    # random column order for the free bits
    cols = rng.permutation(n - 1) + 1
    cols = [int(c) for c in cols]

    found = [None]

    def expired() -> bool:
        if node_limit and nodes[0] >= node_limit:
            return True
        if deadline and time.monotonic() > deadline:
            return True
        return stop_flag is not None and stop_flag.is_set()

    def rec(k: int, bits: int, pluses: int) -> bool:
        if expired():
            return False
        left = (n - 1) - k
        if pluses > plus_need or pluses + left < plus_need:
            return False
        for t in range(n_acc):
            d = dsg[t]
            if d > half or d + left < half:
                return False
        if k == n - 1:
            if pluses == plus_need and all(d == half for d in dsg):
                found[0] = bits
                return True
            return False
        j = cols[k]
        # try +1 then -1, shuffled
        signs = (1, 0) if rng.random() < 0.5 else (0, 1)
        for s in signs:
            nodes[0] += 1
            new_bits = bits | (s << j)
            # update disagreements: we disagree with accepted t iff
            # accepted-bit-j != s
            undo = []
            ok = True
            for t in range(n_acc):
                a_bit = (acc[t] >> j) & 1
                if a_bit != s:
                    dsg[t] += 1
                    undo.append(t)
                    if dsg[t] > half:
                        ok = False
                        break
            if ok and rec(k + 1, new_bits, pluses + s):
                return True
            for t in undo:
                dsg[t] -= 1
        return False

    rec(0, bits0, 0)
    return found[0]


def _backtrack_rows(n: int, accepted: list[int], rng, deadline, stop_flag,
                    node_limit: int = 250_000) -> list[int] | None:
    rows = list(accepted)
    nodes = [0]
    while len(rows) < n:
        if deadline and time.monotonic() > deadline:
            return None
        if stop_flag is not None and stop_flag.is_set():
            return None
        nxt = _build_row(n, rows, rng, deadline, stop_flag, node_limit, nodes)
        if nxt is None:
            return None
        rows.append(nxt)
    return rows


# ---------------------------------------------------------------------------
# Pattern overlay + CSP
# ---------------------------------------------------------------------------


def _overlay_rows(n: int, accepted: list[int], rng, deadline, stop_flag,
                  node_limit: int = 80_000) -> list[int] | None:
    """Greedy overlay of shuffled templates with limited backtrack."""
    need = n - len(accepted)
    if need <= 0:
        return list(accepted)
    pool = templates_for(n, rng)
    pool = [t for t in pool if _dots_ok(t, accepted, n)]
    rng.shuffle(pool)
    chosen = list(accepted)
    nodes = [0]

    def rec(start: int) -> bool:
        if len(chosen) >= n:
            return True
        if node_limit and nodes[0] >= node_limit:
            return False
        if deadline and time.monotonic() > deadline:
            return False
        if stop_flag is not None and stop_flag.is_set():
            return False
        for i in range(start, len(pool)):
            t = pool[i]
            nodes[0] += 1
            if _dots_ok(t, chosen, n):
                chosen.append(t)
                if rec(i + 1):
                    return True
                chosen.pop()
        return False

    return chosen if rec(0) and len(chosen) == n else None


def _csp_rows(n: int, accepted: list[int], rng, deadline, stop_flag,
              node_limit: int = 80_000) -> list[int] | None:
    """MRV + forward-checking over a template domain per remaining row."""
    pool = templates_for(n, rng)
    pool = [t for t in pool if _dots_ok(t, accepted, n)]
    if not pool:
        return None
    n_left = n - len(accepted)
    if n_left <= 0:
        return list(accepted)
    # domain for each remaining slot: indices into pool
    domains = [set(range(len(pool))) for _ in range(n_left)]
    assign = [-1] * n_left
    nodes = [0]
    half = n // 2

    def propagate(slot: int, tid: int) -> list | None:
        """Assign pool[tid] to slot; wipe; return trail or None on fail."""
        trail = []
        tbits = pool[tid]
        for s in range(n_left):
            if s == slot or assign[s] >= 0:
                continue
            dead = [u for u in domains[s]
                    if (pool[u] ^ tbits).bit_count() != half]
            if dead:
                domains[s].difference_update(dead)
                trail.append((s, dead))
                if not domains[s]:
                    return None
        return trail

    def undo(trail):
        for s, dead in trail:
            domains[s].update(dead)

    def rec() -> bool:
        if all(a >= 0 for a in assign):
            return True
        if node_limit and nodes[0] >= node_limit:
            return False
        if deadline and time.monotonic() > deadline:
            return False
        if stop_flag is not None and stop_flag.is_set():
            return False
        # MRV
        open_slots = [s for s in range(n_left) if assign[s] < 0]
        slot = min(open_slots, key=lambda s: len(domains[s]))
        opts = list(domains[slot])
        rng.shuffle(opts)
        saved = set(domains[slot])
        for tid in opts:
            nodes[0] += 1
            assign[slot] = tid
            domains[slot] = {tid}
            trail = propagate(slot, tid)
            if trail is not None and rec():
                return True
            if trail is not None:
                undo(trail)
            assign[slot] = -1
            domains[slot] = set(saved)
        return False

    if not rec():
        return None
    extra = [pool[tid] for tid in assign]
    return list(accepted) + extra


# ---------------------------------------------------------------------------
# Residuals + stochastic (SA / tabu / genetic)
# ---------------------------------------------------------------------------


def _row_contrib(rows: list[int], i: int, n: int) -> int:
    e = 0
    ai = rows[i]
    for k, bk in enumerate(rows):
        if k == i:
            continue
        d = n - 2 * (ai ^ bk).bit_count()
        e += d * d
    return e


def _repair_worst_pair(rows: list[int], n: int, rng) -> bool:
    """Flip two complementary interior bits on the worst residual pair."""
    worst_i = worst_j = -1
    worst = 0
    for i in range(1, len(rows)):
        for j in range(1, i):
            d = abs(n - 2 * (rows[i] ^ rows[j]).bit_count())
            if d > worst:
                worst, worst_i, worst_j = d, i, j
    if worst == 0 or worst_i < 0:
        return False
    a, b = rows[worst_i], rows[worst_j]
    # columns where they agree (too many agreements if dot > 0)
    dot = n - 2 * (a ^ b).bit_count()
    agree = []
    disagree = []
    for j in range(1, n):
        bit = 1 << j
        if (a & bit) == (b & bit):
            agree.append(j)
        else:
            disagree.append(j)
    target = agree if dot > 0 else disagree
    if len(target) < 2:
        return False
    j1, j2 = (int(x) for x in rng.choice(target, size=2, replace=False))
    # flip both bits on the higher-residual row (preserves that row's weight
    # only if the two bits differ).  If they are the same sign, flip j1 on
    # this row and a compensating opposite-sign interior bit.
    victim = worst_i if _row_contrib(rows, worst_i, n) >= _row_contrib(rows, worst_j, n) else worst_j
    bits = rows[victim]
    s1 = (bits >> j1) & 1
    s2 = (bits >> j2) & 1
    if s1 != s2:
        rows[victim] = bits ^ (1 << j1) ^ (1 << j2)
        return True
    # find an opposite-sign interior column
    for j in range(1, n):
        if ((bits >> j) & 1) != s1:
            rows[victim] = bits ^ (1 << j1) ^ (1 << j)
            return True
    return False


def _swap_in_row(rows: list[int], i: int, n: int, rng) -> tuple[int, int] | None:
    """Exchange one +1 and one −1 in row i (interior).  Returns (jp, jn)."""
    bits = rows[i]
    plus = [j for j in range(1, n) if (bits >> j) & 1]
    minus = [j for j in range(1, n) if not ((bits >> j) & 1)]
    if not plus or not minus:
        return None
    jp = int(plus[int(rng.integers(0, len(plus)))])
    jn = int(minus[int(rng.integers(0, len(minus)))])
    rows[i] = bits ^ (1 << jp) ^ (1 << jn)
    return jp, jn


def _stochastic_improve(rows: list[int], n: int, rng, deadline, stop_flag,
                        steps: int, T_start: float, cooling: float,
                        tabu_tenure: int = 12, callback=None,
                        start_step: int = 0, T_end: float = 0.01,
                        ) -> tuple[list[int], dict]:
    """SA + tabu on weight-preserving row swaps.  Frames match tile SA."""
    cur = list(rows)
    if len(cur) < n:
        # pad with random admissible rows
        extra = sample_templates(n, n - len(cur), rng)
        cur = cur + extra
        cur = cur[:n]
    E = residual_energy_rows(cur, n)
    best = list(cur)
    best_E = E
    T = T_start
    accepts = 0
    tabu: dict[tuple[int, int, int], int] = {}
    t0 = time.monotonic()
    step = 0
    while step < steps and T > T_end:
        if deadline and time.monotonic() > deadline:
            break
        if stop_flag is not None and stop_flag.is_set():
            break
        # bias toward high-residual rows (skip row 0)
        weights = np.array([_row_contrib(cur, i, n) for i in range(1, n)],
                           dtype=np.float64)
        if weights.sum() <= 0:
            break
        weights = weights / weights.sum()
        i = 1 + int(rng.choice(n - 1, p=weights))
        moved = _swap_in_row(cur, i, n, rng)
        if moved is None:
            step += 1
            T *= cooling
            continue
        jp, jn = moved
        key = (i, min(jp, jn), max(jp, jn))
        E_new = residual_energy_rows(cur, n)
        banned = tabu.get(key, -1) > step
        delta = E_new - E
        take = (not banned) and (
            delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10))
        )
        if take:
            E = E_new
            accepts += 1
            tabu[key] = step + tabu_tenure
            if E < best_E:
                best = list(cur)
                best_E = E
        else:
            cur[i] ^= (1 << jp) ^ (1 << jn)
        step += 1
        T *= cooling
        if callback is not None and (start_step + step) % 500 == 0:
            H = _matrix_from_rows(best, n)
            callback({
                "step": start_step + step, "T": T, "E": E, "best_E": best_E,
                "accepts": accepts, "H": H, "method": "stochastic",
                "E_res": best_E,
            })
        if best_E < 1e-9:
            break
    return best, dict(
        steps=step, accepts=accepts, best_E=best_E,
        elapsed_s=time.monotonic() - t0, hadamard=(best_E < 1e-9),
        method="stochastic",
    )


def _residual_improve(rows: list[int], n: int, rng, deadline, stop_flag,
                      steps: int, callback=None, start_step: int = 0):
    cur = list(rows)
    if len(cur) < n:
        cur = cur + sample_templates(n, n - len(cur), rng)
        cur = cur[:n]
    E = residual_energy_rows(cur, n)
    best = list(cur)
    best_E = E
    accepts = 0
    t0 = time.monotonic()
    for step in range(1, steps + 1):
        if deadline and time.monotonic() > deadline:
            break
        if stop_flag is not None and stop_flag.is_set():
            break
        if not _repair_worst_pair(cur, n, rng):
            break
        E_new = residual_energy_rows(cur, n)
        if E_new <= E:
            E = E_new
            accepts += 1
            if E < best_E:
                best = list(cur)
                best_E = E
        else:
            # keep the repair with small probability so we do not stall
            if rng.random() < 0.15:
                E = E_new
                accepts += 1
            else:
                cur = list(best)
                E = best_E
        if callback is not None and (start_step + step) % 500 == 0:
            callback({
                "step": start_step + step, "T": 0.0, "E": E, "best_E": best_E,
                "accepts": accepts, "H": _matrix_from_rows(best, n),
                "method": "residual", "E_res": best_E,
            })
        if best_E < 1e-9:
            break
    return best, dict(
        steps=step if steps else 0, accepts=accepts, best_E=best_E,
        elapsed_s=time.monotonic() - t0, hadamard=(best_E < 1e-9),
        method="residual",
    )


def _genetic_rows(n: int, accepted: list[int], rng, deadline, stop_flag,
                  generations: int = 40, pop_size: int = 8) -> list[int] | None:
    """Row-crossover GA (Wikipedia stochastic family)."""
    def random_ind():
        rows = list(accepted)
        extra = sample_templates(n, max(0, n - len(rows)), rng)
        return (rows + extra)[:n]

    pop = [random_ind() for _ in range(pop_size)]
    fit = [residual_energy_rows(p, n) for p in pop]
    best_i = int(np.argmin(fit))
    best, best_E = list(pop[best_i]), fit[best_i]
    for _g in range(generations):
        if deadline and time.monotonic() > deadline:
            break
        if stop_flag is not None and stop_flag.is_set():
            break
        if best_E < 1e-9:
            return best
        # tournament
        i, j = (int(x) for x in rng.choice(pop_size, size=2, replace=False))
        pa = pop[i] if fit[i] <= fit[j] else pop[j]
        i, j = (int(x) for x in rng.choice(pop_size, size=2, replace=False))
        pb = pop[i] if fit[i] <= fit[j] else pop[j]
        child = [pa[0]]  # keep all-+
        for r in range(1, n):
            child.append(pa[r] if rng.random() < 0.5 else pb[r])
        # one residual repair
        _repair_worst_pair(child, n, rng)
        e = residual_energy_rows(child, n)
        worst = int(np.argmax(fit))
        if e <= fit[worst]:
            pop[worst] = child
            fit[worst] = e
            if e < best_E:
                best, best_E = list(child), e
    return best if best_E < 1e-9 else None


def _complete(method: str, n: int, accepted: list[int], rng,
              deadline, stop_flag, node_limit: int) -> list[int] | None:
    if method == "exact":
        return _exact_rows(n, accepted, rng, deadline, stop_flag, node_limit)
    if method == "backtrack":
        return _backtrack_rows(n, accepted, rng, deadline, stop_flag, node_limit)
    if method == "csp":
        return _csp_rows(n, accepted, rng, deadline, stop_flag, node_limit)
    if method == "overlay":
        return _overlay_rows(n, accepted, rng, deadline, stop_flag, node_limit)
    if method == "residual":
        rows, st = _residual_improve(
            accepted if len(accepted) == n else
            accepted + sample_templates(n, n - len(accepted), rng),
            n, rng, deadline, stop_flag, steps=min(node_limit, 8000),
        )
        return rows if st.get("hadamard") else None
    if method == "stochastic":
        seed = accepted if len(accepted) == n else \
            accepted + sample_templates(n, n - len(accepted), rng)
        rows, st = _stochastic_improve(
            seed[:n], n, rng, deadline, stop_flag,
            steps=min(node_limit, 8000), T_start=20.0, cooling=0.997,
        )
        if st.get("hadamard"):
            return rows
        return _genetic_rows(n, accepted, rng, deadline, stop_flag)
    raise ValueError(f"unknown sudoku method {method!r}")


# ---------------------------------------------------------------------------
# Public SA / ILS
# ---------------------------------------------------------------------------


def sudoku_sa(order, T_start=20.0, T_end=0.01, cooling=0.9995,
              max_steps=20000, rng=None, callback=None,
              stop_flag=None, start=None, method="stochastic"):
    """Stochastic + residual SA.  Frames match tile/Gerzon SA.

    ``method`` ``residual`` uses only relational-residual repair;
    anything else is the Wikipedia stochastic family (SA + tabu).
    ``start`` is an optional ±1 warm start.
    """
    rng = rng or np.random.default_rng()
    if start is not None:
        H0 = np.array(start, dtype=np.int8, copy=True)
        if H0.shape != (order, order):
            raise ValueError(f"start shape {H0.shape} != ({order}, {order})")
        H0 = normalize(H0)
    else:
        H0 = random_seed(order, rng).astype(np.int8)
    n = int(H0.shape[0])
    if n < 1:
        raise ValueError("sudoku_sa needs order ≥ 1")
    if n == 1:
        info = dict(steps=0, accepts=0, best_E=0.0, elapsed_s=0.0,
                    hadamard=True, method="backtrack", E_res=0.0)
        return H0, info

    rows = [pack_row(H0[i]) for i in range(n)]
    deadline = None
    t0 = time.monotonic()
    if method == "residual":
        best, st = _residual_improve(
            rows, n, rng, deadline, stop_flag,
            steps=max_steps, callback=callback,
        )
    else:
        best, st = _stochastic_improve(
            rows, n, rng, deadline, stop_flag,
            steps=max_steps, T_start=T_start, cooling=cooling,
            callback=callback, T_end=T_end,
        )
    H = _matrix_from_rows(best, n)
    st = dict(st)
    st.setdefault("method", method)
    st["E_res"] = st.get("best_E", residual_energy_rows(best, n))
    st["elapsed_s"] = time.monotonic() - t0
    st["hadamard"] = bool(st.get("hadamard") or verify(H))
    return H, st


def sudoku_ils(order, T_start=20.0, cooling=0.9995, sa_steps=15000,
               restarts=5, time_budget=None, rng=None,
               stop_flag=None, progress_callback=None, method=None):
    """ILS wrapper: cycle the Wikipedia set until budget / stop.

    ``method`` ``None`` / ``auto`` / ``portfolio`` cycles ``METHODS``.
    A single name pins that technique.  Deterministic solvers
    (exact / backtrack / csp / overlay) run first on small orders;
    residual + stochastic keep going on large ones.  Same contract
    as ``gerzon_ils`` / ``tile_ils``.
    """
    rng = rng or np.random.default_rng()
    n = int(order)
    if method in (None, "auto", "portfolio"):
        cycle = list(METHODS)
    else:
        if method not in METHODS:
            raise ValueError(f"unknown sudoku method {method!r} (have {METHODS})")
        cycle = [method]
    best_H = None
    best_f = None
    it = 0
    t0 = time.monotonic()
    while True:
        if stop_flag is not None and stop_flag.is_set():
            break
        if time_budget and time.monotonic() - t0 > time_budget:
            break
        if time_budget is None and it >= restarts:
            break
        m = cycle[it % len(cycle)]
        deadline = (t0 + time_budget) if time_budget else None
        accepted = [_all_plus(n)]
        H = None
        if m in ("exact", "backtrack", "csp", "overlay"):
            rows = _complete(m, n, accepted, rng, deadline, stop_flag,
                             node_limit=sa_steps)
            if rows is not None and len(rows) == n:
                H = _matrix_from_rows(rows, n)
        if H is None:
            # fall through to a stochastic / residual burst so every
            # iteration still moves the live preview
            use = "residual" if m == "residual" else "stochastic"
            H, _st = sudoku_sa(
                n, T_start=T_start, cooling=cooling, max_steps=sa_steps,
                rng=rng, callback=progress_callback, stop_flag=stop_flag,
                method=use,
            )
        f = residual_energy(H)
        if best_H is None or f < best_f:
            best_H = H.copy()
            best_f = f
        if progress_callback is not None:
            progress_callback({
                "iter": it, "f": f, "best_f": best_f,
                "elapsed_s": time.monotonic() - t0,
                "method": m, "E_res": f,
                "H": best_H,
            })
        if best_f is not None and best_f < 1e-9:
            break
        it += 1
        T_start = min(30.0, T_start * 1.3)
    return best_H, best_f, (best_H is not None and verify(best_H))


if __name__ == "__main__":
    from .hadamard import sylvester, check

    H4 = sylvester(4)
    a = analyze(H4)
    assert a["F"] == 0.0 and a["n_ortho_pairs"] == 6, a
    assert a["gauge_ok"] and a["balance_ok"]
    print("PASS  analyze(H4): F=0, 6 orthogonal pairs")

    rng = np.random.default_rng(0)
    t4 = enumerate_templates(4)
    assert len(t4) == 3
    rows = _exact_rows(4, [_all_plus(4)], rng, None, None, 50_000)
    assert rows is not None and verify(_matrix_from_rows(rows, 4))
    print("PASS  exact-cover H4")

    rows = _backtrack_rows(4, [_all_plus(4)], rng, None, None, 50_000)
    assert rows is not None and verify(_matrix_from_rows(rows, 4))
    print("PASS  backtrack H4")

    rows = _csp_rows(8, [_all_plus(8)], rng, None, None, 80_000)
    assert rows is not None and verify(_matrix_from_rows(rows, 8)), "csp H8 failed"
    print("PASS  CSP overlay-domain H8")

    rows = _overlay_rows(8, [_all_plus(8)], rng, None, None, 80_000)
    assert rows is not None and verify(_matrix_from_rows(rows, 8))
    print("PASS  pattern-overlay H8")

    H, info = sudoku_sa(4, max_steps=3000, rng=np.random.default_rng(1))
    assert info["accepts"] > 0
    print(f"PASS  SA H4 accepts={info['accepts']} E={info['best_E']:.2f}")

    H, f, ok = sudoku_ils(4, sa_steps=5000, restarts=4,
                          rng=np.random.default_rng(2))
    assert ok and verify(H) and f == 0.0, (ok, f)
    print("PASS  ILS H4 is Hadamard")

    H8 = sylvester(8)
    a8 = analyze(H8)
    assert a8["F"] == 0.0 and a8["n_ortho_pairs"] == 28
    print("PASS  analyze(H8)")

    print("sudoku self-check OK")
