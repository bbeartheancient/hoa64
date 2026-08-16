"""Hadamard generator / verifier / max-determinant search.

Bitset core: a row is packed into a Python int with bit j == +1.  Then
  dot(a, b)   = n - 2 * popcount(a ^ b)
  row_sum(a)  = 2 * popcount(a) - n
  H2 type counts between two rows are and/xor popcounts.  In a normalized
  Hadamard matrix any two rows have each of (++, +-, -+, --) exactly n/4
  times, so the columns tile into Hadamard-2 cells.

Search: normalize to first row AND first column all +1 (row/column sign
flips).  For G = H H^T and M = (G - n I) H, flipping a single interior cell
(i, j) from s to -s changes F = sum_{i<j} G_ij^2 by
  dF = -4 s M[i, j] + 4 (n - 1).
Minimizing F is Hadamard's maximal determinant problem: max |det| = n^(n/2)
iff H is a Hadamard matrix.

Constructions: Sylvester (orders 2^k), Paley I (q + 1, q == 3 mod 4 prime),
Paley II (2(q + 1), q == 1 mod 4 prime), and Kronecker products of these.
The channel-count coincidence with ambisonics is real: Gerzon's 1975 AB
module is a Hadamard matrix (rows WXYZ, columns L_B L_F R_F R_B); swapping
the L_F and L_B columns yields Sylvester H₄.  N = 1, 3, 7 give orders
4, 16, 64 (hoa64 = Sylvester H-64).  See ``gerzon.py``.

Smallest open order is 668 = 4 * 167 (167 prime == 7 mod 32).  The best known
approximation is Eliahou's 64-modular H(668) with H H^T == 668 I (mod 64).
If a copy is saved as data/h668_mod64.npy or ~/.cache/h668_mod64.npy it is
used as a warm start for the search.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

__all__ = [
    "pack_row",
    "unpack_row",
    "dot_bits",
    "row_sum_bits",
    "h2_types_bits",
    "h2_stats",
    "check",
    "verify",
    "normalize",
    "det_bound_log10",
    "det_log10",
    "modular_check",
    "sylvester",
    "paley",
    "hadamard_product",
    "hadamard_known",
    "hadamard_orders",
    "random_seed",
    "local_search",
    "perturb",
    "ils_search",
    "save_npy",
    "load_npy",
    "load_modular_seed",
    "default_seeds",
    "selftest",
    "OPEN_ORDERS",
]

OPEN_ORDERS = []


def pack_row(row: np.ndarray) -> int:
    bits = 0
    for j, s in enumerate(np.ravel(row)):
        if s == 1:
            bits |= 1 << j
    return bits


def unpack_row(bits: int, n: int) -> np.ndarray:
    return np.array([1 if (bits >> j) & 1 else -1 for j in range(n)], dtype=np.int8)


def dot_bits(a: int, b: int, n: int) -> int:
    return n - 2 * (a ^ b).bit_count()


def row_sum_bits(a: int, n: int) -> int:
    return 2 * a.bit_count() - n


def h2_types_bits(a: int, b: int, n: int) -> tuple[int, int, int, int]:
    mask = (1 << n) - 1
    na = (~a) & mask
    nb = (~b) & mask
    return (
        (a & b).bit_count(),
        (a & nb).bit_count(),
        (na & b).bit_count(),
        (na & nb).bit_count(),
    )


def h2_stats(H: np.ndarray, pairs: list[tuple[int, int]] | None = None) -> list[dict]:
    n = int(H.shape[0])
    rows = [pack_row(H[i]) for i in range(n)]
    if pairs is None:
        pairs = [(i, i + 1) for i in range(1, n - 1)]
    out = []
    for i, j in pairs:
        t = h2_types_bits(rows[i], rows[j], n)
        out.append(
            dict(pair=(i, j), types=t, balanced=all(v == n // 4 for v in t))
        )
    return out


def det_bound_log10(n: int) -> float:
    return 0.5 * n * math.log10(n)


def det_log10(H: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(H.astype(np.float64))
    if sign == 0:
        return -math.inf
    return float(logdet * math.log10(math.e))


def normalize(H: np.ndarray) -> np.ndarray:
    Hn = np.array(H, dtype=np.int8)
    colflip = np.where(Hn[0] == -1)[0]
    if colflip.size:
        Hn[:, colflip] *= -1
    rowflip = np.where(Hn[:, 0] == -1)[0]
    if rowflip.size:
        Hn[rowflip, :] *= -1
    return Hn


def check(H: np.ndarray, det: bool = False) -> dict:
    A = np.asarray(H)
    n = int(A.shape[1])
    res = dict(valid=True, n=n, is_sign=bool(np.all((A == 1) | (A == -1))))
    if not res["is_sign"]:
        res.update(is_hadamard=False, reason="entries not all +-1")
        return res
    G = A.astype(np.float64) @ A.astype(np.float64).T
    off = G - n * np.eye(n, dtype=np.float64)
    res.update(
        max_off=int(np.abs(off).max()),
        diag_max_err=int(np.abs(np.diag(G) - n).max()),
        f=int((np.sum(G * G) - n**3) // 2),
        max_row_sum=int(np.abs(A.sum(axis=1)).max()),
        det_bound_log10=det_bound_log10(n),
        det_bound=det_bound_log10(n),
    )
    if det:
        res["det_log10"] = det_log10(A)
    h2 = h2_stats(A)
    res["h2_all_balanced"] = bool(np.all([r["balanced"] for r in h2]))
    res["is_hadamard"] = bool(
        res["max_off"] == 0 and res["diag_max_err"] == 0
    )
    return res


def _verify_fast(H: np.ndarray) -> bool:
    A = np.asarray(H)
    G = A.astype(np.float64) @ A.astype(np.float64).T
    n = A.shape[0]
    return bool(
        np.abs(np.diag(G) - n).max() < 0.5
        and np.abs(G - n * np.eye(n, dtype=np.float64)).max() < 0.5
    )


def verify(H: np.ndarray) -> bool:
    return check(H)["is_hadamard"]


def modular_check(H: np.ndarray, m: int) -> dict:
    n = int(H.shape[0])
    R = H.astype(np.int64) @ H.astype(np.int64).T
    rem = np.abs(R - n * np.eye(n, dtype=np.int64)) % m
    return dict(ok=bool(np.all(rem == 0)), mod=m, max_residue=int(rem.max()))


def _factor(q: int) -> list[int]:
    fac = []
    d = 2
    n = q
    while d * d <= n:
        while n % d == 0:
            fac.append(d)
            n //= d
        d += 1
    if n > 1:
        fac.append(n)
    return fac


def _is_prime(q: int) -> bool:
    facs = _factor(q)
    return q >= 2 and len(facs) == 1 and facs[0] == q


def _is_prime_power(q: int) -> tuple[int, int] | None:
    facs = _factor(q)
    if len(facs) == 0:
        return None
    p = facs[0]
    if all(f == p for f in facs):
        return (p, len(facs))
    return None


def _legendre(q: int) -> np.ndarray:
    chi = np.full(q, -1, dtype=np.int64)
    chi[0] = 0
    for x in range(1, q):
        chi[(x * x) % q] = 1
    return chi


def sylvester(n: int) -> np.ndarray | None:
    if n < 1 or (n & (n - 1)) != 0:
        return None
    H = np.array([[1]], dtype=np.int8)
    while H.shape[0] < n:
        top = np.concatenate([H, H], axis=1)
        bot = np.concatenate([H, -H], axis=1)
        H = np.concatenate([top, bot], axis=0)
    return H


def _legendre_table_finite_field(q: int, p: int, e: int) -> np.ndarray:
    from .finite_field import legendre_symbol
    return np.array([legendre_symbol(a, p, e) for a in range(q)], dtype=np.int64)


def _field_ops(q, p, e):
    """Build addition/subtraction index tables for F_{p^e}."""
    from .finite_field import _poly_from_int, _poly_to_int
    elems = [_poly_from_int(i, p, e) for i in range(q)]
    add = np.zeros((q, q), dtype=np.int64)
    sub = np.zeros((q, q), dtype=np.int64)
    for i in range(q):
        for j in range(q):
            res_add = [(a + b) % p for a, b in zip(elems[i], elems[j])]
            res_sub = [(a - b) % p for a, b in zip(elems[i], elems[j])]
            add[i, j] = _poly_to_int(res_add, p)
            sub[i, j] = _poly_to_int(res_sub, p)
    return add, sub


def _paley_type_i_generic(q: int, p: int, e: int) -> np.ndarray:
    chi = _legendre_table_finite_field(q, p, e)
    _, sub = _field_ops(q, p, e)
    r = np.arange(q)
    Q = chi[sub[r[:, None], r[None, :]]]
    H = np.zeros((q + 1, q + 1), dtype=np.int64)
    H[0, :] = 1
    H[:, 0] = 1
    H[1:, 1:] = Q - np.eye(q, dtype=np.int64)
    return normalize(H.astype(np.int8))


def _paley_type_ii_generic(q: int, p: int, e: int) -> np.ndarray:
    chi = _legendre_table_finite_field(q, p, e)
    _, sub = _field_ops(q, p, e)
    r = np.arange(q)
    Qsub = chi[sub[r[:, None], r[None, :]]]
    v = q + 1
    C = np.zeros((v, v), dtype=np.int64)
    C[1:, 1:] = Qsub
    C[1:, 0] = 1
    C[0, 1:] = 1
    I = np.eye(v, dtype=np.int64)
    top = np.concatenate([C + I, C - I], axis=1)
    bot = np.concatenate([C - I, -(C + I)], axis=1)
    return normalize(np.concatenate([top, bot], axis=0).astype(np.int8))


def _paley_type_i(q: int) -> np.ndarray:
    chi = _legendre(q)
    r = np.arange(q)
    D = (r[None, :] - r[:, None]) % q
    Q = chi[D]
    H = np.zeros((q + 1, q + 1), dtype=np.int64)
    H[0, :] = 1
    H[:, 0] = 1
    H[1:, 1:] = Q - np.eye(q, dtype=np.int64)
    return normalize(H.astype(np.int8))


def _paley_type_ii(q: int) -> np.ndarray:
    chi = _legendre(q)
    v = q + 1
    C = np.zeros((v, v), dtype=np.int64)
    r = np.arange(q)
    D = (r[None, :] - r[:, None]) % q
    C[1:, 1:] = chi[D]
    C[1:, 0] = 1
    C[0, 1:] = 1
    I = np.eye(v, dtype=np.int64)
    top = np.concatenate([C + I, C - I], axis=1)
    bot = np.concatenate([C - I, -(C + I)], axis=1)
    return normalize(np.concatenate([top, bot], axis=0).astype(np.int8))


def _paley_from_q(q: int, is_type_ii: bool) -> np.ndarray | None:
    if is_type_ii:
        if q < 5 or q % 4 != 1:
            return None
    else:
        if q < 3 or q % 4 != 3:
            return None
    if _is_prime(q):
        return _paley_type_i(q) if not is_type_ii else _paley_type_ii(q)
    pp = _is_prime_power(q)
    if pp is not None:
        p, e = pp
        try:
            return (_paley_type_i_generic(q, p, e) if not is_type_ii
                    else _paley_type_ii_generic(q, p, e))
        except RuntimeError:
            pass
    return None


def paley(n: int) -> np.ndarray | None:
    if n < 4 or n % 4 != 0:
        return None
    q = n - 1
    H = _paley_from_q(q, is_type_ii=False)
    if H is not None:
        return H
    if n % 2 == 0:
        q2 = n // 2 - 1
        return _paley_from_q(q2, is_type_ii=True)
    return None


def hadamard_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.kron(A, B).astype(np.int8)


_HAD_FOUND: dict[int, np.ndarray] = {}
_HAD_FAILED: set[int] = set()


def hadamard_known(n: int) -> np.ndarray | None:
    if n in _HAD_FOUND:
        return _HAD_FOUND[n]
    if n in _HAD_FAILED:
        return None
    res = None
    if n == 1:
        res = np.array([[1]], dtype=np.int8)
    elif n == 2:
        res = np.array([[1, 1], [1, -1]], dtype=np.int8)
    elif (n & (n - 1)) == 0:
        res = sylvester(n)
    else:
        res = paley(n)
        if res is None:
            for d in range(2, int(math.isqrt(n)) + 1):
                if n % d:
                    continue
                A = hadamard_known(d)
                B = hadamard_known(n // d)
                if A is not None and B is not None:
                    res = hadamard_product(A, B)
                    break
    if res is None:
        p = Path.home() / "open_hadamard" / f"hadamard_{n}.csv"
        if p.is_file():
            H = np.loadtxt(str(p), delimiter=",", dtype=np.int8)
            res = normalize(H)

    if res is None and n % 4 == 0:
        q = n // 4
        if q >= 5 and q % 4 == 1 and _is_prime_power(q):
            if hadamard_known(q - 1) is not None:
                from .miyamoto import miyamoto_construction
                try:
                    H_base = hadamard_known(q - 1)
                    res = miyamoto_construction(q, H_base)
                    res = normalize(res)
                except Exception:
                    pass
    if res is not None:
        if _verify_fast(res):
            _HAD_FOUND[n] = res
        else:
            res = None
            _HAD_FAILED.add(n)
    else:
        _HAD_FAILED.add(n)
    return res


def hadamard_orders(N: int) -> list[int]:
    return [n for n in range(1, N + 1) if hadamard_known(n) is not None]


def random_seed(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    if n <= 1:
        return np.array([[1]], dtype=np.int8)
    H = np.ones((n, n), dtype=np.int8)
    half = n // 2
    for i in range(1, n):
        cells = rng.choice(n - 1, size=half, replace=False) + 1
        H[i, cells] = -1
    return H


def make_stats(H: np.ndarray) -> dict:
    n = int(H.shape[0])
    G = H.astype(np.int64) @ H.astype(np.int64).T
    off = G - n * np.eye(n, dtype=np.int64)
    return dict(
        f=int((np.sum(G * G) - n**3) // 2),
        max_off=int(np.abs(off).max()),
        det_bound_log10=det_bound_log10(n),
        gauge_ok=bool(np.all(H[0] == 1) and np.all(H[:, 0] == 1)),
        diag_ok=bool(np.all(np.diag(G) == n)),
    )


def _apply_flip(H, HT, G, M, i, j, n):
    s = int(H[i, j])
    oldrow = G[i].copy()
    oldrow_i = H[i].astype(np.int64).copy()
    H[i, j] = -s
    HT[j, i] = -s
    G[i] = H[i].astype(np.int64) @ HT
    G[:, i] = G[i]
    M[i] = G[i] @ H - n * H[i].astype(np.int64)
    kk = np.arange(n) != i
    v = H[kk, j].astype(np.int64)
    M[kk, :] += np.outer(-2 * s * v, oldrow_i)
    M[kk, j] += -2 * s * oldrow[kk] + 4 * v
    return H, HT, G, M


def local_search(
    H: np.ndarray,
    max_flips: int = 100000,
    min_gain: int = 4,
    report_every: int = 500,
    callback=None,
    stop_flag=None,
) -> tuple[np.ndarray, dict]:
    n = int(H.shape[0])
    Hm = np.array(H, dtype=np.int8)
    HT = Hm.T.copy()
    G = Hm.astype(np.int64) @ HT
    M = (G - n * np.eye(n, dtype=np.int64)) @ Hm.astype(np.int64)
    flips = 0
    t0 = time.monotonic()
    while flips < max_flips:
        D = -4 * Hm * M + 4 * (n - 1)
        flat = D[1:, 1:]
        idx = int(np.argmin(flat))
        dmin = int(flat.ravel()[idx])
        if dmin > -min_gain:
            break
        i = 1 + idx // (n - 1)
        j = 1 + idx % (n - 1)
        Hm, HT, G, M = _apply_flip(Hm, HT, G, M, i, j, n)
        flips += 1
        if flips % report_every == 0:
            if callback:
                st = make_stats(Hm)
                st["H"] = Hm
                callback(st)
            if stop_flag is not None and stop_flag.is_set():
                break
    st = make_stats(Hm)
    st["flips"] = flips
    st["elapsed_s"] = time.monotonic() - t0
    return Hm, st


def perturb(H: np.ndarray, rng: np.random.Generator | None = None, frac: float = 0.05) -> np.ndarray:
    rng = rng or np.random.default_rng()
    n = int(H.shape[0])
    k = max(1, int(frac * (n - 1) * (n - 1)))
    cells = rng.choice((n - 1) * (n - 1), size=k, replace=False)
    P = np.array(H, dtype=np.int8)
    sub = P[1:, 1:]
    sub[cells // (n - 1), cells % (n - 1)] *= -1
    return P


def default_seeds(order: int, rng: np.random.Generator | None = None) -> list[np.ndarray]:
    rng = rng or np.random.default_rng()
    seeds = []
    known = hadamard_known(order)
    if known is not None:
        seeds.append(known)
    mod = load_modular_seed(order)
    if mod is not None:
        seeds.append(mod)
    seeds.append(random_seed(order, rng))
    return seeds


def ils_search(
    order: int,
    seeds: list[np.ndarray] | None = None,
    inner_flips: int = 100000,
    outer_iters: int = 20,
    time_budget: float | None = None,
    frac: float = 0.05,
    seed_int: int | None = None,
    print_progress: bool = True,
    iter_callback=None,
    stop_flag=None,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed_int)
    if seeds is None:
        seeds = default_seeds(order, rng)
    best = None
    bestf = None
    it = 0
    t0 = time.monotonic()
    while True:
        if stop_flag is not None and stop_flag.is_set():
            break
        if time_budget is not None and time.monotonic() - t0 > time_budget:
            break
        if time_budget is None and it >= outer_iters:
            break
        if best is not None and it > 0:
            H0 = perturb(best, rng, frac)
        else:
            H0 = np.array(seeds[it % len(seeds)], dtype=np.int8)

        def _cb(st):
            print(
                f"  flips@{st['f']} maxoff={st['max_off']} "
                f"bound_log10={st['det_bound_log10']:.3f}"
            )

        def _stream(st):
            # inner-descent frames: keep the console log and stream live
            # stats to iter_callback so long flips don't sit silent
            if print_progress:
                _cb(st)
            if iter_callback is not None:
                iter_callback(st)

        H, st = local_search(
            H0,
            max_flips=inner_flips,
            report_every=max(500, inner_flips // (50 if iter_callback is not None else 10)),
            callback=_stream if (print_progress or iter_callback is not None) else None,
            stop_flag=stop_flag,
        )
        if print_progress:
            print(f"[iter {it}] f={st['f']} maxoff={st['max_off']} flips={st['flips']}")
        if best is None or st["f"] < bestf:
            best = H
            bestf = st["f"]
        if iter_callback is not None:
            iter_callback(
                {
                    "iter": it,
                    "f": st["f"],
                    "best_f": bestf,
                    "det_log10": det_log10(best) if best.shape[0] <= 500 else None,
                    "is_hadamard": bool(bestf == 0),
                    "H": best,
                }
            )
        it += 1
    bestst = make_stats(best)
    n = int(best.shape[0])
    bestst.update(
        iters=it,
        elapsed_s=time.monotonic() - t0,
        det_log10=det_log10(best) if n <= 500 else None,
        is_hadamard=verify(best),
    )
    return best, bestst


_MODULAR668_FILES = [
    Path(__file__).with_name("data") / "h668_mod64.npy",
    Path.home() / ".cache" / "h668_mod64.npy",
]


def load_modular_seed(order: int = 668) -> np.ndarray | None:
    for p in _MODULAR668_FILES:
        if p.is_file():
            H = np.load(p)
            H = np.asarray(H, dtype=np.int8)
            if H.ndim == 2 and H.shape[0] == H.shape[1] == order:
                return normalize(H)
    return None


def save_npy(path, H: np.ndarray) -> None:
    np.save(Path(path), np.asarray(H, dtype=np.int8))


def load_npy(path) -> np.ndarray | None:
    try:
        return np.asarray(np.load(Path(path)), dtype=np.int8)
    except Exception:
        return None


def selftest() -> int:
    def expect(cond: bool, msg: str) -> None:
        if not cond:
            raise AssertionError(msg)

    for n in [1, 2, 4, 8, 16, 32, 64]:
        expect(verify(sylvester(n)), f"sylvester({n}) failed")
    for n in [4, 8, 12, 20, 44, 60]:
        expect(paley(n) is not None and verify(paley(n)), f"paley I order {n}")
    for n in [12, 28, 36, 60]:
        expect(verify(paley(n)), f"paley II order {n}")
    for n in [12, 20, 24, 28, 36, 44, 48, 60, 80, 96, 120, 240]:
        H = hadamard_known(n)
        expect(H is not None and verify(H), f"hadamard_known({n}) failed")
    expect(hadamard_known(6) is None, "order 6 should not be constructible")
    expect(hadamard_known(668) is not None and verify(hadamard_known(668)), "order 668 should resolve from CSV")
    expect(hadamard_known(716) is not None and verify(hadamard_known(716)), "order 716 should resolve from CSV")
    expect(hadamard_known(892) is not None and verify(hadamard_known(892)), "order 892 should resolve from CSV")

    expect(check(sylvester(8))["h2_all_balanced"], "H8 adjacent-row H2 tiling")

    rng = np.random.default_rng(0)
    H0 = random_seed(12, rng)
    expect(np.all(H0[0] == 1) and np.all(H0[:, 0] == 1), "seed gauge")
    expect(np.all(H0[1:].sum(axis=1) == 0), "seed row balance")

    n = 12
    H1 = random_seed(n, rng)
    HT1 = H1.T.copy()
    G1 = H1.astype(np.int64) @ HT1
    M1 = (G1 - n * np.eye(n, dtype=np.int64)) @ H1.astype(np.int64)
    Hc, HTc, Gc, Mc = H1.copy(), HT1.copy(), G1.copy(), M1.copy()
    for _ in range(30):
        i = int(rng.integers(1, n))
        j = int(rng.integers(1, n))
        Hc, HTc, Gc, Mc = _apply_flip(Hc, HTc, Gc, Mc, i, j, n)
        Gb = Hc.astype(np.int64) @ Hc.astype(np.int64).T
        Mb = (Gb - n * np.eye(n, dtype=np.int64)) @ Hc.astype(np.int64)
        expect(np.array_equal(Gc, Gb), "incremental G != brute force")
        expect(np.array_equal(Mc, Mb), "incremental M != brute force")

    s = int(H1[2, 3])
    dD = -4 * s * int(M1[2, 3]) + 4 * (n - 1)
    f_old = int((np.sum(G1 * G1) - n**3) // 2)
    H2 = H1.copy()
    H2[2, 3] *= -1
    G2 = H2.astype(np.int64) @ H2.astype(np.int64).T
    f_new = int((np.sum(G2 * G2) - n**3) // 2)
    expect(f_old + dD == f_new, "dF formula mismatch")

    P = perturb(sylvester(8), np.random.default_rng(1), frac=0.05)
    Hf, st = local_search(P, max_flips=50000)
    expect(np.all(Hf[0] == 1) and np.all(Hf[:, 0] == 1), "search broke gauge")
    expect(st["f"] == 0 and verify(Hf), "order-8 search failed to reach Hadamard")
    print("selftest: all checks passed")
    return 0