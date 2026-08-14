"""Williamson / Goethals-Seidel sequence search for Hadamard matrices.

For order n = 4k, search four symmetric +/-1 sequences (a,b,c,d) of length k
whose power-spectral-density sum equals 4k at every frequency.  The resulting
circulants A,B,C,D satisfy A^2+B^2+C^2+D^2 = 4k I_k and assemble into an
n x n Hadamard (Williamson layout).  A Goethals-Seidel variant includes
anti-diagonal reflections R for additional generality.

Search uses FFT-based PSD (O(k log k) per flip) and single-flip greedy
descent over the ~2k free variables, with ILS restart outer loop.
"""

from __future__ import annotations

import math, time
import numpy as np


def circulant_random(k: int, rng=None) -> np.ndarray:
    """Random +/-1 sequence for a general (non-symmetric) circulant matrix."""
    rng = rng or np.random.default_rng()
    return rng.choice([-1.0, 1.0], size=k).astype(np.float64)


def symmetric_random(k: int, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    half = k // 2
    n_free = half + 1               # positions 0..half (free under symmetry)
    a = rng.choice([-1.0, 1.0], size=n_free)
    full = np.empty(k, dtype=np.float64)
    full[0] = a[0]
    for t in range(1, half + 1):
        full[t] = a[t]
        full[k - t] = a[t]
    return full


def _flip_sequence(seq, i):
    seq[i] *= -1.0
    k = len(seq)
    if i != 0:
        j = k - i
        if j != i:
            seq[j] *= -1.0


def _fft_power(s):
    return np.abs(np.fft.fft(s)) ** 2


def psd_objective(a, b, c, d):
    target = 4.0 * len(a)
    t = _fft_power(a) + _fft_power(b) + _fft_power(c) + _fft_power(d)
    return float(np.sum((t - target) ** 2))


def seq_to_circulant(s):
    k = len(s)
    M = np.empty((k, k), dtype=np.int64)
    for i in range(k):
        M[i] = np.roll(s, i)
    return M


def williamson_assemble(a, b, c, d):
    k = len(a)
    A = seq_to_circulant(a)
    B = seq_to_circulant(b)
    C = seq_to_circulant(c)
    D = seq_to_circulant(d)
    top = np.concatenate([A, B, C, D], axis=1)
    m2 = np.concatenate([-B, A, -D, C], axis=1)
    m3 = np.concatenate([-C, D, A, -B], axis=1)
    m4 = np.concatenate([-D, -C, B, A], axis=1)
    return np.concatenate([top, m2, m3, m4], axis=0).astype(np.int8)


def gs_assemble(a, b, c, d):
    k = len(a)
    A = seq_to_circulant(a)
    B = seq_to_circulant(b)
    C = seq_to_circulant(c)
    D = seq_to_circulant(d)
    R = np.eye(k, dtype=np.int64)[::-1]
    BR = B @ R; BTR = B.T @ R
    CR = C @ R; CTR = C.T @ R
    DR = D @ R; DTR = D.T @ R
    top  = np.concatenate([A, BR, CR, DR], axis=1)
    r2   = np.concatenate([-BR, A, -DTR, CTR], axis=1)
    r3   = np.concatenate([-CR, DTR, A, -BTR], axis=1)
    r4   = np.concatenate([-DR, -CTR, BTR, A], axis=1)
    return np.concatenate([top, r2, r3, r4], axis=0).astype(np.int8)


def williamson_search(k, max_flips=50000, tol=1.0, rng=None, start=None,
                      callback=None, stop_flag=None, report_every=500):
    """Single-flip greedy PSD descent. Returns (a,b,c,d,stats).

    Optional streaming hooks (zero behavior change when omitted):
    callback(dict) is called every report_every flips with
    {"step", "f", "best_f", "H"} (H = current best matrix, for live
    previews; never serialized); stop_flag (threading.Event) breaks the
    descent early when set, returning the current best.
    """
    rng = rng or np.random.default_rng()
    target = 4.0 * float(k)
    if start is None:
        seqs = [symmetric_random(k, rng) for _ in range(4)]
    else:
        seqs = [np.array(s, dtype=np.float64, copy=True) for s in start]
    pows = [_fft_power(s) for s in seqs]
    tot = sum(pows)
    f_cur = float(np.sum((tot - target) ** 2))

    best = [s.copy() for s in seqs]
    best_f = f_cur
    flips = 0
    half = k // 2 + 1
    t0 = time.monotonic()

    while flips < max_flips:
        best_delta = 0.0
        best_move = None
        for si, (seq, pw) in enumerate(zip(seqs, pows)):
            for idx in range(half):
                _flip_sequence(seq, idx)
                pw_new = _fft_power(seq)
                tot_new = tot - pw + pw_new
                fn = float(np.sum((tot_new - target) ** 2))
                delta = fn - f_cur
                _flip_sequence(seq, idx)
                if delta < best_delta - 1e-12:
                    best_delta = delta
                    best_move = (si, idx, fn, pw_new)
        if best_delta > -tol or best_move is None:
            break
        si, idx, fn, pw_new = best_move
        _flip_sequence(seqs[si], idx)
        pows[si][:] = pw_new
        tot = sum(pows)
        f_cur = fn
        flips += 1
        if fn < best_f:
            best_f = fn
            best = [s.copy() for s in seqs]
        if callback is not None and flips % report_every == 0:
            callback({"step": flips, "f": f_cur, "best_f": best_f,
                      "H": williamson_assemble(*best)})
        if stop_flag is not None and stop_flag.is_set():
            break

    return (*best, dict(
        f=best_f, flips=flips, is_williamson=(best_f < 1e-6),
        elapsed_s=time.monotonic() - t0))


def williamson_ils(k, inner_flips=20000, outer_iters=20, time_budget=None,
                   frac=0.05, rng=None, stop_flag=None, progress_callback=None):
    """ILS outer loop: perturb + greedy descent repeats.

    stop_flag (threading.Event, optional): checked at each outer-iteration
    boundary (and forwarded to the inner descent) — break out early and
    return the current best when set.
    progress_callback (optional): called once per outer iteration with
    {"iter", "f", "best_f", "elapsed_s"} and forwarded to the inner descent
    (per-500-flip frames) so long descents stream live progress.
    """
    rng = rng or np.random.default_rng()
    half = k // 2 + 1
    best_seq = None
    best_f = None
    it = 0
    t0 = time.monotonic()
    while it < outer_iters:
        if stop_flag is not None and stop_flag.is_set():
            break
        if time_budget and time.monotonic() - t0 > time_budget:
            break
        if best_seq is not None and it > 0:
            n_pert = max(1, int(frac * half * 4))
            seqs = [s.copy() for s in best_seq]
            for _ in range(n_pert):
                si = int(rng.integers(0, 4))
                idx = int(rng.integers(0, half))
                _flip_sequence(seqs[si], idx)
        else:
            seqs = [symmetric_random(k, rng) for _ in range(4)]
        a,b,c,d,st = williamson_search(
            k, max_flips=inner_flips, start=seqs, stop_flag=stop_flag,
            callback=progress_callback)
        if best_seq is None or st["f"] < best_f:
            best_seq = (a.copy(), b.copy(), c.copy(), d.copy())
            best_f = st["f"]
        if progress_callback is not None:
            progress_callback({"iter": it, "f": st["f"], "best_f": best_f,
                               "elapsed_s": time.monotonic() - t0})
        it += 1
    if best_seq is None:  # cancelled before the first descent finished
        return None, "cancelled", None, False
    a,b,c,d = best_seq
    H, method = williamson_to_hadamard(k, a, b, c, d)
    from .hadamard import verify
    return H, method, best_f, verify(H)


def gs_circulant_search(k: int, max_flips=50000, tol=1.0, rng=None, start=None,
                        callback=None, stop_flag=None, report_every=500):
    """Single-flip greedy PSD descent for general circulant GS sequences.

    No symmetry constraint — any +/-1 sequence produces a circulant matrix.
    Circulant matrices automatically commute, satisfying the GS condition.
    Returns (a,b,c,d,stats).

    Optional streaming hooks (zero behavior change when omitted):
    callback(dict) is called every report_every flips with
    {"step", "f", "best_f", "H"} (H = current best matrix, for live
    previews; never serialized); stop_flag (threading.Event) breaks the
    descent early when set, returning the current best.
    """
    rng = rng or np.random.default_rng()
    target = 4.0 * float(k)
    if start is None:
        seqs = [circulant_random(k, rng) for _ in range(4)]
    else:
        seqs = [np.array(s, dtype=np.float64, copy=True) for s in start]
    pows = [_fft_power(s) for s in seqs]
    tot = sum(pows)
    f_cur = float(np.sum((tot - target) ** 2))

    best = [s.copy() for s in seqs]
    best_f = f_cur
    flips = 0
    t0 = time.monotonic()

    while flips < max_flips:
        best_delta = 0.0
        best_move = None
        for si, (seq, pw) in enumerate(zip(seqs, pows)):
            for idx in range(k):
                seq[idx] *= -1.0
                pw_new = _fft_power(seq)
                tot_new = tot - pw + pw_new
                fn = float(np.sum((tot_new - target) ** 2))
                delta = fn - f_cur
                seq[idx] *= -1.0  # undo
                if delta < best_delta - 1e-12:
                    best_delta = delta
                    best_move = (si, idx, fn, pw_new)
        if best_delta > -tol or best_move is None:
            break
        si, idx, fn, pw_new = best_move
        seqs[si][idx] *= -1.0
        pows[si][:] = pw_new
        tot = sum(pows)
        f_cur = fn
        flips += 1
        if fn < best_f:
            best_f = fn
            best = [s.copy() for s in seqs]
        if callback is not None and flips % report_every == 0:
            callback({"step": flips, "f": f_cur, "best_f": best_f,
                      "H": williamson_assemble(*best)})
        if stop_flag is not None and stop_flag.is_set():
            break
    return (*best, dict(
        f=best_f, flips=flips, is_gs=(best_f < 1e-6),
        elapsed_s=time.monotonic() - t0))


def gs_circulant_ils(k, inner_flips=20000, outer_iters=20, time_budget=None,
                     frac=0.05, rng=None, stop_flag=None, progress_callback=None):
    """ILS outer loop for general circulant GS search.

    stop_flag (threading.Event, optional): checked at each outer-iteration
    boundary (and forwarded to the inner descent) — break out early and
    return the current best when set.
    progress_callback (optional): called once per outer iteration with
    {"iter", "f", "best_f", "elapsed_s"} and forwarded to the inner descent
    (per-500-flip frames) so long descents stream live progress.
    """
    rng = rng or np.random.default_rng()
    best_seq = None; best_f = None; it = 0; t0 = time.monotonic()
    while it < outer_iters:
        if stop_flag is not None and stop_flag.is_set(): break
        if time_budget and time.monotonic() - t0 > time_budget: break
        if best_seq is not None and it > 0:
            n_pert = max(1, int(frac * k * 4))
            seqs = [s.copy() for s in best_seq]
            for _ in range(n_pert):
                si = int(rng.integers(0, 4))
                idx = int(rng.integers(0, k))
                seqs[si][idx] *= -1.0
        else:
            seqs = [circulant_random(k, rng) for _ in range(4)]
        a,b,c,d,st = gs_circulant_search(k, max_flips=inner_flips, tol=0.0, start=seqs,
                                         stop_flag=stop_flag,
                                         callback=progress_callback)
        if best_seq is None or st["f"] < best_f:
            best_seq = (a.copy(), b.copy(), c.copy(), d.copy())
            best_f = st["f"]
        if progress_callback is not None:
            progress_callback({"iter": it, "f": st["f"], "best_f": best_f,
                               "elapsed_s": time.monotonic() - t0})
        it += 1
    if best_seq is None:  # cancelled before the first descent finished
        return None, "cancelled", None, False
    a,b,c,d = best_seq
    H, method = williamson_to_hadamard(k, a, b, c, d)
    from .hadamard import verify
    return H, method, best_f, verify(H)


def williamson_to_hadamard(k, a, b, c, d, use_gs=True):
    from .hadamard import verify
    Hw = williamson_assemble(a, b, c, d)
    if verify(Hw):
        return Hw, "williamson"
    if use_gs:
        Hg = gs_assemble(a, b, c, d)
        if verify(Hg):
            return Hg, "goethals-seidel"
    return Hw, "unverified"


def search_order(order, max_flips=50000, ils_iters=5, rng=None):
    if order % 4 != 0:
        raise ValueError("order must be 4k")
    k = order // 4
    a,b,c,d,st = williamson_search(k, max_flips=max_flips, rng=rng)
    if st["f"] < 1e-6:
        H, method = williamson_to_hadamard(k, a, b, c, d)
        return H, method, st
    if ils_iters > 0:
        H, method, f, verified = williamson_ils(k, inner_flips=max_flips,
                                                 outer_iters=ils_iters, rng=rng)
        if verified:
            return H, method, dict(f=f, is_williamson=True)
    return None, None, st
