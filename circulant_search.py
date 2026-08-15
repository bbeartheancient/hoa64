#!/usr/bin/env python3
"""Circulant Hadamard evolver — start from H.4, search for H(m^2), m even.

A circulant Hadamard matrix needs a first row a where |DFT(a)[j]|^2 = n
for all frequencies.  This forces n = m^2 with row sum = +/- m.
"""

import os, sys, time, math
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hoa64.hadamard import verify

OUT = Path.home() / "open_hadamard"


def circulant_matrix(a):
    n = len(a)
    H = np.empty((n, n), dtype=np.int8)
    for i in range(n):
        H[i] = np.roll(a, i)
    return H


def psd_objective(a):
    target = float(len(a))
    pw = np.abs(np.fft.fft(a.astype(np.float64))) ** 2
    return float(np.sum((pw - target) ** 2))


def psd_search(n, max_flips=100000, tol=1.0, rng=None, start_seq=None,
               callback=None, stop_flag=None, report_every=500):
    """Single-flip greedy PSD descent for a single sequence of length n=m^2.

    Optional streaming hooks (zero behavior change when omitted):
    callback(dict) is called every report_every flips with
    {"step", "f", "best_f", "H"} (H = current best matrix, for live
    previews; never serialized); stop_flag (threading.Event) breaks the
    descent early when set, returning the current best.
    """
    rng = rng or np.random.default_rng()
    m = int(math.isqrt(n))
    n_plus = (n + m) // 2
    if start_seq is None:
        a = np.full(n, -1.0, dtype=np.float64)
        pos = rng.choice(n, size=n_plus, replace=False)
        a[pos] = 1.0
    else:
        a = np.array(start_seq, dtype=np.float64, copy=True)

    target = float(n)
    pw = np.abs(np.fft.fft(a)) ** 2
    f_cur = float(np.sum((pw - target) ** 2))
    best_a = a.copy(); best_f = f_cur
    flips = 0; t0 = time.monotonic()

    while flips < max_flips:
        best_delta = 0.0; best_move = None
        for idx in range(n):
            a[idx] *= -1.0
            pw_new = np.abs(np.fft.fft(a)) ** 2
            fn = float(np.sum((pw_new - target) ** 2))
            delta = fn - f_cur
            a[idx] *= -1.0
            if delta < best_delta - 1e-12:
                best_delta = delta; best_move = (idx, fn, pw_new)
        if best_delta > -tol or best_move is None:
            break
        idx, fn, pw_new = best_move
        a[idx] *= -1.0
        pw[:] = pw_new; f_cur = fn; flips += 1
        if fn < best_f:
            best_f = fn; best_a = a.copy()
        if callback is not None and flips % report_every == 0:
            callback({"step": flips, "f": f_cur, "best_f": best_f,
                      "H": circulant_matrix(np.round(best_a).astype(np.int8))})
        if stop_flag is not None and stop_flag.is_set():
            break
    return best_a, best_f, flips, time.monotonic() - t0


def search_ils(n, inner_flips=20000, outer_iters=30, time_budget=None,
               rng=None, frac=0.05, stop_flag=None, progress_callback=None):
    rng = rng or np.random.default_rng()
    m = int(math.isqrt(n)); n_plus = (n + m) // 2
    best_a = None; best_f = None; it = 0; t0 = time.monotonic()
    while True:
        if stop_flag is not None and stop_flag.is_set(): break
        if time_budget and time.monotonic() - t0 > time_budget: break
        if not time_budget and it >= outer_iters: break
        if best_a is not None and it > 0:
            a = np.round(best_a).astype(np.float64)
            kp = max(1, int(frac * n))
            for _ in range(kp):
                i = int(rng.integers(0, n)); j = int(rng.integers(0, n))
                if a[i] != a[j]:
                    a[i] *= -1.0; a[j] *= -1.0
        else:
            a = np.full(n, -1.0, dtype=np.float64)
            pos = rng.choice(n, size=n_plus, replace=False)
            a[pos] = 1.0
        a_res, f, flips, _ = psd_search(n, max_flips=inner_flips, tol=0.0,
                                          start_seq=a, stop_flag=stop_flag,
                                          callback=progress_callback)
        if best_a is None or f < best_f:
            best_a = a_res.copy(); best_f = f
        if progress_callback is not None:
            progress_callback({"iter": it, "f": f, "best_f": best_f,
                               "elapsed_s": time.monotonic() - t0})
        it += 1
    if best_a is None:  # cancelled before the first descent finished
        return None, None, False
    H = circulant_matrix(np.round(best_a).astype(np.int8))
    return H, best_f, verify(H)


def main():
    os.system("clear" if sys.platform != "win32" else "cls")
    print("Circulant Hadamard Evolver — start from H.4\n", flush=True)

    cycle = 0
    search_m = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
                32, 34, 36, 38, 40, 44, 50, 56, 62, 68, 74, 80, 86, 92, 98]
    try:
        for m in search_m:
            n = m * m
            p = OUT / f"circulant_hadamard_{n}.csv"
            if p.is_file():
                H = np.loadtxt(str(p), delimiter=",", dtype=np.int8)
                if verify(H):
                    print(f"  n={n:4d}: [known]"); continue
            if n == 4:
                a = np.array([1,1,1,-1], dtype=np.int8)
                H = circulant_matrix(a)
                np.savetxt(str(p), H.astype(np.int8), delimiter=",", fmt="%d")
                print(f"  n={n:4d}: H.4 seed"); continue

            cycle += 1
            print(f"  [{cycle:04d}] n={n:4d} (m={m:2d}) ...", end=" ", flush=True)
            t0 = time.monotonic()
            H, best_f, ok = search_ils(n, inner_flips=5000, outer_iters=10,
                                        time_budget=60, rng=np.random.default_rng())
            dt = time.monotonic() - t0
            if ok and n > 4:
                np.savetxt(str(p), H.astype(np.int8), delimiter=",", fmt="%d")
                print(f"FOUND! f={best_f:.1f} ({dt:.1f}s)", flush=True)
            else:
                print(f"f={best_f:.1f} ({dt:.1f}s)", flush=True)

    except KeyboardInterrupt:
        print(f"\nStopped after {cycle} cycles.", flush=True)


if __name__ == "__main__":
    main()
