"""GCP-based Hadamard construction from Priyanshu/Majhi/Paul (arXiv:2510.12315).

Theorem 3: let (a,b) be a GCP of length N, (c,d) its complementary mate.
Build circulants A=Cir(a), B=Cir(b), C=Cir(c), D=Cir(d). Then
  G = [[A, B], [C, D]]
is a Hadamard matrix of order 2N.

GCPs are generated via GBFs (Lemma 1) for powers of two and combined via
Kronecker (Lemma 2).  Complementary mates via Lemma 3.
"""

from __future__ import annotations

import numpy as np


def seq_to_circulant(s: np.ndarray) -> np.ndarray:
    N = len(s)
    M = np.empty((N, N), dtype=np.int64)
    for i in range(N):
        M[i] = np.roll(s, i)
    return M


def aacf(s, lag):
    n = len(s)
    if lag < 0 or lag >= n:
        return 0
    return int(np.dot(s[:n - lag], s[lag:]))


def is_gcp(a, b):
    n = len(a)
    for lag in range(1, n):
        if aacf(a, lag) + aacf(b, lag) != 0:
            return False
    return True


def complementary_mate(a, b):
    a, b = np.asarray(a), np.asarray(b)
    c = np.flip(b)       # b*←  (reverse conjugate; binary so conjugate = identity)
    d = -np.flip(a)      # −a*←
    return c, d


def gcp_from_gbf(m, perm=None, coeffs=None, theta=0, theta_prime=0):
    """GCP of length 2^m via GBF (Lemma 1, h=1 for binary)."""
    L = 1 << m
    if perm is None:
        perm = list(range(m))
    if coeffs is None:
        coeffs = [0] * m

    a = np.zeros(L, dtype=np.int64)
    b = np.zeros(L, dtype=np.int64)
    for idx in range(L):
        x = [((idx >> i) & 1) for i in range(m)]
        f = 0
        for i in range(m - 1):
            f += x[perm[i]] * x[perm[i + 1]]
        for k in range(m):
            f += coeffs[k] * x[k]
        f = f % 2
        a[idx] = 1 if (f + theta) % 2 == 0 else -1
        b[idx] = 1 if (f + x[perm[0]] + theta_prime) % 2 == 0 else -1
    return a, b


def gcp_of_length_n(n, rng=None):
    """Return a GCP of length n (supports powers of 2, 10, 26, 2x10, 2x26)."""
    if n == 1:
        a = np.array([1], dtype=np.int64)
        b = np.array([1], dtype=np.int64)
        return a, b
    if n == 2:
        a = np.array([1, 1], dtype=np.int64)
        b = np.array([1, -1], dtype=np.int64)
        return a, b
    if (n & (n - 1)) == 0 and n >= 4:
        m = n.bit_length() - 1
        a, b = gcp_from_gbf(m)
        return a, b
    if n == 5:
        a = np.array([1, 1, -1, 1, -1], dtype=np.int64)
        b = np.array([-1, -1, -1, 1, 1], dtype=np.int64)
        if is_gcp(a, b):
            return a, b
    if n == 10:
        a = np.array([1, 1, -1, 1, -1, 1, 1, 1, 1, -1], dtype=np.int64)
        b = np.array([1, 1, 1, -1, 1, 1, 1, -1, -1, 1], dtype=np.int64)
        if is_gcp(a, b):
            return a, b
    if n == 13:
        a = np.array([1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1], dtype=np.int64)
        b = np.array([1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1], dtype=np.int64)
        if is_gcp(a, b):
            return a, b
    if n == 26:
        a = np.array([1,1,1,1,1,-1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,1,1], dtype=np.int64)
        b = np.array([1,1,1,-1,1,1,1,1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,-1,-1,-1], dtype=np.int64)
        if is_gcp(a, b):
            return a, b
    for d in range(2, n // 2 + 1):
        if n % d == 0:
            a1, b1 = gcp_of_length_n(d, rng)
            a2, b2 = gcp_of_length_n(n // d, rng)
            if a1 is not None and a2 is not None:
                a, b = kron_gcp(a1, b1, a2, b2)
                if is_gcp(a, b):
                    return a, b
    return None, None


def kron_gcp(a, b, c, d):
    """Lemma 2: Kronecker product of two GCPs → combined GCP of len |a|*|c|."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    e = np.kron(a, (c + d) / 2.0) - np.kron(np.conj(b[::-1]), (c - d) / 2.0)
    f = np.kron(b, (c + d) / 2.0) + np.kron(np.conj(a[::-1]), (c - d) / 2.0)
    return np.round(e).astype(np.int64), np.round(f).astype(np.int64)


def build_hadamard_from_gcp(a, b):
    """Theorem 3 + Corollary 2: build Hadamard from GCP + mate."""
    N = len(a)
    c, d = complementary_mate(a, b)
    A = seq_to_circulant(a)
    B = seq_to_circulant(b)
    C = seq_to_circulant(c)
    D = seq_to_circulant(d)
    top = np.concatenate([A, B], axis=1)
    bot = np.concatenate([C, D], axis=1)
    return np.concatenate([top, bot], axis=0).astype(np.int8)


def construct(order, rng=None):
    """Try to construct a Hadamard of given order via GCP method."""
    if order % 2 != 0 or order < 2:
        return None
    N = order // 2
    a, b = gcp_of_length_n(N, rng)
    if a is None:
        return None
    H = build_hadamard_from_gcp(a, b)
    from .hadamard import verify
    if verify(H):
        return H
    return None


def constructible_orders(max_n):
    """Return list of orders constructible by GCP method up to max_n."""
    from .hadamard import verify
    out = []
    for N in range(1, max_n // 2 + 1):
        a, b = gcp_of_length_n(N)
        if a is not None:
            H = build_hadamard_from_gcp(a, b)
            if verify(H):
                out.append(2 * N)
    return sorted(set(out))
