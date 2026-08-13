"""Miyamoto construction: H(4q) from H(q-1) when q ≡ 1 mod 4 is a prime power.

Translated from SageMath's hadamard_matrix_miyamoto_construction.
Reference: Miyamoto, J. Combin. Theory A 57.1 (1991), pp. 86-108.
"""

import numpy as np


def _conference_matrix(q):
    """Build symmetric Paley conference matrix of order q+1 over F_q.
    C[i,i]=0, C[i,j]=χ(j-i), C[0,j]=C[j,0]=1 for j≥1."""
    from hoa64.finite_field import legendre_symbol
    from hoa64.hadamard import _is_prime_power
    pp = _is_prime_power(q)
    if pp is None:
        raise ValueError(f"{q} is not a prime power")
    p, e = pp
    n = q + 1
    C = np.zeros((n, n), dtype=np.int64)
    if e > 1:
        # use finite field Legendre
        from hoa64.hadamard import _field_ops, _legendre_table_finite_field
        chi = _legendre_table_finite_field(q, p, e)
        _, sub = _field_ops(q, p, e)
        r = np.arange(q)
        C[1:, 1:] = chi[sub[r[:, None], r[None, :]]]
    else:
        # prime q — use integer Legendre
        chi = np.full(q, -1, dtype=np.int64)
        chi[0] = 0
        for x in range(1, q):
            chi[(x * x) % q] = 1
        r = np.arange(q)
        D = (r[None, :] - r[:, None]) % q
        C[1:, 1:] = chi[D]
    C[1:, 0] = 1
    C[0, 1:] = 1
    return C


def miyamoto_construction(q, H_qm1):
    """Build H(4q) from H(q-1). q ≡ 1 mod 4, prime power.

    Returns Hadamard matrix of order 4q (numpy int8, normalized).
    """
    m = (q - 1) // 2
    C = _conference_matrix(q)              # order q+1

    # rearrange conference matrix rows/columns
    neg = [i for i in range(2, m + 2) if C[1, i] == -1]
    pos = [i for i in range(m + 2, 2 * m + 2) if C[1, i] == 1]
    for i, j in zip(neg, pos):
        C[[i, j]] = C[[j, i]]              # swap rows
        C[:, [i, j]] = C[:, [j, i]]        # swap columns

    C1 = -C[2:2+m, 2:2+m].copy()
    C2 =  C[2:2+m, 2+m:2+2*m].copy()
    C4 =  C[2+m:2+2*m, 2+m:2+2*m].copy()

    K = np.asarray(H_qm1, dtype=np.int64)
    h = (q - 1) // 2
    K1 = K[0:h, 0:h].copy()
    K2 = K[0:h, h:2*h].copy()
    K3 = -K[h:2*h, 0:h].copy()
    K4 = K[h:2*h, h:2*h].copy()

    Zr = np.zeros((m, m), dtype=np.int64)
    I = np.eye(m, dtype=np.int64)

    Us = [[ C1,  C2,  Zr,  Zr],
          [C2.T, C4,  Zr,  Zr],
          [ Zr,  Zr,  C1,  C2],
          [ Zr,  Zr, C2.T, C4]]

    Vs = [[   I,  Zr,  K1,  K2],
          [  Zr,   I,  K3,  K4],
          [K1.T,K3.T,   I,  Zr],
          [K2.T,K4.T,  Zr,   I]]

    Tij = {}
    for i in range(4):
        for j in range(4):
            U = Us[i][j]
            V = Vs[i][j]
            T00 = np.asarray(U + V, dtype=np.int64)
            T01 = np.asarray(U - V, dtype=np.int64)
            top = np.concatenate([T00, T01], axis=1)
            bot = np.concatenate([T01, T00], axis=1)
            Tij[(i, j)] = np.concatenate([top, bot], axis=0)

    e = np.ones((1, 2 * m), dtype=np.int64)
    eT = e.T
    one = np.ones((1, 1), dtype=np.int64)

    def hcat(*arrs):
        return np.concatenate(arrs, axis=1)
    def vcat(*arrs):
        return np.concatenate(arrs, axis=0)

    row0 = hcat( one,      -e,   one,      e,   one,      e,   one,      e)
    row1 = hcat(  -eT, Tij[(0,0)],  eT, Tij[(0,1)],  eT, Tij[(0,2)],  eT, Tij[(0,3)])
    row2 = hcat( -one,     -e,   one,     -e,   one,      e,  -one,     -e)
    row3 = hcat(  -eT,-Tij[(1,0)], -eT, Tij[(1,1)],  eT, Tij[(1,2)], -eT,-Tij[(1,3)])
    row4 = hcat( -one,     -e,  -one,     -e,   one,     -e,   one,      e)
    row5 = hcat(  -eT,-Tij[(2,0)], -eT,-Tij[(2,1)], -eT, Tij[(2,2)],  eT, Tij[(2,3)])
    row6 = hcat( -one,     -e,   one,      e,  -one,     -e,   one,     -e)
    row7 = hcat(  -eT,-Tij[(3,0)],  eT, Tij[(3,1)], -eT,-Tij[(3,2)], -eT, Tij[(3,3)])

    H = vcat(row0, row1, row2, row3, row4, row5, row6, row7).astype(np.int8)
    return H


def miyamoto_from_cache(order):
    """Try miyamoto construction using cached H(order/4 - 1)."""
    if order % 4 != 0:
        return None
    q = order // 4
    if q % 4 != 1:
        return None
    from hoa64.hadamard import _is_prime_power, hadamard_known, normalize, verify
    if _is_prime_power(q) is None:
        return None
    H = hadamard_known(q - 1)
    if H is None:
        return None
    result = miyamoto_construction(q, H)
    result = normalize(result)
    if verify(result):
        return result
    return None
