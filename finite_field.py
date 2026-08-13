"""Finite-field arithmetic for Paley prime-power constructions.

Field elements are represented as coefficient lists (a_0, a_1, ..., a_{e-1})
mod p, with multiplication modulo an irreducible polynomial of degree e.

Exports:
    legendre_symbol(a, p, e, irr_poly) -> -1 | 0 | 1
    find_irr_poly(p, e)       -> tuple[int,...]  (coefficients of x^e)
"""

from __future__ import annotations


def _pmul(a, b, p, irr):
    """Multiply two polynomials modulo irreducible irr over GF(p)."""
    deg = len(irr) - 1
    need = len(a) + len(b) - 1
    res = [0] * need
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b):
            if bj == 0:
                continue
            res[i + j] = (res[i + j] + ai * bj) % p
    while len(res) > deg:
        if res[-1] == 0:
            res.pop()
            continue
        lead = res[-1]
        shift = len(res) - deg - 1
        for k, ck in enumerate(irr[:-1]):
            if ck:
                res[shift + k] = (res[shift + k] - lead * ck) % p
        res.pop()
    while len(res) < deg:
        res.append(0)
    return res


def _field_pow(a, exp, p, irr):
    """Exponentiation by squaring in the polynomial representation."""
    e = len(irr) - 1
    result = [1] + [0] * (e - 1)
    base = list(a)
    while exp > 0:
        if exp & 1:
            result = _pmul(result, base, p, irr)
        exp >>= 1
        if exp:
            base = _pmul(base, base, p, irr)
    return result


def _poly_from_int(x, p, e):
    """Map integer 0..p^e-1 to coefficient list."""
    coeffs = []
    for _ in range(e):
        coeffs.append(x % p)
        x //= p
    return coeffs


def _poly_to_int(coeffs, p):
    """Map coefficient list back to integer."""
    v = 0
    mul = 1
    for c in coeffs:
        v += c * mul
        mul *= p
    return v


def legendre_symbol(a, p, e, irr=None):
    """Legendre symbol χ(a) in F_{p^e}. Returns -1, 0, or 1."""
    if irr is None:
        irr = _irr_for(p, e)
    if a == 0:
        return 0
    q = p ** e
    exp = (q - 1) // 2
    poly = _poly_from_int(a, p, e)
    res = _field_pow(poly, exp, p, irr)
    # result should be (1) or (p-1) as constant polynomial
    r = _poly_to_int(res, p)
    if r == 1:
        return 1
    if r == p - 1:
        return -1
    return 0


_IRR_CACHE = {}


def _irr_for(p, e):
    if (p, e) in _IRR_CACHE:
        return _IRR_CACHE[(p, e)]
    irr = find_irr_poly(p, e)
    _IRR_CACHE[(p, e)] = irr
    return irr


_IRR_TABLE = {
    (3, 2):  (1, 0, 1),                 # x^2 + 1 over F_3        (needed for q=9)
    (3, 3):  (1, 2, 0, 1),             # x^3 + 2x + 1 over F_3   (order 28)
    (3, 4):  (2, 1, 0, 0, 1),          # x^4 + x + 2 over F_3     (order 324)
    (3, 5):  (1, 2, 0, 0, 0, 1),       # x^5 + 2x + 1 over F_3   (order 244)
    (3, 6):  (2, 2, 1, 0, 2, 0, 1),     # x^6 + 2x^4 + x^2 + 2x + 2 over F_3 (order 2916)
    (5, 2):  (2, 0, 1),                 # x^2 + 2 over F_5        (order 52)
    (5, 4):  (2, 0, 0, 0, 1),          # x^4 + 2 over F_5        (order 1252)
    (7, 2):  (4, 0, 1),                 # x^2 + 4 over F_7        (order 100)
    (7, 3):  (2, 0, 0, 1),             # x^3 + 2 over F_7        (order 344)
    (11, 3): (2, 0, 2, 1),             # x^3 + 2x^2 + 2 over F_11 (order 1332)
    (13, 2): (2, 0, 1),                 # x^2 + 2 over F_13       (order 340)
    (17, 2): (3, 0, 1),                 # x^2 + 3 over F_17       (order 580)
    (19, 2): (1, 0, 1),                 # x^2 + 1 over F_19       (order 724)
    (23, 2): (1, 0, 1),                 # x^2 + 1 over F_23       (order 1060)
}


def find_irr_poly(p, e):
    if e == 1:
        return (0,)
    key = (p, e)
    if key in _IRR_TABLE:
        return _IRR_TABLE[key]
    raise RuntimeError(f"no precomputed irreducible poly for F_{p}^{e}")


def _poly_gcd(a, b, p):
    a, b = list(a), list(b)
    while any(b):
        rem = _poly_rem(a, b, p)
        a, b = b, rem
    return a


def _poly_rem(a, b, p):
    a = list(a)
    b = list(b)
    while len(a) >= len(b) and any(a):
        factor = a[-1] * _modinv(b[-1], p) % p
        shift = len(a) - len(b)
        for i, bi in enumerate(b):
            if bi:
                a[shift + i] = (a[shift + i] - factor * bi) % p
        while a and a[-1] == 0:
            a.pop()
    return a if a else [0]


def _modinv(a, p):
    return pow(a, -1, p)
