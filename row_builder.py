"""Row-by-row construction using α,β,γ,δ counting + block decompression.

For order n = 4k, normalized form (row0 & col0 all +1).

Observed structure in H.668 (from block analysis):
  - First 3 interior columns form an H4 header encoding the row index mod 4
  - Remaining 4k−3 columns split into 3 blocks:
      B₁: length 2k−1   B₂: length k−1   B₃: length k−1
  - Fill rule: for row r (1-based), the block signs depend on r mod 4.
    (This generalises to: for ANY row pattern, we need the blocks to
     satisfy orthogonality between rows of different parity classes.)

For ALL rows r, the parity class (r mod 4) determines the sign pattern
across the 3 header columns + the 3 content blocks.

This implements a PARAMETERISED construction: given k, choose the
3×4 sign matrix for the header, then populate blocks via expansion.
"""
import numpy as np
from itertools import product


def _build_h4_header():
    """The four sign patterns for the 2-column H4 header (rows 0-3),
    using cols 1,2 after the all-+ col 0."""
    return np.array([
        [ 1,  1],   # class 0 (row 0,4,8,...): ++
        [ 1, -1],   # class 1 (row 1,5,9,...): +-
        [-1,  1],   # class 2 (row 2,6,10,...): -+
        [-1, -1],   # class 3 (row 3,7,11,...): --
    ], dtype=np.int8)


def _fill_block_row(k, row_len, pattern):
    """Pattern is a string like '+-+-' giving alternating signs, or
    a constant sign ('+' or '-')."""
    row = np.empty(row_len, dtype=np.int8)
    if pattern == '+':
        row[:] = 1
    elif pattern == '-':
        row[:] = -1
    else:
        for j in range(row_len):
            row[j] = 1 if pattern[j % len(pattern)] == '+' else -1
    return row


def _build_matrix(k, header, block_patterns):
    """
    header: 4×3 array, signs for the first 3 interior columns per parity class
    block_patterns: dict mapping (parity_class, block_idx) -> sign_pattern
        parity_class ∈ {0,1,2,3}  (row_index mod 4, where row 0 is all +)
        block_idx ∈ {0,1,2} → sign pattern string or constant

    Returns normalized Hadamard of order 4k if constraints are satisfied.
    """
    n = 4 * k
    b0_len = 2 * k - 1
    b1_len = k - 1
    b2_len = k - 1

    H = np.ones((n, n), dtype=np.int8)
    # row 0 and col 0 are already +1

    for r in range(1, n):
        pclass = r % 4  # 1,2,3,0,1,2,3,0,...
        # header columns 1,2,3
        H[r, 1:3] = header[pclass]

        # block 0
        start = 3
        H[r, start:start + b0_len] = _fill_block_row(
            k, b0_len, block_patterns[(pclass, 0)])

        # block 1
        start = 3 + b0_len
        H[r, start:start + b1_len] = _fill_block_row(
            k, b1_len, block_patterns[(pclass, 1)])

        # block 2
        start = 3 + b0_len + b1_len
        H[r, start:start + b2_len] = _fill_block_row(
            k, b2_len, block_patterns[(pclass, 2)])

    return H


def _check_balance(H, k):
    """Check that rows 1..n-1 are balanced (sum 0)."""
    n = 4 * k
    for r in range(1, n):
        if np.sum(H[r]) != 0:
            return False
    return True


def _check_ortho(H, k):
    n = 4 * k
    G = H.astype(np.float64) @ H.astype(np.float64).T
    return np.abs(G - n * np.eye(n)).max() < 0.5


def search_block_patterns(k, header):
    """Brute-force search over block sign patterns for a valid Hadamard."""
    from .hadamard import verify

    # block patterns to try: constant +, constant -, or alternating +- / -+
    choices = ['+', '-', '+-', '-+']
    for bp0, bp1, bp2 in product(choices[:3], repeat=3):
        patterns = {}
        for pclass in range(4):
            patterns[(pclass, 0)] = bp0
            patterns[(pclass, 1)] = bp1
            patterns[(pclass, 2)] = bp2
        H = _build_matrix(k, header, patterns)
        if not _check_balance(H, k):
            continue
        if verify(H):
            return H
    # no solution with uniform block patterns
    return None


def try_construct(order):
    """Try to build order via block decompression."""
    if order % 4 != 0 or order < 4:
        return None
    k = order // 4
    header = _build_h4_header()
    return search_block_patterns(k, header)
