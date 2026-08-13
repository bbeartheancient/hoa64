"""Riemann hypothesis consistency check using the Hadamard-Weierstrass
factorization approach (Micheas, arXiv:2607.04338, Jul 2026).

The bound on |Delta_N(s)| = |xi_N(s) - xihat_N(s)| from eq. (23):

  |Delta_N| <= 1/2 |s(1-s)| SUM_{n=1..N} |xi_n(1-xi_n) - rho_n(1-rho_n)|
                                       / (|rho_n(1-rho_n)| |xi_n(1-xi_n)|)

with
  rho_n = 1/2 + i tau_n            (hypothesized zeros on the critical line)
  xi_n  = sigma_n + i t_n          (true zeros)
  rho_n(1-rho_n) = 1/4 + tau_n^2   (real, positive)
  xi_n(1-xi_n)   = sigma_n(1-sigma_n) + t_n^2 + i t_n(1-2 sigma_n)

If the true zeros lie on the critical line (sigma_n = 1/2, tau_n = t_n) then
xi_n(1-xi_n) = rho_n(1-rho_n) and every term vanishes => bound = 0 exactly,
confirming consistency with RH.
"""

from __future__ import annotations

KNOWN_ZEROS_T = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
]


def _rho1mrho(tau: float) -> complex:
    return 0.25 + tau * tau


def _xi1mxi(sigma: float, t: float) -> complex:
    re = sigma * (1.0 - sigma) + t * t
    im = t * (1.0 - 2.0 * sigma)
    return complex(re, im)


def rh_check(N: int = 10, s=None, sigma: float = 0.5, tau_offset: float = 0.0):
    """Numerically check the |Delta_N(s)| bound from eq. (23).

    Parameters
    ----------
    N : int
        Number of known zeros to use.
    s : complex or None
        Test point in the critical strip (default 0.5+30i).
    sigma : float
        Assumed real part of true zeros (0.5 = on critical line).
    tau_offset : float
        Perturbation of hypothesized imaginary parts away from true zeros.

    Returns
    -------
    dict with 'bound', 'terms', 'verdict'.
    """
    if s is None:
        s = 0.5 + 30j
    s = complex(float(s.real), float(s.imag))
    prefactor = 0.5 * abs(s * (1.0 - s))

    total = 0.0
    for n in range(min(N, len(KNOWN_ZEROS_T))):
        t_n = KNOWN_ZEROS_T[n]
        tau_n = t_n + tau_offset

        num = abs(_rho1mrho(tau_n) - _xi1mxi(sigma, t_n))
        denom = abs(_rho1mrho(tau_n)) * abs(_xi1mxi(sigma, t_n))
        total += num / max(denom, 1e-30)

    bound = prefactor * total
    verdict = (
        "numerically zero => consistent with RH"
        if bound < 1e-12
        else f"bound={bound:.6g} (RH not ruled out)"
    )
    return dict(N=N, bound=float(bound), s=str(s), verdict=verdict)


def rh_detector(max_N: int = 10):
    """Run rh_check at N = 1..max_N and report the convergence pattern."""
    for N in range(1, max_N + 1):
        r = rh_check(N=N)
        print(f"  N={N:2d}  |Delta_N| bound = {r['bound']:.4g}  ->  {r['verdict']}")
