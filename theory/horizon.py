"""Closed-form provenance horizon and the green-count lengths that bracket it."""
import math

import numpy as np
from scipy.stats import binom, norm

GRID_COLUMNS = ((1e-3, 0.01), (1e-2, 0.05), (0.05, 0.10))
GRID_REGIMES = (
    ("light", 0.02, 0.8),
    ("moderate", 0.02, 0.5),
    ("aggressive", 0.02, 0.2),
    ("high-quality", 0.01, 0.5),
)
TIGHT_ROWS = (
    ("light", 0.32, 0.61),
    ("moderate", 0.32, 0.23),
    ("aggressive", 0.32, 0.04),
    ("high-quality WM", 0.16, 0.23),
)
FRONTIER_ALPHAS = (1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2)


def horizon_constant(alpha, beta):
    """ln(1 / (1 - (1 - alpha - beta)^2)), the horizon in units of 1 / (eta * gamma)."""
    if not (0.0 < alpha + beta < 1.0):
        raise ValueError("requires 0 < alpha + beta < 1")
    c = (1.0 - alpha - beta) ** 2
    return math.log(1.0 / (1.0 - c))


def L_star(alpha, beta, gamma, eta):
    if gamma <= 0 or eta <= 0:
        raise ValueError("gamma and eta must be positive")
    return horizon_constant(alpha, beta) / (eta * gamma)


def bern_kl(p, q):
    """KL(Bern(p) || Bern(q)) in nats."""
    if not (0.0 <= p <= 1.0 and 0.0 < q < 1.0):
        raise ValueError("invalid Bernoulli parameters")
    out = 0.0
    if p > 0:
        out += p * math.log(p / q)
    if p < 1:
        out += (1 - p) * math.log((1 - p) / (1 - q))
    return out


def f_kl(delta, g=0.5):
    """KL(Bern(g + delta) || Bern(g))."""
    return bern_kl(g + delta, g)


def delta_star(gamma, g=0.5, tol=1e-14):
    """Positive bias delta with KL(Bern(g + delta) || Bern(g)) = gamma, by bisection."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    upper = 1.0 - g
    if gamma >= f_kl(upper, g):
        raise ValueError("gamma exceeds the maximal Bernoulli divergence at this g")
    lo, hi = 0.0, upper
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f_kl(mid, g) < gamma:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def L_ach(alpha, beta, gamma, eta, leading_order=False):
    """Midpoint-threshold Hoeffding length; leading_order uses delta*^2 = gamma / 2."""
    d2 = gamma / 2.0 if leading_order else delta_star(gamma) ** 2
    return 2.0 / (eta * d2) * math.log(1.0 / min(alpha, beta))


def L_op(alpha, beta, gamma, eta, leading_order=False):
    """Balanced-threshold Hoeffding length; leading_order uses delta*^2 = gamma / 2."""
    d2 = gamma / 2.0 if leading_order else delta_star(gamma) ** 2
    s = math.sqrt(math.log(1.0 / alpha)) + math.sqrt(math.log(1.0 / beta))
    return s * s / (2.0 * eta * d2)


def post_edit_rate(gamma, eta, g=0.5):
    """Green rate after the edit, g + sqrt(eta) * delta*(gamma)."""
    return g + math.sqrt(eta) * delta_star(gamma, g)


def L_clt(alpha, beta, gamma, eta, g=0.5):
    """Two-variance Gaussian threshold length for the green-count test."""
    p_e = post_edit_rate(gamma, eta, g)
    z_a, z_b = norm.ppf(1 - alpha), norm.ppf(1 - beta)
    num = z_a * math.sqrt(g * (1 - g)) + z_b * math.sqrt(p_e * (1 - p_e))
    return (num / (p_e - g)) ** 2


def L_clt_constant_single_variance(alpha, beta):
    """Single-variance Gaussian threshold length in units of 1 / (eta * gamma)."""
    z_a, z_b = norm.ppf(1 - alpha), norm.ppf(1 - beta)
    return (z_a + z_b) ** 2 / 2.0


def L_exact(alpha, beta, gamma, eta, g=0.5, L_max=100_000):
    """Smallest L at which a deterministic green-count threshold meets both error budgets."""
    p_e = post_edit_rate(gamma, eta, g)
    for L in range(1, L_max + 1):
        k = int(binom.ppf(1 - alpha, L, g)) + 1
        if binom.sf(k - 1, L, g) > alpha:
            k += 1
        if binom.cdf(k - 1, L, p_e) <= beta:
            return L
    raise RuntimeError("L_exact not found below L_max")


def model_power(L, gamma, eta, alpha, g=0.5):
    """Gaussian power of the green-count z-test at length L against the post-edit rate."""
    z = norm.ppf(1 - alpha)
    tau = g + z * math.sqrt(g * (1 - g) / L)
    p_e = post_edit_rate(gamma, eta, g)
    return float(norm.cdf((p_e - tau) / math.sqrt(p_e * (1 - p_e) / L)))


def unit_fraction_below(lengths, L_threshold):
    """Fraction of lengths strictly below the ceiling of L_threshold."""
    arr = np.asarray(lengths)
    return float(np.mean(arr < math.ceil(L_threshold)))
