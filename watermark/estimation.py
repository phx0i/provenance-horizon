"""Estimators of the per-token distortion gamma and the edit contraction eta."""
import math
from dataclasses import dataclass

import numpy as np


def gamma_hat(per_step_kls):
    """Mean per-token KL over all generated tokens."""
    flat = np.concatenate([np.asarray(k, dtype=float) for k in per_step_kls if len(k)])
    return float(flat.mean())


def gaussian_kl(m1, v1, m0, v0):
    return math.log(math.sqrt(v0) / math.sqrt(v1)) + (v1 + (m1 - m0) ** 2) / (2.0 * v0) - 0.5


def statistic_kl(stat_h1, stat_h0):
    """KL between Gaussian fits to the detection statistic under H1 and H0."""
    a = np.asarray(stat_h1, dtype=float)
    b = np.asarray(stat_h0, dtype=float)
    v1, v0 = max(a.var(ddof=1), 1e-12), max(b.var(ddof=1), 1e-12)
    return gaussian_kl(float(a.mean()), v1, float(b.mean()), v0)


@dataclass(frozen=True)
class EtaEstimate:
    eta: float
    kl_before: float
    kl_after: float


def eta_hat(stat_h1_before, stat_h0_before, stat_h1_after, stat_h0_after):
    """Ratio of the detection-statistic KL after the edit to the KL before it, capped at one."""
    before = statistic_kl(stat_h1_before, stat_h0_before)
    after = statistic_kl(stat_h1_after, stat_h0_after)
    ratio = after / before if before > 0 else float("nan")
    return EtaEstimate(eta=float(min(1.0, ratio)), kl_before=before, kl_after=after)
