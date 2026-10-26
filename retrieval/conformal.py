"""Split-conformal calibration of the retrieval score, group-conditional variant, and log commitment."""
import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np


def threshold_index(n, alpha):
    """Order-statistic index ceil((1 - alpha)(n + 1)), one-based."""
    return int(math.ceil((1.0 - alpha) * (n + 1) - 1e-12))


def min_calibration_size(alpha):
    return math.ceil(1.0 / alpha) - 1


def conformal_threshold(cal_scores, alpha):
    s = np.sort(np.asarray(cal_scores, dtype=float))
    n = len(s)
    if n < min_calibration_size(alpha):
        raise ValueError(f"need at least {min_calibration_size(alpha)} calibration scores, got {n}")
    k = threshold_index(n, alpha)
    if k > n:
        raise ValueError("threshold index exceeds n")
    return float(s[k - 1])


def jitter(scores, rng, scale=1e-9):
    """Randomised tie-breaking by uniform noise."""
    return np.asarray(scores, dtype=float) + rng.uniform(0.0, scale, size=np.shape(scores))


@dataclass
class ConformalDetector:
    """Flags a score above the split-conformal threshold of the calibration scores."""

    alpha: float
    cal_scores: np.ndarray
    tie_break_seed: object = 0

    def __post_init__(self):
        self.cal_scores = np.asarray(self.cal_scores, dtype=float)
        self._rng = np.random.default_rng(self.tie_break_seed) if self.tie_break_seed is not None else None
        cal = jitter(self.cal_scores, self._rng) if self._rng is not None else self.cal_scores
        self.threshold = conformal_threshold(cal, self.alpha)
        self.n = len(self.cal_scores)
        self.k = threshold_index(self.n, self.alpha)

    def detect(self, scores):
        s = np.asarray(scores, dtype=float)
        if self._rng is not None:
            s = jitter(s, self._rng)
        return s > self.threshold


def _stable_int(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


@dataclass
class MondrianConformalDetector:
    """One split-conformal threshold per group."""

    alpha: float
    cal_scores: object
    cal_groups: object
    tie_break_seed: object = 0

    def __post_init__(self):
        scores = np.asarray(self.cal_scores, dtype=float)
        groups = list(self.cal_groups)
        self.detectors = {}
        for g in sorted(set(groups), key=str):
            idx = [i for i, gg in enumerate(groups) if gg == g]
            seed = None if self.tie_break_seed is None else self.tie_break_seed + _stable_int(str(g)) % 10_000
            self.detectors[g] = ConformalDetector(self.alpha, scores[idx], tie_break_seed=seed)

    def detect(self, scores, groups):
        out = np.zeros(len(scores), dtype=bool)
        for g, det in self.detectors.items():
            idx = [i for i, gg in enumerate(groups) if gg == g]
            if idx:
                out[idx] = det.detect(np.asarray(scores, dtype=float)[idx])
        return out

    @property
    def thresholds(self):
        return {g: d.threshold for g, d in self.detectors.items()}


@dataclass(frozen=True)
class BetaSpread:
    a: int
    b: int
    mean: float
    sd: float


def beta_spread(n, alpha):
    """Beta(n + 1 - k, k) law of the conditional false-positive rate for a calibration set of size n."""
    k = threshold_index(n, alpha)
    a, b = n + 1 - k, k
    mean = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1))
    return BetaSpread(a=a, b=b, mean=mean, sd=math.sqrt(var))


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def merkle_root(leaves):
    level = [_sha256(("leaf:" + s).encode("utf-8")) for s in leaves]
    if not level:
        return _sha256(b"")
    n = len(level)
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        level = [_sha256(("node:" + level[i] + level[i + 1]).encode()) for i in range(0, len(level), 2)]
    return _sha256(f"root:{n}:{level[0]}".encode())


@dataclass(frozen=True)
class Commitment:
    log_root: str
    sim_digest: str
    n_entries: int


def commit(log_entries, sim_config):
    """Merkle root of the log entries and a digest of the similarity configuration."""
    digest = _sha256(json.dumps(sim_config, sort_keys=True, default=str).encode())
    return Commitment(log_root=merkle_root(log_entries), sim_digest=digest, n_entries=len(log_entries))
