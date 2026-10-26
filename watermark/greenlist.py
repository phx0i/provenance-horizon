"""Green-list family watermark and the green-count z-test."""
import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class GreenListWatermark:
    """Keyed green/red partition; context_width 1 seeds it by the previous token, 0 by the key alone."""

    vocab_size: int
    key: int
    green_fraction: float = 0.5
    bias: float = 2.0
    context_width: int = 1

    def __post_init__(self):
        if self.context_width not in (0, 1):
            raise ValueError("context_width must be 0 or 1")
        self._n_green = int(math.floor(self.green_fraction * self.vocab_size))
        self._cache = {}

    @property
    def name(self):
        return "unigram" if self.context_width == 0 else "greenlist"

    def green_mask(self, prev_token):
        ctx = None if self.context_width == 0 else prev_token
        m = self._cache.get(ctx)
        if m is None:
            seed = [self.key] if ctx is None else [self.key, int(ctx)]
            rng = np.random.default_rng(seed)
            perm = rng.permutation(self.vocab_size)
            m = np.zeros(self.vocab_size, dtype=bool)
            m[perm[: self._n_green]] = True
            if self.context_width == 0 or len(self._cache) < 4096:
                self._cache[ctx] = m
        return m

    def is_green(self, prev_token, token):
        return bool(self.green_mask(prev_token)[token])

    def process_logits(self, logits, prev_token):
        if self.context_width == 1 and prev_token is None:
            return logits
        out = logits.copy()
        out[self.green_mask(prev_token)] += self.bias
        return out

    def scored_positions(self, ids):
        """(context, token) pairs scored by the detector; with context width 1 the first token is skipped."""
        ids = list(ids)
        if self.context_width == 0:
            return [(None, t) for t in ids]
        return [(ids[i - 1], ids[i]) for i in range(1, len(ids))]

    def green_count(self, ids):
        pairs = self.scored_positions(ids)
        S = sum(1 for ctx, tok in pairs if self.is_green(ctx, tok))
        return S, len(pairs)

    def green_fraction_of(self, ids):
        S, T = self.green_count(ids)
        return S / T if T > 0 else float("nan")


def z_score(S, T, g):
    if T <= 0:
        return float("-inf")
    return (S - g * T) / math.sqrt(T * g * (1.0 - g))


def z_threshold(alpha):
    return float(norm.ppf(1.0 - alpha))


class ZTestDetector:
    """Flags a text when the green-count z-score exceeds the one-sided nominal threshold."""

    def __init__(self, wm, alpha=0.01):
        self.wm = wm
        self.alpha = alpha
        self.threshold = z_threshold(alpha)

    def statistic(self, ids):
        S, T = self.wm.green_count(ids)
        return z_score(S, T, self.wm.green_fraction)

    def detect(self, ids):
        return self.statistic(ids) > self.threshold


def make_watermark(cfg, vocab_size):
    scheme = cfg["scheme"]
    ctx = {"greenlist": 1, "unigram": 0}[scheme]
    return GreenListWatermark(vocab_size=vocab_size, key=int(cfg["key"]),
                              green_fraction=float(cfg.get("green_fraction", 0.5)),
                              bias=float(cfg.get("bias", 2.0)), context_width=ctx)
