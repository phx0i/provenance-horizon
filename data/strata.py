"""Boilerplate stratum of third-party text, for the false-positive rate on repetitive boilerplate."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from retrieval.index import ngrams, words


@dataclass
class BoilerplateDetector:
    n: int = 5
    threshold: float = 0.3
    min_filings: int = 2

    def fit(self, texts_with_filing: Iterable[tuple[str, str]]) -> "BoilerplateDetector":
        """Counts, for every word n-gram, the number of distinct filings containing it."""
        df: dict[tuple, set] = defaultdict(set)
        for filing_id, text in texts_with_filing:
            for g in set(ngrams(words(text), self.n)):
                df[g].add(filing_id)
        self.df = {g: len(s) for g, s in df.items()}
        return self

    def duplication(self, text: str, own_filing: str | None = None) -> float:
        grams = set(ngrams(words(text), self.n))
        if not grams:
            return 0.0
        own = 1 if own_filing is not None else 0
        hits = sum(1 for g in grams if self.df.get(g, 0) - own >= self.min_filings)
        return hits / len(grams)

    def is_boilerplate(self, text: str, own_filing: str | None = None) -> bool:
        return self.duplication(text, own_filing) >= self.threshold

    def groups(self, items: Sequence[dict]) -> list[str]:
        return ["boilerplate" if self.is_boilerplate(it["text"], it.get("filing_id")) else "standard" for it in items]
