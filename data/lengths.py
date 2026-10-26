"""Sentence segmentation, token counting, length statistics and the fraction of units below the horizon."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from theory.horizon import L_star, unit_fraction_below


class SentenceSegmenter:
    """spaCy sentence boundaries; long sections are processed in paragraph chunks."""

    def __init__(self, model: str = "en_core_web_sm", max_chars: int = 200_000):
        import spacy

        self.nlp = spacy.load(model, disable=["ner", "lemmatizer"])
        self.nlp.max_length = max_chars + 1
        self.max_chars = max_chars
        self.version = self.nlp.meta.get("version")

    def _chunks(self, text: str) -> Iterator[str]:
        buf, size = [], 0
        for para in text.split("\n"):
            if size + len(para) > self.max_chars and buf:
                yield "\n".join(buf)
                buf, size = [], 0
            buf.append(para)
            size += len(para) + 1
        if buf:
            yield "\n".join(buf)

    def __call__(self, text: str) -> list[str]:
        out = []
        for chunk in self._chunks(text):
            for s in self.nlp(chunk).sents:
                t = " ".join(s.text.split())
                if t:
                    out.append(t)
        return out


class TokenCounter:
    """Token counts with a Hugging Face tokenizer."""

    def __init__(self, name: str = "meta-llama/Meta-Llama-3-8B", revision: str | None = None):
        from transformers import AutoTokenizer

        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision)

    def __call__(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))


@dataclass
class UnitRecord:
    filing_id: str
    section: str
    unit: str
    index: int
    text: str
    n_tokens: int


def write_units(records: Iterable[UnitRecord], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.__dict__) + "\n")
            n += 1
    return n


def read_units(path: Path, unit: str | None = None) -> Iterator[UnitRecord]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = UnitRecord(**json.loads(line))
                if unit is None or r.unit == unit:
                    yield r


def build_units(sections, segmenter: SentenceSegmenter, counter: TokenCounter,
                min_sentence_tokens: int = 1) -> Iterator[UnitRecord]:
    """Yields the excerpt unit and the sentence units of every section record."""
    for rec in sections:
        n_exc = counter(rec.text)
        yield UnitRecord(rec.filing_id, rec.section, "excerpt", 0, rec.text, n_exc)
        for i, s in enumerate(segmenter(rec.text)):
            n = counter(s)
            if n >= min_sentence_tokens:
                yield UnitRecord(rec.filing_id, rec.section, "sentence", i, s, n)


def length_stats(lengths: Sequence[int]) -> dict:
    a = np.asarray(lengths, dtype=float)
    if a.size == 0:
        return {"count": 0}
    return {
        "count": int(a.size),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
    }


UNIT_TABLE_ETAS = (("light", 0.8), ("moderate", 0.5), ("aggressive", 0.2))


def unit_table(sentence_lengths: Sequence[int], excerpt_lengths: Sequence[int],
               alpha: float = 1e-2, beta: float = 0.05, gamma: float = 0.02,
               etas=UNIT_TABLE_ETAS) -> dict:
    """Percent of units strictly shorter than the ceiling-rounded horizon per edit strength, with medians."""
    rows = {}
    thresholds = {name: math.ceil(L_star(alpha, beta, gamma, eta)) for name, eta in etas}
    for unit, arr in (("sentence", sentence_lengths), ("filing excerpt", excerpt_lengths)):
        a = np.asarray(arr)
        rows[unit] = {
            "median_tokens": float(np.median(a)) if a.size else float("nan"),
            **{name: 100.0 * unit_fraction_below(a, thresholds[name]) for name, _ in etas},
        }
    return {"L_star": thresholds, "rows": rows}
