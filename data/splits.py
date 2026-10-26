"""Filing-level splits, third-party windows of a fixed token length, and host prompts."""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from data.lengths import UnitRecord, read_units

HOST_PROMPT = ("Summarize the following SEC filing excerpt in 2-3 sentences, focusing on key risk "
               "factors and financial metrics.")


@dataclass
class Prompt:
    filing_id: str
    section: str
    text: str
    excerpt_truncated: bool = False


@dataclass
class SplitPlan:
    prompt_filings: list[str]
    null_filings: list[str]
    calibration_filings: list[str]
    test_null_filings: list[str]
    null_only_filings: list[str] = field(default_factory=list)


def plan_splits(filing_ids: Sequence[str], split_seed: int, prompt_fraction: float = 0.5,
                calibration_fraction: float = 0.5, null_only_ids: Sequence[str] = ()) -> SplitPlan:
    """Shuffles filings into prompt and null pools; null-only filings join the null pool, which splits into calibration and test filings."""
    null_only = sorted(set(null_only_ids) & set(filing_ids))
    ids = sorted(set(filing_ids) - set(null_only))
    rng = np.random.default_rng(split_seed)
    rng.shuffle(ids)
    n_prompt = max(1, int(round(prompt_fraction * len(ids))))
    prompt, null = ids[:n_prompt], ids[n_prompt:]
    if null_only:
        null = null + null_only
        rng.shuffle(null)
    if not null:
        raise ValueError("need at least two filings to form prompt and null pools")
    n_cal = max(1, int(round(calibration_fraction * len(null))))
    cal, test = null[:n_cal], null[n_cal:]
    if not test:
        test, cal = cal[-1:], cal[:-1] or cal[-1:]
    return SplitPlan(prompt_filings=prompt, null_filings=null, calibration_filings=cal, test_null_filings=test,
                     null_only_filings=null_only)


class Corpus:
    """Units grouped by filing and section; filings whose sections are all null-only never serve as prompts."""

    def __init__(self, units: Sequence[UnitRecord], null_only_sections: Sequence[str] = ()):
        self.sentences: dict[tuple[str, str], list[UnitRecord]] = defaultdict(list)
        self.excerpts: dict[tuple[str, str], UnitRecord] = {}
        for u in units:
            if u.unit == "sentence":
                self.sentences[(u.filing_id, u.section)].append(u)
            elif u.unit == "excerpt":
                self.excerpts[(u.filing_id, u.section)] = u
        for v in self.sentences.values():
            v.sort(key=lambda r: r.index)
        keys = set(self.excerpts) | set(self.sentences)
        self.filing_ids = sorted({k[0] for k in keys})
        null_only = set(null_only_sections)
        sections_by_filing: dict[str, set] = defaultdict(set)
        for f, sec in keys:
            sections_by_filing[f].add(sec)
        self.null_only_ids = sorted(f for f, secs in sections_by_filing.items() if secs and secs <= null_only)

    @staticmethod
    def load(path: Path, null_only_sections: Sequence[str] = ()) -> "Corpus":
        return Corpus(list(read_units(path)), null_only_sections=null_only_sections)

    def sections_of(self, filing_ids: Sequence[str]) -> list[tuple[str, str]]:
        wanted = set(filing_ids)
        return sorted(k for k in self.sentences if k[0] in wanted)


def make_window(sentences: Sequence[str], start: int, L: int, encode: Callable[[str], list[int]],
                decode: Callable[[Sequence[int]], str]) -> Optional[str]:
    """Consecutive sentences from start until at least L tokens, truncated to exactly L tokens."""
    ids: list[int] = []
    i = start
    while len(ids) < L and i < len(sentences):
        piece = sentences[i] if not ids else " " + sentences[i]
        ids.extend(encode(piece))
        i += 1
    if len(ids) < L:
        return None
    return decode(ids[:L])


def draw_windows(corpus: Corpus, filing_ids: Sequence[str], n: int, L: int, rng: np.random.Generator,
                 encode, decode, max_tries: int = 50) -> list[dict]:
    """n windows of L tokens from the sections of filing_ids, with provenance metadata."""
    keys = corpus.sections_of(filing_ids)
    if not keys:
        raise ValueError("no sections available for the requested filings")
    out: list[dict] = []
    tries = 0
    while len(out) < n and tries < max_tries * n:
        tries += 1
        k = keys[int(rng.integers(len(keys)))]
        sents = [u.text for u in corpus.sentences[k]]
        if not sents:
            continue
        start = int(rng.integers(len(sents)))
        w = make_window(sents, start, L, encode, decode)
        if w is not None:
            out.append({"text": w, "filing_id": k[0], "section": k[1], "start": start})
    if len(out) < n:
        raise ValueError(f"could only draw {len(out)} of {n} windows of length {L}; corpus too small")
    return out


def make_prompt(excerpt: str, encode, decode, max_excerpt_tokens: int, instruction: str = HOST_PROMPT) -> str:
    ids = encode(excerpt)[:max_excerpt_tokens]
    return f"{instruction}\n\n{decode(ids)}\n\nSummary:"


def draw_prompts(corpus: Corpus, filing_ids: Sequence[str], n: int, rng: np.random.Generator,
                 encode, decode, max_excerpt_tokens: int, min_excerpt_tokens: int = 50) -> list[Prompt]:
    """n prompts built from excerpt units; when the pool is smaller than n, later passes reuse them, with a random window for excerpts longer than max_excerpt_tokens."""
    keys = [k for k in sorted(corpus.excerpts) if k[0] in set(filing_ids)]
    keys = [k for k in keys if corpus.excerpts[k].n_tokens >= min_excerpt_tokens]
    if not keys:
        raise ValueError("no excerpts available for prompts")
    order = rng.permutation(len(keys))
    out: list[Prompt] = []
    i = 0
    while len(out) < n:
        k = keys[int(order[i % len(keys)])]
        exc = corpus.excerpts[k]
        ids = encode(exc.text)
        if len(ids) > max_excerpt_tokens and i >= len(keys):
            start = int(rng.integers(0, len(ids) - max_excerpt_tokens + 1))
            text = decode(ids[start: start + max_excerpt_tokens])
        else:
            text = decode(ids[:max_excerpt_tokens])
        out.append(Prompt(filing_id=k[0], section=k[1], text=make_prompt(text, encode, decode, max_excerpt_tokens),
                          excerpt_truncated=len(ids) > max_excerpt_tokens))
        i += 1
    return out
