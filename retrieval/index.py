"""Retrieval score against the output log: embedding, n-gram, exact n-gram, LCS and hybrid similarities."""
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
SCORE_VARIANTS = ("hybrid", "embed", "ngram", "exact_ngram", "lcs")


def words(text):
    return _TOKEN_RE.findall(text.lower())


def ngrams(tokens, n):
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


class Encoder:
    name = "abstract"
    n_truncated = 0

    def describe(self):
        return {"type": self.name}

    def encode(self, texts):
        raise NotImplementedError


class HFEncoder(Encoder):
    """Mean-pooled, L2-normalised transformer embeddings with a fixed text prefix."""

    name = "hf"

    def __init__(self, model_name="intfloat/e5-large-v2", prefix="query: ", device=None, batch_size=32,
                 max_length=512, revision=None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModel.from_pretrained(model_name, revision=revision).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.prefix = prefix
        self.batch_size = batch_size
        self.max_length = max_length
        self.name = model_name
        self.revision = revision
        self.n_truncated = 0

    def describe(self):
        return {"type": "hf", "model": self.name, "revision": self.revision, "prefix": self.prefix,
                "max_length": self.max_length}

    def encode(self, texts):
        """Embeddings of the prefixed texts; n_truncated counts inputs cut at max_length."""
        torch = self.torch
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = [self.prefix + t for t in texts[i: i + self.batch_size]]
            self.n_truncated += sum(len(ids) > self.max_length for ids in self.tokenizer(batch)["input_ids"])
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length,
                                 return_tensors="pt").to(self.device)
            with torch.inference_mode():
                h = self.model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1))


class NgramBM25:
    """BM25 over word n-gram terms, with scores normalised by the query's self-score."""

    def __init__(self, docs, n=5, k1=1.5, b=0.75):
        self.n, self.k1, self.b = n, k1, b
        self.doc_terms = [Counter(ngrams(words(d), n)) for d in docs]
        self.doc_len = np.array([sum(c.values()) for c in self.doc_terms], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if len(docs) else 1.0
        self.N = len(docs)
        self.postings = defaultdict(list)
        for j, c in enumerate(self.doc_terms):
            for t, tf in c.items():
                self.postings[t].append((j, tf))

    def _idf(self, term):
        p = self.postings.get(term)
        df = len(p) if p else 0
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def _bm25_term(self, idf, tf, dl):
        return idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))

    def scores(self, query):
        q = Counter(ngrams(words(query), self.n))
        out = np.zeros(self.N)
        if not q:
            return out
        self_score = 0.0
        qdl = float(sum(q.values()))
        for t, qtf in q.items():
            idf = self._idf(t)
            self_score += qtf * self._bm25_term(idf, qtf, qdl)
            for j, tf in self.postings.get(t, ()):
                out[j] += qtf * self._bm25_term(idf, tf, self.doc_len[j])
        if self_score <= 0:
            return out
        return np.clip(out / self_score, 0.0, 1.0)

    def exact_containment(self, query):
        """Fraction of the query's distinct n-grams present in each document."""
        q = set(ngrams(words(query), self.n))
        out = np.zeros(self.N)
        if not q:
            return out
        for t in q:
            for j, _ in self.postings.get(t, ()):
                out[j] += 1.0
        return out / len(q)


def lcs_length(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


@dataclass
class IndexConfig:
    score: str = "hybrid"
    embed_weight: float = 0.7
    ngram_weight: float = 0.3
    ngram_n: int = 5
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    def describe(self, encoder):
        d = self.__dict__.copy()
        d["encoder"] = encoder
        return d


class LogIndex:
    """Brute-force maximum similarity of a query against every log entry."""

    def __init__(self, entries, encoder, cfg):
        if cfg.score not in SCORE_VARIANTS:
            raise ValueError(f"unknown score variant {cfg.score!r}")
        self.entries = list(entries)
        self.cfg = cfg
        self.encoder = encoder
        self.bm25 = NgramBM25(self.entries, n=cfg.ngram_n, k1=cfg.bm25_k1, b=cfg.bm25_b)
        if cfg.score in ("hybrid", "embed"):
            if encoder is None:
                raise ValueError("embedding-based scores need an encoder")
            self.emb = encoder.encode(self.entries)
        else:
            self.emb = None
        if cfg.score == "lcs":
            self._bow = [Counter(words(e)) for e in self.entries]
            self._n_words = [sum(b.values()) for b in self._bow]

    def sim_all(self, query, query_emb=None):
        s = self.cfg.score
        if s == "ngram":
            return self.bm25.scores(query)
        if s == "exact_ngram":
            return self.bm25.exact_containment(query)
        if s == "lcs":
            return self._lcs_scores(query)
        if query_emb is None:
            query_emb = self.encoder.encode([query])[0]
        cos = self.emb @ query_emb
        if s == "embed":
            return cos
        return self.cfg.embed_weight * cos + self.cfg.ngram_weight * self.bm25.scores(query)

    def _lcs_scores(self, query):
        """LCS over words divided by the longer length; entries whose bag-of-words bound cannot exceed the best are skipped."""
        qw = words(query)
        qb = Counter(qw)
        out = np.zeros(len(self.entries))
        if not qw:
            return out
        bound = np.array([sum((qb & b).values()) / max(len(qw), n, 1) for b, n in zip(self._bow, self._n_words)])
        best = 0.0
        for j in np.argsort(-bound, kind="stable"):
            if bound[j] <= best:
                break
            out[j] = lcs_length(qw, words(self.entries[j])) / max(len(qw), self._n_words[j], 1)
            best = max(best, out[j])
        return out

    def score(self, query, query_emb=None):
        sims = self.sim_all(query, query_emb)
        return float(sims.max()) if len(sims) else 0.0

    def scores(self, queries):
        if self.cfg.score in ("hybrid", "embed"):
            embs = self.encoder.encode(list(queries))
            return np.array([self.score(q, e) for q, e in zip(queries, embs)])
        return np.array([self.score(q) for q in queries])


def build_encoder(cfg):
    kind = cfg.get("type", "hf")
    if kind == "hf":
        return HFEncoder(cfg.get("model", "intfloat/e5-large-v2"), prefix=cfg.get("prefix", "query: "),
                         device=cfg.get("device"), batch_size=int(cfg.get("batch_size", 32)),
                         max_length=int(cfg.get("max_length", 512)), revision=cfg.get("revision"))
    raise ValueError(f"unknown encoder type {kind!r}")


def build_index_config(cfg):
    return IndexConfig(score=cfg.get("score", "hybrid"), embed_weight=float(cfg.get("embed_weight", 0.7)),
                       ngram_weight=float(cfg.get("ngram_weight", 0.3)), ngram_n=int(cfg.get("ngram_n", 5)),
                       bm25_k1=float(cfg.get("bm25_k1", 1.5)), bm25_b=float(cfg.get("bm25_b", 0.75)))
