"""Experiment protocol: fixed context, per-length log and calibration set, per-seed coverage and false-positive evaluation."""

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from data.splits import Corpus, Prompt, SplitPlan, draw_prompts, draw_windows, plan_splits
from data.strata import BoilerplateDetector
from retrieval.conformal import ConformalDetector, MondrianConformalDetector, beta_spread, commit
from retrieval.index import Encoder, IndexConfig, LogIndex, build_encoder, build_index_config
from watermark.edits import EditChannel, build_edit_channel
from watermark.estimation import eta_hat, gamma_hat
from watermark.greenlist import GreenListWatermark, ZTestDetector, make_watermark
from watermark.hostlm import HostLM, load_host_lm


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


class JsonCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get_value(self, key: dict):
        p = self.root / f"{_hash(key)}.json"
        return json.loads(p.read_text())["value"] if p.exists() else None

    def put(self, key: dict, value) -> None:
        p = self.root / f"{_hash(key)}.json"
        p.write_text(json.dumps({"key": key, "value": value}))


def _mean(x: Sequence[float]) -> float:
    return float(np.mean(x)) if len(x) else float("nan")


@dataclass
class Context:
    cfg: dict
    lm: HostLM
    corpus: Corpus
    plan: SplitPlan
    cache: JsonCache
    watermarks: dict[str, GreenListWatermark]
    encoder: Optional[Encoder]
    index_cfg: IndexConfig
    boilerplate: BoilerplateDetector
    log_prompts: list[Prompt] = field(default_factory=list)
    L: Optional[int] = None
    log_texts: list[str] = field(default_factory=list)
    calibration: list[dict] = field(default_factory=list)
    index: Optional[LogIndex] = None
    cal_scores: Optional[np.ndarray] = None
    n_encoder_truncated_index: int = 0
    log: Callable[[str], None] = print

    def generate(self, prompts: Sequence[Prompt], L: int, seed: int, wm: Optional[GreenListWatermark] = None,
                 record_kl: bool = False, tag: str = "") -> list[dict]:
        dec = self.cfg["decoding"]
        host = {k: self.cfg["host"].get(k) for k in ("type", "revision", "dtype", "quantization", "tokenizer", "tokenizer_revision")}
        key = {"kind": "gen", "model": self.lm.name, "host": host, "L": L, "seed": seed, "tag": tag,
               "wm": None if wm is None else {"scheme": wm.name, "key": wm.key, "g": wm.green_fraction, "bias": wm.bias},
               "prompts": _hash([p.text for p in prompts]), "T": dec["temperature"], "top_p": dec["top_p"]}
        cached = self.cache.get_value(key)
        if cached is not None:
            return cached
        out = []
        t0 = time.time()
        for i, p in enumerate(prompts):
            r = self.lm.generate(self.lm.encode(p.text), L, seed=seed * 1_000_003 + i,
                                 temperature=dec["temperature"], top_p=dec["top_p"],
                                 processor=None if wm is None else wm.process_logits, record_kl=record_kl)
            out.append({"text": self.lm.decode(r.ids), "ids": r.ids, "prompt_filing": p.filing_id,
                        "per_step_kl": r.per_step_kl})
        self.log(f"  generated {len(prompts)} x {L} tokens ({tag or 'base'}) in {time.time() - t0:.1f}s")
        self.cache.put(key, out)
        return out

    def edit(self, channel: EditChannel, texts: Sequence[str], seed: int, tag: str) -> list[str]:
        cacheable = channel.name in set(self.cfg.get("cache_channels", ["dipper", "gpt4o", "recursive"]))
        key = {"kind": "edit", "channel": channel.describe(), "seed": seed, "tag": tag, "texts": _hash(list(texts))}
        if cacheable:
            cached = self.cache.get_value(key)
            if cached is not None:
                return cached
        rng = np.random.default_rng([seed, int(hashlib.sha256(channel.name.encode()).hexdigest()[:8], 16)])
        t0 = time.time()
        out = [channel.apply(t, rng) for t in texts]
        if cacheable:
            self.cache.put(key, out)
            self.log(f"  edited {len(texts)} texts with {channel.name} in {time.time() - t0:.1f}s")
        return out

    def encode(self, text: str) -> list[int]:
        return self.lm.encode(text)

    def decode(self, ids) -> str:
        return self.lm.decode(ids)


def build_context(cfg: dict, log: Callable[[str], None] = print) -> Context:
    """Loads the host model and corpus, plans the filing-level splits and draws the log prompts."""
    paths = cfg["paths"]
    lm = load_host_lm(cfg["host"])
    corpus = Corpus.load(Path(paths["units"]), null_only_sections=tuple(cfg["data"].get("null_only_sections") or ()))
    sp = cfg["splits"]
    plan = plan_splits(corpus.filing_ids, split_seed=int(sp["split_seed"]),
                       prompt_fraction=float(sp.get("prompt_fraction", 0.5)),
                       calibration_fraction=float(sp.get("calibration_fraction", 0.5)),
                       null_only_ids=corpus.null_only_ids)
    wms = {name: make_watermark(w, lm.vocab_size) for name, w in (cfg.get("watermarks") or {}).items()}
    bp = cfg.get("strata", {}).get("boilerplate", {})
    boilerplate = BoilerplateDetector(n=int(bp.get("n", 5)), threshold=float(bp.get("threshold", 0.3)),
                                      min_filings=int(bp.get("min_filings", 2)))
    boilerplate.fit((k[0], u.text) for k, units in corpus.sentences.items() for u in units)
    ctx = Context(cfg=cfg, lm=lm, corpus=corpus, plan=plan, cache=JsonCache(Path(paths["cache_dir"])), watermarks=wms,
                  encoder=build_encoder(cfg["retrieval"]["encoder"]), index_cfg=build_index_config(cfg["retrieval"]["index"]),
                  boilerplate=boilerplate, log=log)
    log(f"corpus: {len(corpus.filing_ids)} filings; prompt filings {len(plan.prompt_filings)}, "
        f"calibration filings {len(plan.calibration_filings)}, test null filings {len(plan.test_null_filings)}, "
        f"null-only filings {len(plan.null_only_filings)}")
    rng = np.random.default_rng(int(sp["log_seed"]))
    ctx.log_prompts = draw_prompts(corpus, plan.prompt_filings, int(sp["log_size"]), rng, ctx.encode, ctx.decode,
                                   max_excerpt_tokens=int(cfg["host"]["max_excerpt_tokens"]))
    log(f"log prompts with the excerpt cut at {cfg['host']['max_excerpt_tokens']} tokens: "
        f"{sum(p.excerpt_truncated for p in ctx.log_prompts)} of {len(ctx.log_prompts)}")
    return ctx


def prepare_length(ctx: Context, L: int) -> Context:
    """Generates the output log, draws the calibration windows and builds the index for output length L."""
    if ctx.L == L:
        return ctx
    sp = ctx.cfg["splits"]
    ctx.L = L
    ctx.log_texts = [g["text"] for g in ctx.generate(ctx.log_prompts, L, int(sp["log_seed"]), tag="log")]
    rng = np.random.default_rng(int(sp["calibration_seed"]))
    ctx.calibration = draw_windows(ctx.corpus, ctx.plan.calibration_filings, int(sp["calibration_size"]), L, rng,
                                   ctx.encode, ctx.decode)
    n0 = ctx.encoder.n_truncated if ctx.encoder else 0
    ctx.index = LogIndex(ctx.log_texts, ctx.encoder, ctx.index_cfg)
    ctx.cal_scores = ctx.index.scores([c["text"] for c in ctx.calibration])
    ctx.n_encoder_truncated_index = (ctx.encoder.n_truncated if ctx.encoder else 0) - n0
    return ctx


def commitment(ctx: Context) -> dict:
    """Commitment to the log and the similarity configuration."""
    sim_cfg = ctx.index_cfg.describe(ctx.encoder.describe() if ctx.encoder else "none")
    return commit(ctx.log_texts, sim_cfg).__dict__


def run_seed(ctx: Context, seed: int, L: int, channel_names: Sequence[str], *,
             edit_channels: Optional[dict[str, EditChannel]] = None) -> dict:
    """Watermark tests and the conformal test under each edit channel for one seed and output length."""
    cfg = ctx.cfg
    prepare_length(ctx, L)
    alpha = float(cfg["alpha"])
    n_pos, n_neg = int(cfg["n_positive"]), int(cfg["n_negative"])
    rng = np.random.default_rng(seed)
    ctx.log(f"seed {seed}, L={L}")
    n0 = ctx.encoder.n_truncated if ctx.encoder else 0

    pos_prompts = draw_prompts(ctx.corpus, ctx.plan.prompt_filings, n_pos, rng, ctx.encode, ctx.decode,
                               max_excerpt_tokens=int(cfg["host"]["max_excerpt_tokens"]))
    t2_idx = rng.choice(len(ctx.log_texts), size=min(n_pos, len(ctx.log_texts)), replace=False)
    t2_positives = [ctx.log_texts[i] for i in t2_idx]
    negatives = draw_windows(ctx.corpus, ctx.plan.test_null_filings, n_neg, L, rng, ctx.encode, ctx.decode)
    neg_texts = [n["text"] for n in negatives]
    neg_groups = ctx.boilerplate.groups(negatives)

    t1_positives: dict[str, list[dict]] = {}
    for name, wm in ctx.watermarks.items():
        t1_positives[name] = ctx.generate(pos_prompts, L, seed, wm=wm, record_kl=True, tag=f"t1_{name}")

    edit_channels = {} if edit_channels is None else edit_channels
    det_conf = ConformalDetector(alpha, ctx.cal_scores, tie_break_seed=seed)
    flags_neg_t2 = det_conf.detect(ctx.index.scores(neg_texts))
    result = {"seed": seed, "L": L, "alpha": alpha, "n_positive": n_pos, "n_negative": n_neg, "channels": {},
              "gamma_hat": {name: gamma_hat([g["per_step_kl"] for g in gens]) for name, gens in t1_positives.items()},
              "boilerplate_fraction": float(np.mean([g == "boilerplate" for g in neg_groups])),
              "t2": {"fpr": _mean(flags_neg_t2),
                     "fpr_boilerplate": _mean([f for f, g in zip(flags_neg_t2, neg_groups) if g == "boilerplate"]),
                     "threshold": det_conf.threshold, "n_calibration": det_conf.n,
                     "beta_spread": beta_spread(det_conf.n, alpha).__dict__},
              "truncation": {"positive_prompts_excerpt": sum(p.excerpt_truncated for p in pos_prompts),
                             "log_prompts_excerpt": sum(p.excerpt_truncated for p in ctx.log_prompts),
                             "encoder_inputs_log_and_calibration": ctx.n_encoder_truncated_index}}

    ids_before = {name: [ctx.encode(g["text"]) for g in t1_positives[name]] for name in ctx.watermarks}
    neg_ids_clean = [ctx.encode(t) for t in neg_texts]
    z_before = {name: {"h1": [ZTestDetector(wm, alpha).statistic(i) for i in ids_before[name]],
                       "h0": [ZTestDetector(wm, alpha).statistic(i) for i in neg_ids_clean]}
                for name, wm in ctx.watermarks.items()}

    for cname in channel_names:
        ch = edit_channels.get(cname) or build_edit_channel(cname, cfg["edits"], lm=ctx.lm, cache=edit_channels,
                                                            encoder=ctx.encoder)
        edit_channels[cname] = ch
        cres: dict = {"t1": {}, "t2": {}}
        neg_edit = ctx.edit(ch, neg_texts, seed, "neg") if cfg.get("edit_negatives", False) else neg_texts
        neg_ids = [ctx.encode(t) for t in neg_edit]

        for name, wm in ctx.watermarks.items():
            det = ZTestDetector(wm, alpha)
            pos_edit = ctx.edit(ch, [g["text"] for g in t1_positives[name]], seed, f"pos_{name}")
            pos_ids = [ctx.encode(t) for t in pos_edit]
            z_pos = [det.statistic(ids) for ids in pos_ids]
            z_neg = [det.statistic(ids) for ids in neg_ids]
            flags_pos = [z > det.threshold for z in z_pos]
            flags_neg = [z > det.threshold for z in z_neg]
            entry = {"tpr": _mean(flags_pos), "fpr": _mean(flags_neg),
                     "fpr_boilerplate": _mean([f for f, g in zip(flags_neg, neg_groups) if g == "boilerplate"]),
                     "mean_green_rate_pos": _mean([wm.green_fraction_of(ids) for ids in pos_ids]),
                     "mean_green_rate_neg": _mean([wm.green_fraction_of(ids) for ids in neg_ids]),
                     "mean_scored_len_pos": _mean([wm.green_count(ids)[1] for ids in pos_ids])}
            est = eta_hat(z_before[name]["h1"], z_before[name]["h0"], z_pos, z_neg)
            entry["eta_hat"] = est.eta
            entry["kl_before"], entry["kl_after"] = est.kl_before, est.kl_after
            cres["t1"][name] = entry

        t2_edit = ctx.edit(ch, t2_positives, seed, "t2pos")
        cres["t2"] = {"tpr": _mean(det_conf.detect(ctx.index.scores(t2_edit)))}
        result["channels"][cname] = cres
        ctx.log(f"  {cname}: " + ", ".join(f"{k} TPR={v['tpr']:.2f}/FPR={v['fpr']:.3f}" for k, v in cres["t1"].items())
                + f", conformal TPR={cres['t2']['tpr']:.2f}/FPR={result['t2']['fpr']:.3f}")
    result["truncation"]["encoder_inputs_queries"] = (ctx.encoder.n_truncated if ctx.encoder else 0) - n0
    return result


def full_null_fpr(ctx: Context, seed: int, L: int, n_null: int) -> dict:
    """False-positive rate on a third-party null set, marginal and on the boilerplate stratum."""
    prepare_length(ctx, L)
    alpha = float(ctx.cfg["alpha"])
    rng = np.random.default_rng([seed, 7])
    nulls = draw_windows(ctx.corpus, ctx.plan.test_null_filings, n_null, L, rng, ctx.encode, ctx.decode)
    groups = ctx.boilerplate.groups(nulls)
    texts = [n["text"] for n in nulls]
    out = {"seed": seed, "L": L, "n_null": n_null, "boilerplate_fraction": float(np.mean([g == "boilerplate" for g in groups])),
           "methods": {}}
    ids = [ctx.encode(t) for t in texts]
    for name, wm in ctx.watermarks.items():
        det = ZTestDetector(wm, alpha)
        flags = [det.detect(i) for i in ids]
        out["methods"][name] = {"fpr": _mean(flags),
                                "fpr_boilerplate": _mean([f for f, g in zip(flags, groups) if g == "boilerplate"])}
    det_conf = ConformalDetector(alpha, ctx.cal_scores, tie_break_seed=seed)
    flags = det_conf.detect(ctx.index.scores(texts))
    out["methods"]["conformal"] = {"fpr": _mean(flags),
                                   "fpr_boilerplate": _mean([f for f, g in zip(flags, groups) if g == "boilerplate"])}
    return out


def ablation(ctx: Context, seed: int, L: int, channel_name: str, variants: Sequence[str],
             calibration_sizes: Sequence[int], edit_channels: Optional[dict] = None) -> dict:
    """Coverage and false-positive rate per score variant and per calibration-set size."""
    cfg = ctx.cfg
    prepare_length(ctx, L)
    alpha = float(cfg["alpha"])
    n_pos, n_neg = int(cfg["n_positive"]), int(cfg["n_negative"])
    rng = np.random.default_rng([seed, 11])
    edit_channels = {} if edit_channels is None else edit_channels
    ch = edit_channels.get(channel_name) or build_edit_channel(channel_name, cfg["edits"], lm=ctx.lm, cache=edit_channels,
                                                               encoder=ctx.encoder)
    edit_channels[channel_name] = ch
    t2_idx = rng.choice(len(ctx.log_texts), size=min(n_pos, len(ctx.log_texts)), replace=False)
    pos = ctx.edit(ch, [ctx.log_texts[i] for i in t2_idx], seed, "abl_pos")
    negatives = draw_windows(ctx.corpus, ctx.plan.test_null_filings, n_neg, L, rng, ctx.encode, ctx.decode)
    neg = [n["text"] for n in negatives]
    cal = [c["text"] for c in ctx.calibration]
    out = {"seed": seed, "L": L, "channel": channel_name, "variants": {}, "calibration_sizes": {}}
    for v in variants:
        icfg = IndexConfig(**{**ctx.index_cfg.__dict__, "score": v})
        idx = LogIndex(ctx.log_texts, ctx.encoder if v in ("hybrid", "embed") else None, icfg)
        det = ConformalDetector(alpha, idx.scores(cal), tie_break_seed=seed)
        out["variants"][v] = {"tpr": _mean(det.detect(idx.scores(pos))), "fpr": _mean(det.detect(idx.scores(neg)))}
        ctx.log(f"  ablation {v}: TPR={out['variants'][v]['tpr']:.2f} FPR={out['variants'][v]['fpr']:.3f}")
    pos_scores, neg_scores = ctx.index.scores(pos), ctx.index.scores(neg)
    for n in calibration_sizes:
        n = int(min(n, len(ctx.cal_scores)))
        det = ConformalDetector(alpha, ctx.cal_scores[:n], tie_break_seed=seed)
        out["calibration_sizes"][n] = {"tpr": _mean(det.detect(pos_scores)), "fpr": _mean(det.detect(neg_scores)),
                                       "beta_spread": beta_spread(n, alpha).__dict__}
    return out


def hard_negatives(ctx: Context, seed: int, L: int, generator: HostLM, n_cal_hard: int, n_test_hard: int,
                   null_only_fraction: float = 0.0) -> dict:
    """False-positive rate on second-generator summaries pooled with null-only windows, marginal and group-conditional."""
    cfg = ctx.cfg
    prepare_length(ctx, L)
    alpha = float(cfg["alpha"])
    dec = cfg["decoding"]
    rng = np.random.default_rng([seed, 12])
    n_neg = int(cfg["n_negative"])
    n_total = n_cal_hard + n_test_hard
    null_only = set(ctx.plan.null_only_filings)
    n_text = int(round(null_only_fraction * n_total)) if null_only else 0
    hard_prompts = draw_prompts(ctx.corpus, [f for f in ctx.plan.null_filings if f not in null_only], n_total - n_text, rng,
                                ctx.encode, ctx.decode, max_excerpt_tokens=int(cfg["host"]["max_excerpt_tokens"]))
    key = {"kind": "hard", "model": generator.name, "generator": cfg["hard_negatives"]["generator"], "L": L, "seed": seed,
           "prompts": _hash([p.text for p in hard_prompts]), "T": dec["temperature"], "top_p": dec["top_p"]}
    hard = ctx.cache.get_value(key)
    if hard is None:
        hard = [generator.decode(generator.generate(generator.encode(p.text), L, seed=seed * 7919 + i,
                                                    temperature=dec["temperature"], top_p=dec["top_p"]).ids)
                for i, p in enumerate(hard_prompts)]
        ctx.cache.put(key, hard)
    if n_text:
        hard = hard + [w["text"] for w in draw_windows(ctx.corpus, sorted(null_only), n_text, L, rng, ctx.encode, ctx.decode)]
        hard = [hard[i] for i in rng.permutation(len(hard))]
    hard_cal, hard_test = hard[:n_cal_hard], hard[n_cal_hard:]
    negatives = draw_windows(ctx.corpus, ctx.plan.test_null_filings, n_neg, L, rng, ctx.encode, ctx.decode)
    std_test = [n["text"] for n in negatives]
    s_hard_cal, s_hard_test, s_std = ctx.index.scores(hard_cal), ctx.index.scores(hard_test), ctx.index.scores(std_test)
    marginal = ConformalDetector(alpha, ctx.cal_scores, tie_break_seed=seed)
    mondrian = MondrianConformalDetector(alpha, np.concatenate([ctx.cal_scores, s_hard_cal]),
                                         ["standard"] * len(ctx.cal_scores) + ["hard_negative"] * len(s_hard_cal),
                                         tie_break_seed=seed)
    return {"seed": seed, "L": L, "generator": generator.name, "n_null_only": n_text,
            "n_prompts_excerpt_truncated": sum(p.excerpt_truncated for p in hard_prompts),
            "standard_marginal": _mean(marginal.detect(s_std)),
            "hard_marginal": _mean(marginal.detect(s_hard_test)),
            "hard_group_conditional": _mean(mondrian.detect(s_hard_test, ["hard_negative"] * len(s_hard_test))),
            "standard_group_conditional": _mean(mondrian.detect(s_std, ["standard"] * len(s_std))),
            "thresholds": {str(k): v for k, v in mondrian.thresholds.items()}}


def adaptive_editor(ctx: Context, seed: int, L: int, base_name: str, variants: Sequence[str],
                    edit_channels: Optional[dict] = None) -> dict:
    """Coverage of the base channel and of the adaptive editor built on it, per index score variant."""
    cfg = ctx.cfg
    prepare_length(ctx, L)
    alpha = float(cfg["alpha"])
    n_pos = int(cfg["n_positive"])
    rng = np.random.default_rng([seed, 13])
    edit_channels = {} if edit_channels is None else edit_channels
    for name in (base_name, "adaptive"):
        if name not in edit_channels:
            edit_channels[name] = build_edit_channel(name, cfg["edits"], lm=ctx.lm, cache=edit_channels, encoder=ctx.encoder)
    t2_idx = rng.choice(len(ctx.log_texts), size=min(n_pos, len(ctx.log_texts)), replace=False)
    src = [ctx.log_texts[i] for i in t2_idx]
    edited = {"reference": ctx.edit(edit_channels[base_name], src, seed, "adp_ref"),
              "adaptive": ctx.edit(edit_channels["adaptive"], src, seed, "adp_adv")}
    cal = [c["text"] for c in ctx.calibration]
    out = {"seed": seed, "L": L, "base": base_name, "variants": {}}
    for v in variants:
        icfg = IndexConfig(**{**ctx.index_cfg.__dict__, "score": v})
        idx = LogIndex(ctx.log_texts, ctx.encoder if v in ("hybrid", "embed") else None, icfg)
        det = ConformalDetector(alpha, idx.scores(cal), tie_break_seed=seed)
        out["variants"][v] = {k: _mean(det.detect(idx.scores(t))) for k, t in edited.items()}
        ctx.log(f"  adaptive editor, {v} index: reference TPR={out['variants'][v]['reference']:.2f}, "
                f"adaptive TPR={out['variants'][v]['adaptive']:.2f}")
    return out


def aggregate(results: Sequence[dict]) -> dict:
    """Mean and standard deviation over seeds of every numeric result."""
    def collect(d, prefix, acc):
        for k, v in d.items():
            p = prefix + (k,)
            if isinstance(v, dict):
                collect(v, p, acc)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                acc.setdefault(p, []).append(float(v))

    acc: dict = {}
    for r in results:
        collect({"channels": r["channels"], "gamma_hat": r["gamma_hat"], "t2": r["t2"]}, (), acc)
    out = {}
    for p, vals in acc.items():
        out["/".join(p)] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                            "n_seeds": len(vals)}
    return out
