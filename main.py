"""Entry point: closed-form quantities, data preparation, coverage and calibration experiments, full runs."""
import argparse
import json
import math
from pathlib import Path

from data import edgar
from data.lengths import SentenceSegmenter, TokenCounter, build_units, length_stats, read_units, unit_table, write_units
from evaluation.config import load_config
from evaluation.protocol import (ablation, adaptive_editor, aggregate, build_context, commitment, full_null_fpr,
                                 hard_negatives, prepare_length, run_seed)
from evaluation.report import calibration_tables, coverage_tables
from theory import horizon as T
from watermark.hostlm import load_host_lm

ROOT = Path(__file__).resolve().parent


def _results_dir(cfg):
    rd = ROOT / cfg["paths"]["results_dir"]
    rd.mkdir(parents=True, exist_ok=True)
    return rd


def _resolve(cfg):
    for k in ("sections", "raw_dir", "units", "cache_dir"):
        cfg["paths"][k] = str(ROOT / cfg["paths"][k])
    return cfg


def cmd_theory(args):
    """Closed-form horizon and bracketing lengths over the parameter grids, or at one operating point."""
    if args.alpha is not None:
        a, b, g, e = args.alpha, args.beta, args.gamma, args.eta
        out = {"alpha": a, "beta": b, "gamma": g, "eta": e, "L_star": T.L_star(a, b, g, e), "L_ach": T.L_ach(a, b, g, e),
               "L_op": T.L_op(a, b, g, e), "L_clt": T.L_clt(a, b, g, e), "L_exact": T.L_exact(a, b, g, e),
               "delta_star": T.delta_star(g)}
        print(json.dumps(out, indent=1))
        return
    out = {
        "horizon_grid": {name: {f"alpha={a:g},beta={b:g}": math.ceil(T.L_star(a, b, g, e)) for a, b in T.GRID_COLUMNS}
                         for name, g, e in T.GRID_REGIMES},
        "resampling_probe": {name: {"gamma": g, "eta": e, "L_star": round(T.L_star(.01, .05, g, e)),
                                    "L_clt": round(T.L_clt(.01, .05, g, e)), "L_exact": T.L_exact(.01, .05, g, e)}
                             for name, g, e in T.TIGHT_ROWS},
        "frontier": {f"eta={eta}": {f"{a:g}": round(T.L_star(a, .05, .02, eta), 1) for a in T.FRONTIER_ALPHAS}
                     for eta in (0.2, 0.5, 0.8)},
        "length_constants": {f"{a:g}": {"L_ach": round(T.L_ach(a, .05, .02, 1.0, leading_order=True) * .02, 2),
                                        "L_op": round(T.L_op(a, .05, .02, 1.0, leading_order=True) * .02, 2),
                                        "L_clt": round(T.L_clt_constant_single_variance(a, .05), 2),
                                        "L_star": round(T.horizon_constant(a, .05), 2)} for a in T.FRONTIER_ALPHAS},
    }
    if args.config:
        rd = _results_dir(load_config(args.config))
        (rd / "theory.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def cmd_prepare_data(args):
    """Downloads a seeded sample of filings and writes their risk-factor and MD&A sections."""
    cfg = _resolve(load_config(args.config))
    d = cfg["data"]
    n = args.n_filings or int(d["n_filings"])
    out = Path(cfg["paths"]["sections"])
    out.parent.mkdir(parents=True, exist_ok=True)
    done = edgar.processed_filings(out) | ({r.filing_id for r in edgar.read_sections(out)} if out.exists() else set())
    client = edgar.SECClient(args.user_agent)
    refs = edgar.sample_filings(client, n, years=tuple(d["years"]), forms=tuple(d.get("forms", edgar.FORMS)),
                                max_per_company=int(d["max_per_company"]), seed=int(d["sample_seed"]))
    Path(str(out) + ".manifest.json").write_text(json.dumps([r.__dict__ for r in refs], indent=1))
    refs = [r for r in refs if r.filing_id not in done]
    print(f"{len(refs)} filings to download ({len(done)} already present)")
    raw = Path(cfg["paths"]["raw_dir"]) if args.keep_raw else None
    n_rec = edgar.download_and_extract(client, refs, out, raw_dir=raw)
    print(f"wrote {n_rec} section records to {out}")


def cmd_lengths(args):
    """Segments sections into sentences, counts tokens, and writes length statistics and unit fractions."""
    cfg = _resolve(load_config(args.config))
    d = cfg["data"]
    seg = SentenceSegmenter(d["spacy_model"])
    counter = TokenCounter(d["length_tokenizer"], revision=d.get("length_tokenizer_revision"))
    sections = list(edgar.read_sections(Path(cfg["paths"]["sections"])))
    units = Path(cfg["paths"]["units"])
    n = write_units(build_units(sections, seg, counter), units)
    null_only = set(d.get("null_only_sections") or ())
    sent = [u.n_tokens for u in read_units(units, "sentence") if u.section not in null_only]
    exc = [u.n_tokens for u in read_units(units, "excerpt") if u.section not in null_only]
    n_null_only = sum(s.section in null_only for s in sections)
    stats = {"n_sections": len(sections) - n_null_only, "n_null_only_sections": n_null_only, "n_units": n,
             "spacy_model_version": seg.version, "tokenizer": counter.name,
             "sentence": length_stats(sent), "excerpt": length_stats(exc)}
    etas = tuple(cfg["unit_table_etas"].items())
    tab = unit_table(sent, exc, alpha=float(cfg["alpha"]), beta=float(cfg["beta"]), gamma=float(cfg["gamma_budget"]), etas=etas)
    rd = _results_dir(cfg)
    (rd / "length_stats.json").write_text(json.dumps(stats, indent=1))
    (rd / "unit_fractions.json").write_text(json.dumps(tab, indent=1))
    print(json.dumps({"length_stats": stats, "unit_fractions": tab}, indent=1))


def cmd_coverage(args):
    """Coverage and false-positive rates of the watermark tests and the conformal test under each edit channel."""
    cfg = _resolve(load_config(args.config))
    channels = args.channels or (cfg.get("sweep_channels") if args.sweep else None) or cfg["channels"]
    seeds = args.seeds or cfg["seeds"]
    lengths = cfg["length_sweep"] if args.sweep else cfg["lengths"]
    rd = _results_dir(cfg)
    ctx = build_context(cfg)
    edit_channels = {}
    summary = {}
    tag = "coverage_sweep" if args.sweep else "coverage"
    for L in lengths:
        prepare_length(ctx, int(L))
        (rd / f"commitment_L{L}.json").write_text(json.dumps(commitment(ctx), indent=1))
        results = []
        for seed in seeds:
            r = run_seed(ctx, int(seed), int(L), channels, edit_channels=edit_channels)
            (rd / f"{tag}_L{L}_seed{seed}.json").write_text(json.dumps(r, indent=1))
            results.append(r)
        summary[str(L)] = aggregate(results)
    (rd / f"{tag}_summary.json").write_text(json.dumps(summary, indent=1))
    md = coverage_tables(cfg, summary, channels)
    (rd / f"{tag}_tables.md").write_text(md)
    print(md)


def cmd_calibration(args):
    """False-positive calibration on the null set, the score-variant ablation, hard negatives and the adaptive editor."""
    cfg = _resolve(load_config(args.config))
    seeds = args.seeds or cfg["seeds"]
    L = int(cfg["L"])
    rd = _results_dir(cfg)
    ctx = prepare_length(build_context(cfg), L)
    out = {"calibration": [], "ablation": [], "hard_negatives": [], "adaptive": []}
    edit_channels = {}
    for seed in seeds:
        out["calibration"].append(full_null_fpr(ctx, int(seed), L, int(cfg["calibration_eval"]["n_null_full"])))
        out["ablation"].append(ablation(ctx, int(seed), L, cfg["retrieval"]["ablation_channel"],
                                        cfg["retrieval"]["ablation_variants"], cfg["retrieval"]["calibration_sizes"],
                                        edit_channels=edit_channels))
    if not args.skip_hard_negatives:
        hn = cfg["hard_negatives"]
        gen = ctx.lm if hn["generator"]["model"] == cfg["host"]["model"] else load_host_lm(hn["generator"])
        for seed in seeds:
            out["hard_negatives"].append(hard_negatives(ctx, int(seed), L, gen, int(hn["n_calibration"]), int(hn["n_test"]),
                                                        null_only_fraction=float(hn.get("null_only_fraction", 0.0))))
    if not args.skip_adaptive:
        for seed in seeds:
            out["adaptive"].append(adaptive_editor(ctx, int(seed), L, cfg["edits"]["adaptive"]["base"],
                                                   cfg["retrieval"]["adaptive_variants"], edit_channels=edit_channels))
    (rd / "calibration.json").write_text(json.dumps(out, indent=1))
    md = calibration_tables(cfg, out)
    (rd / "calibration_tables.md").write_text(md)
    print(md)


def cmd_all(args):
    """Runs every stage on one configuration: theory, lengths, coverage, the length sweep and calibration."""
    cfg_path = args.config
    cfg = load_config(cfg_path)
    ns = argparse.Namespace
    cmd_theory(ns(alpha=None, config=cfg_path))
    cmd_lengths(ns(config=cfg_path))
    cmd_coverage(ns(config=cfg_path, sweep=False, channels=None, seeds=None))
    cmd_coverage(ns(config=cfg_path, sweep=True, channels=cfg.get("sweep_channels"), seeds=None))
    cmd_calibration(ns(config=cfg_path, seeds=None, skip_hard_negatives=False, skip_adaptive=False))
    print("run complete")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command")
    p = sub.add_parser("theory")
    p.add_argument("--config", default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.02)
    p.add_argument("--eta", type=float, default=0.5)
    p.set_defaults(func=cmd_theory)
    p = sub.add_parser("prepare-data")
    p.add_argument("--config", required=True)
    p.add_argument("--user-agent", required=True)
    p.add_argument("--n-filings", type=int, default=None)
    p.add_argument("--keep-raw", action="store_true")
    p.set_defaults(func=cmd_prepare_data)
    p = sub.add_parser("lengths")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_lengths)
    p = sub.add_parser("coverage")
    p.add_argument("--config", required=True)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--channels", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.set_defaults(func=cmd_coverage)
    p = sub.add_parser("calibration")
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--skip-hard-negatives", action="store_true")
    p.add_argument("--skip-adaptive", action="store_true")
    p.set_defaults(func=cmd_calibration)
    p = sub.add_parser("all")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_all)
    args = ap.parse_args()
    if args.command is None:
        ap.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
