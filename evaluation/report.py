"""Markdown tables from aggregated coverage and calibration results."""
import numpy as np

from theory.horizon import model_power


def _get(agg, key):
    return agg.get(key, {"mean": float("nan"), "sd": 0.0})


def coverage_tables(cfg, summary_by_L, channels=None):
    """Coverage and false-positive rate per method and channel, the Gaussian model of the z-test, and the estimated gamma."""
    lines = []
    channels = channels or cfg["channels"]
    alpha = float(cfg["alpha"])
    for L, agg in summary_by_L.items():
        lines.append(f"\n### L = {L}: coverage (TPR at the nominal {100 * alpha:g}% threshold), mean over seeds [max sd]\n")
        lines.append("| Method | " + " | ".join(channels) + " | max sd |")
        lines.append("|---|" + "---|" * (len(channels) + 1))
        wms = cfg.get("watermarks") or {}
        rows = [(f"T1 {wm}", f"channels/{{c}}/t1/{wm}/tpr") for wm in wms] + [("T2 conformal retrieval", "channels/{c}/t2/tpr")]
        for label, pattern in rows:
            cells = [_get(agg, pattern.format(c=c)) for c in channels]
            lines.append(f"| {label} | " + " | ".join(f"{e['mean']:.2f}" for e in cells) + f" | {max(e['sd'] for e in cells):.2f} |")
        lines.append(f"\n### L = {L}: realized false-positive rate, mean over seeds\n")
        lines.append("| Method | " + " | ".join(channels) + " |")
        lines.append("|---|" + "---|" * len(channels))
        null = "edited third-party text" if cfg.get("edit_negatives", False) else "third-party text"
        for wm in wms:
            cells = [_get(agg, f"channels/{c}/t1/{wm}/fpr")["mean"] for c in channels]
            lines.append(f"| T1 {wm} ({null}) | " + " | ".join(f"{100 * v:.1f}%" for v in cells) + " |")
        t2 = _get(agg, "t2/fpr")["mean"]
        lines.append("| T2 conformal retrieval (third-party text) | " + " | ".join(f"{100 * t2:.1f}%" for _ in channels) + " |")
        if "greenlist" in wms:
            lines.append(f"\n### L = {L}: green-list z-test and Gaussian model\n")
            lines.append("| Edit channel | eta_hat | Deployed | Model |")
            lines.append("|---|---|---|---|")
            g = _get(agg, "gamma_hat/greenlist")["mean"]
            for c in channels:
                eta = _get(agg, f"channels/{c}/t1/greenlist/eta_hat")["mean"]
                dep = _get(agg, f"channels/{c}/t1/greenlist/tpr")["mean"]
                try:
                    mod = model_power(int(L), g, max(min(eta, 1.0), 1e-6), alpha)
                except Exception:
                    mod = float("nan")
                lines.append(f"| {c} | {eta:.2f} | {dep:.2f} | {mod:.2f} |")
        lines.append(f"\n### L = {L}: estimated per-token distortion gamma_hat (nats)\n")
        for wm in wms:
            e = _get(agg, f"gamma_hat/{wm}")
            lines.append(f"- {wm}: {e['mean']:.3f} (sd {e['sd']:.3f})")
    return "\n".join(lines)


def calibration_tables(cfg, out):
    """False-positive calibration, score-variant ablation, hard negatives and the adaptive editor."""
    lines = ["### False-positive rate on the third-party null set (mean ± sd over seeds)", "",
             "| Method | Marginal | Boilerplate |", "|---|---|---|"]
    methods = list(out["calibration"][0]["methods"])
    for m in methods:
        marg = [r["methods"][m]["fpr"] for r in out["calibration"]]
        boil = [r["methods"][m]["fpr_boilerplate"] for r in out["calibration"]]
        boil = [b for b in boil if not np.isnan(b)]
        sd = 100 * np.std(marg, ddof=1) if len(marg) > 1 else 0.0
        boil_cell = f"{100 * np.mean(boil):.1f}%" if boil else "n/a"
        lines.append(f"| {m} | {100 * np.mean(marg):.2f} ± {sd:.2f}% | {boil_cell} |")
    frac = np.mean([r["boilerplate_fraction"] for r in out["calibration"]])
    lines += ["", f"boilerplate fraction of the null set: {frac:.3f}", "",
              "### Construction ablation (mean over seeds)", "", "| Variant | TPR | FPR |", "|---|---|---|"]
    for v in cfg["retrieval"]["ablation_variants"]:
        t = [r["variants"][v]["tpr"] for r in out["ablation"]]
        f = [r["variants"][v]["fpr"] for r in out["ablation"]]
        lines.append(f"| {v} | {np.mean(t):.2f} | {100 * np.mean(f):.1f}% |")
    for n in out["ablation"][0]["calibration_sizes"]:
        rows = [r["calibration_sizes"][n] for r in out["ablation"]]
        bs = rows[0]["beta_spread"]
        lines.append(f"| calibration n={n} | {np.mean([r['tpr'] for r in rows]):.2f} | "
                     f"{100 * np.mean([r['fpr'] for r in rows]):.1f} ± {100 * bs['sd']:.1f}% (Beta({bs['a']},{bs['b']})) |")
    if out["hard_negatives"]:
        lines += ["", "### Hard negatives (mean over seeds)", "", "| Null set | FPR |", "|---|---|"]
        for k in ("standard_marginal", "hard_marginal", "hard_group_conditional", "standard_group_conditional"):
            lines.append(f"| {k} | {100 * np.mean([r[k] for r in out['hard_negatives']]):.1f}% |")
        lines.append(f"\nnull-only texts among the hard negatives per seed: {np.mean([r['n_null_only'] for r in out['hard_negatives']]):.0f}")
    if out.get("adaptive"):
        lines += ["", "### Adaptive editor coverage (mean over seeds)", "", "| Index | Base channel | Adaptive editor |", "|---|---|---|"]
        for v in out["adaptive"][0]["variants"]:
            ref = np.mean([r["variants"][v]["reference"] for r in out["adaptive"]])
            adv = np.mean([r["variants"][v]["adaptive"] for r in out["adaptive"]])
            lines.append(f"| {v} | {ref:.2f} | {adv:.2f} |")
    return "\n".join(lines)
