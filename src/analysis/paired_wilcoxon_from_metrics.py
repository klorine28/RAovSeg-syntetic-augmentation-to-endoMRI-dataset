#!/usr/bin/env python3
"""
Paired-Wilcoxon test on per-subject DSC arrays across variants.

The `evaluate.py` output stores `per_subject.{mode}` records — one dict per
test subject with the DSC (and other) metrics. This script walks a set of
metrics JSONs (typically one per (variant, seed) combination), pivots to
(variant, subject) → list-of-DSCs-across-seeds, and runs paired Wilcoxon
signed-rank tests between a nominated baseline variant and every other
variant.

Why paired: the same 8 test subjects appear in every run. Testing paired
across subjects controls for per-subject difficulty (e.g., D2-005 and
D2-023 are near-impossible for any RAovSeg configuration). Testing
unpaired against 0.290 (a fixed number) misses that structure and
inflates variance.

Output:
    <out_json>       — full per-comparison stats + effect sizes
    <out_png>        — forest plot: per-variant Δ vs baseline with CI
    stdout           — human-readable summary table

Usage:
    python -m src.analysis.paired_wilcoxon_from_metrics \\
        --metrics-glob 'runs/raovseg_aug_*/predictions/metrics_ov.json' \\
        --baseline-glob 'runs/raovseg_real_only_seed*/predictions/metrics_ov.json' \\
        --variant-from-path 'runs/raovseg_aug_(.+?)_seed' \\
        --seed-from-path 'seed(\\d+)' \\
        --out-json metrics/paired_wilcoxon_ov.json \\
        --out-png  figures/fig_paired_wilcoxon_ov.png

Or, if you already have per-subject arrays as a JSON of the form
    { "variant_name": { "seed_0": { "D2-005": 0.02, "D2-016": 0.48, ... },
                        "seed_1": { ... } }, ... }
then use --precomputed <json>:
    python -m src.analysis.paired_wilcoxon_from_metrics \\
        --precomputed metrics/per_subject_dsc_dump.json \\
        --baseline real_only \\
        --out-json ... --out-png ...
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _extract_per_subject_dsc(metrics_json: dict, mode: str = "raw",
                             metric_key: str = "dsc") -> Dict[str, float]:
    """From an evaluate.py output dict, pull per-subject DSCs for a given
    postprocessing mode (usually "raw" or "postprocessed"; falls back to
    whatever's present)."""
    per_subj = metrics_json.get("per_subject", {})
    if not per_subj:
        return {}
    modes_available = list(per_subj.keys())
    if mode not in per_subj:
        mode = modes_available[0]
    out: Dict[str, float] = {}
    for rec in per_subj[mode]:
        subj = rec.get("subject") or rec.get("subject_id") or rec.get("id")
        if subj is None:
            continue
        val = rec.get(metric_key)
        if val is None:
            for k in ("dice", "dsc_mean", "dsc_value"):
                if k in rec:
                    val = rec[k]
                    break
        if val is None:
            continue
        try:
            out[str(subj)] = float(val)
        except (TypeError, ValueError):
            continue
    return out


def _load_from_globs(metrics_glob: List[str],
                     variant_regex: str,
                     seed_regex: str,
                     mode: str = "raw",
                     metric_key: str = "dsc",
                     label: str | None = None,
                     variant_name: str | None = None,
                     ) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Walk one or more glob patterns, extract per-subject DSCs, index by
    (variant, seed). If `variant_name` is set, forces every matched file to
    that variant (useful for --baseline-glob)."""
    v_re = re.compile(variant_regex)
    s_re = re.compile(seed_regex)
    out: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    n_loaded = 0
    for pat in metrics_glob:
        for p in sorted(Path().glob(pat)):
            try:
                with p.open() as f:
                    j = json.load(f)
            except Exception as e:
                print(f"  skip {p}: {type(e).__name__}: {e}")
                continue
            if variant_name is not None:
                v = variant_name
            else:
                m = v_re.search(str(p))
                if m is None:
                    print(f"  skip {p}: variant regex didn't match")
                    continue
                v = m.group(1)
            ms = s_re.search(str(p))
            s = ms.group(1) if ms else "0"
            per_subj = _extract_per_subject_dsc(j, mode=mode, metric_key=metric_key)
            if not per_subj:
                print(f"  skip {p}: no per_subject data")
                continue
            key = f"seed_{s}"
            if key in out[v]:
                print(f"  warn: {v}/{key} already loaded — overwriting with {p}")
            out[v][key] = per_subj
            n_loaded += 1
    if label:
        print(f"[{label}] loaded {n_loaded} metrics files across "
              f"{len(out)} variants")
    return out


def _paired_wilcoxon(baseline: Dict[str, List[float]],
                     variant: Dict[str, List[float]]
                     ) -> Tuple[float, float, int, List[str]]:
    """Paired signed-rank test on per-subject means across seeds.

    For each subject, take the mean DSC across seeds (baseline and
    variant). Then paired Wilcoxon on the two mean-arrays keyed by
    subject. Returns (statistic, pvalue, n_pairs, common_subjects).
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        raise SystemExit("scipy not installed. `pip install scipy` and retry.")
    common = sorted(set(baseline) & set(variant))
    if len(common) < 3:
        return (float("nan"), float("nan"), len(common), common)
    b = np.array([np.mean(baseline[k]) for k in common])
    v = np.array([np.mean(variant[k]) for k in common])
    diffs = v - b
    if np.allclose(diffs, 0):
        return (0.0, 1.0, len(common), common)
    try:
        stat, p = wilcoxon(v, b, zero_method="wilcox", alternative="two-sided")
    except ValueError as e:
        print(f"  wilcoxon failed: {e}")
        return (float("nan"), float("nan"), len(common), common)
    return (float(stat), float(p), len(common), common)


def _pivot_to_subject_lists(per_seed: Dict[str, Dict[str, float]]
                            ) -> Dict[str, List[float]]:
    """{seed_0: {subj: dsc}, ...} → {subj: [dsc_seed0, dsc_seed1, ...]}"""
    subj_lists: Dict[str, List[float]] = defaultdict(list)
    for seed_key in sorted(per_seed):
        for subj, dsc in per_seed[seed_key].items():
            subj_lists[subj].append(float(dsc))
    return dict(subj_lists)


def _summarise(subj_lists: Dict[str, List[float]]) -> dict:
    """Per-variant summary — mean over subjects of per-subject means, etc."""
    if not subj_lists:
        return {"n_subjects": 0}
    per_subj_mean = {s: float(np.mean(vs)) for s, vs in subj_lists.items()}
    per_subj_std  = {s: float(np.std(vs))  for s, vs in subj_lists.items()}
    means = np.array(list(per_subj_mean.values()))
    return {
        "n_subjects": len(subj_lists),
        "n_seeds_min": int(min(len(v) for v in subj_lists.values())),
        "n_seeds_max": int(max(len(v) for v in subj_lists.values())),
        "overall_mean_of_subj_means": float(means.mean()),
        "overall_std_of_subj_means":  float(means.std()),
        "per_subject_mean": per_subj_mean,
        "per_subject_std":  per_subj_std,
    }


def _forest_plot(comparisons: List[dict], baseline: str, out_png: Path) -> None:
    """One row per variant: Δ = variant_mean - baseline_mean with a 95 % CI
    of Δ estimated from per-subject bootstrap. Marker filled if p < 0.05."""
    if not comparisons:
        print("[plot] no comparisons; skipping forest plot")
        return
    fig, ax = plt.subplots(figsize=(9, 0.6 + 0.5 * len(comparisons)))
    ys = np.arange(len(comparisons))
    for i, c in enumerate(comparisons):
        d_mean = c["delta_mean"]
        ci_lo, ci_hi = c["delta_ci95"]
        p = c["p_value"]
        sig = (p < 0.05) if not np.isnan(p) else False
        color = "#2CA02C" if d_mean > 0 else "#C44E52"
        ax.errorbar([d_mean], [i], xerr=[[d_mean - ci_lo], [ci_hi - d_mean]],
                    fmt="o", color=color, ecolor=color,
                    markerfacecolor=color if sig else "white",
                    markeredgecolor=color, capsize=4, markersize=9, lw=1.6)
        ax.text(ci_hi + 0.005, i, f" p={p:.3g} (n={c['n_pairs']})",
                va="center", fontsize=9)
    ax.axvline(0.0, color="black", lw=0.7, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([c["variant"] for c in comparisons])
    ax.set_xlabel(f"Δ DSC vs baseline ({baseline})  —  filled marker = p < 0.05")
    ax.set_title(f"Paired Wilcoxon: {baseline} vs each augmented variant",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[saved] {out_png}")


def _bootstrap_ci(delta: np.ndarray, n_boot: int = 5000,
                  ci: float = 95.0, seed: int = 0
                  ) -> Tuple[float, float]:
    """Percentile bootstrap CI on the mean of a per-subject delta array."""
    rng = np.random.default_rng(seed)
    n = delta.size
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        boot_means[i] = delta[rng.integers(0, n, size=n)].mean()
    lo = float(np.percentile(boot_means, (100 - ci) / 2))
    hi = float(np.percentile(boot_means, 100 - (100 - ci) / 2))
    return lo, hi


def _run_one(metrics_glob, baseline_glob, variant_regex, seed_regex,
             baseline, mode, metric_key, precomputed,
             out_json: Path, out_png: Path) -> None:
    """The original single-target/single-metric flow, factored so the CLI
    can loop over combinations."""
    # --- Load per-subject arrays ---
    if precomputed:
        with precomputed.open() as f:
            raw = json.load(f)
    else:
        variants = _load_from_globs(
            metrics_glob, variant_regex, seed_regex,
            mode=mode, metric_key=metric_key, label="variants")
        base = _load_from_globs(
            baseline_glob, variant_regex, seed_regex,
            mode=mode, metric_key=metric_key, label="baseline",
            variant_name=baseline)
        raw = {**variants, **base}

    if baseline not in raw:
        print(f"[{out_json.stem}] baseline '{baseline}' not found; skipping")
        return

    pivoted = {v: _pivot_to_subject_lists(per_seed) for v, per_seed in raw.items()}
    summaries = {v: _summarise(sl) for v, sl in pivoted.items()}
    print(f"\n=== per-variant summary — metric={metric_key} ===")
    print(f"{'variant':<25} n_subj  n_seeds  mean±std")
    for v, s in summaries.items():
        if s["n_subjects"] == 0:
            print(f"{v:<25} 0")
        else:
            print(f"{v:<25} {s['n_subjects']:>6}  "
                  f"[{s['n_seeds_min']},{s['n_seeds_max']}]  "
                  f"{s['overall_mean_of_subj_means']:.4f} ± "
                  f"{s['overall_std_of_subj_means']:.4f}")

    baseline_arr = pivoted[baseline]
    comparisons: List[dict] = []
    for v, sl in pivoted.items():
        if v == baseline:
            continue
        stat, pval, n_pairs, common = _paired_wilcoxon(baseline_arr, sl)
        b_arr = np.array([np.mean(baseline_arr[k]) for k in common])
        v_arr = np.array([np.mean(sl[k]) for k in common])
        deltas = v_arr - b_arr
        d_mean = float(deltas.mean()) if deltas.size else float("nan")
        d_ci = _bootstrap_ci(deltas) if deltas.size >= 3 else (float("nan"), float("nan"))
        comparisons.append({
            "variant": v,
            "baseline": baseline,
            "n_pairs": n_pairs,
            "common_subjects": common,
            "wilcoxon_stat": stat,
            "p_value": pval,
            "delta_mean": d_mean,
            "delta_ci95": list(d_ci),
            "per_subject_delta": {s: float(d) for s, d in zip(common, deltas)},
        })

    print(f"\n=== paired Wilcoxon — metric={metric_key} ===")
    print(f"{'variant':<25} n  Δmean   CI95              p-value")
    for c in sorted(comparisons, key=lambda x: x["delta_mean"], reverse=True):
        ci = c["delta_ci95"]
        print(f"{c['variant']:<25} {c['n_pairs']:>2}  "
              f"{c['delta_mean']:+.4f}  "
              f"[{ci[0]:+.4f}, {ci[1]:+.4f}]  "
              f"{c['p_value']:.4g}")

    out = {
        "baseline": baseline,
        "mode": mode,
        "metric_key": metric_key,
        "summaries": summaries,
        "comparisons": comparisons,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {out_json}")

    _forest_plot(sorted(comparisons, key=lambda x: x["delta_mean"], reverse=True),
                 baseline=baseline, out_png=out_png)


def _expand_template(patterns: List[str], target: str) -> List[str]:
    """Replace `{target}` in each pattern. If a pattern has no placeholder,
    return it unchanged."""
    return [p.replace("{target}", target) for p in patterns]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    ap.add_argument("--metrics-glob", nargs="+", default=None,
                    help="Glob(s) matching augmented-variant metrics JSONs. "
                         "Use `{target}` as a placeholder that gets replaced "
                         "per target (e.g. 'runs/raov_aug_*/metrics_{target}.json').")
    ap.add_argument("--baseline-glob", nargs="+", default=None,
                    help="Glob(s) matching baseline metrics JSONs. Same "
                         "`{target}` placeholder support.")
    ap.add_argument("--variant-from-path", default=r"aug_(.+?)_seed",
                    help="Regex with one group capturing the variant name from the path")
    ap.add_argument("--seed-from-path", default=r"seed[_]?(\d+)",
                    help="Regex with one group capturing the seed from the path")
    ap.add_argument("--baseline", default="real_only",
                    help="Name to use for baseline group in outputs")
    ap.add_argument("--mode", default="raw",
                    help="Which per_subject.<mode> to read (raw | postprocessed | full | ...)")
    ap.add_argument("--metric-key", "--metric-keys", nargs="+", default=["dsc"],
                    dest="metric_keys",
                    help="One or more per-subject record fields to run. "
                         "Common: dsc, hd95_mm, sensitivity, precision, iou. "
                         "Default: dsc. If multiple, one JSON + PNG per metric.")
    ap.add_argument("--targets", nargs="+", default=["ov"],
                    help="Targets to loop over (typically ov, ut). Requires "
                         "`{target}` placeholder in --metrics-glob / --baseline-glob. "
                         "Default: ov (single).")
    ap.add_argument("--precomputed", type=Path, default=None,
                    help="Skip file loading; read a pre-flattened JSON.")
    ap.add_argument("--out-json", type=Path, required=True,
                    help="Output JSON path. `{target}` and `{metric}` "
                         "placeholders substituted per combination.")
    ap.add_argument("--out-png",  type=Path, required=True,
                    help="Output PNG path with same `{target}`/`{metric}` support.")
    args = ap.parse_args()

    if args.precomputed:
        if len(args.targets) > 1 or len(args.metric_keys) > 1:
            raise SystemExit("--precomputed supports only one target × one metric.")
        _run_one(args.metrics_glob, args.baseline_glob,
                 args.variant_from_path, args.seed_from_path,
                 args.baseline, args.mode, args.metric_keys[0], args.precomputed,
                 args.out_json, args.out_png)
        return

    if not args.metrics_glob or not args.baseline_glob:
        raise SystemExit("Supply --metrics-glob and --baseline-glob, "
                         "or use --precomputed.")

    for target in args.targets:
        mg = _expand_template(args.metrics_glob, target)
        bg = _expand_template(args.baseline_glob, target)
        for metric_key in args.metric_keys:
            oj = Path(str(args.out_json).replace("{target}", target).replace("{metric}", metric_key))
            op = Path(str(args.out_png).replace("{target}", target).replace("{metric}", metric_key))
            print(f"\n{'='*70}\n TARGET={target}  METRIC={metric_key}\n{'='*70}")
            _run_one(mg, bg, args.variant_from_path, args.seed_from_path,
                     args.baseline, args.mode, metric_key, None,
                     oj, op)

    # --- Load per-subject arrays ---
    if args.precomputed:
        with args.precomputed.open() as f:
            raw = json.load(f)
    else:
        if not args.metrics_glob or not args.baseline_glob:
            raise SystemExit("Supply --metrics-glob and --baseline-glob, "
                             "or use --precomputed.")
        variants = _load_from_globs(
            args.metrics_glob, args.variant_from_path, args.seed_from_path,
            mode=args.mode, metric_key=args.metric_key, label="variants")
        base = _load_from_globs(
            args.baseline_glob, args.variant_from_path, args.seed_from_path,
            mode=args.mode, metric_key=args.metric_key, label="baseline",
            variant_name=args.baseline)
        raw = {**variants, **base}

    if args.baseline not in raw:
        raise SystemExit(f"Baseline '{args.baseline}' not found. "
                         f"Available: {list(raw)}")

    # --- Pivot: {variant: {subject: [dsc_across_seeds]}} ---
    pivoted = {v: _pivot_to_subject_lists(per_seed) for v, per_seed in raw.items()}

    # --- Summaries ---
    summaries = {v: _summarise(sl) for v, sl in pivoted.items()}
    print("\n=== per-variant summary ===")
    print(f"{'variant':<25} n_subj  n_seeds  mean±std")
    for v, s in summaries.items():
        if s["n_subjects"] == 0:
            print(f"{v:<25} 0")
        else:
            print(f"{v:<25} {s['n_subjects']:>6}  "
                  f"[{s['n_seeds_min']},{s['n_seeds_max']}]  "
                  f"{s['overall_mean_of_subj_means']:.4f} ± "
                  f"{s['overall_std_of_subj_means']:.4f}")

    # --- Paired Wilcoxon per variant vs baseline ---
    baseline_arr = pivoted[args.baseline]
    comparisons: List[dict] = []
    for v, sl in pivoted.items():
        if v == args.baseline:
            continue
        stat, pval, n_pairs, common = _paired_wilcoxon(baseline_arr, sl)
        # Per-subject deltas for CI
        b_arr = np.array([np.mean(baseline_arr[k]) for k in common])
        v_arr = np.array([np.mean(sl[k]) for k in common])
        deltas = v_arr - b_arr
        d_mean = float(deltas.mean()) if deltas.size else float("nan")
        d_ci = _bootstrap_ci(deltas) if deltas.size >= 3 else (float("nan"), float("nan"))
        comparisons.append({
            "variant": v,
            "baseline": args.baseline,
            "n_pairs": n_pairs,
            "common_subjects": common,
            "wilcoxon_stat": stat,
            "p_value": pval,
            "delta_mean": d_mean,
            "delta_ci95": list(d_ci),
            "per_subject_delta": {s: float(d) for s, d in zip(common, deltas)},
        })

    print("\n=== paired Wilcoxon (variant vs baseline) ===")
    print(f"{'variant':<25} n  Δmean   CI95              p-value")
    for c in sorted(comparisons, key=lambda x: x["delta_mean"], reverse=True):
        ci = c["delta_ci95"]
        print(f"{c['variant']:<25} {c['n_pairs']:>2}  "
              f"{c['delta_mean']:+.4f}  "
              f"[{ci[0]:+.4f}, {ci[1]:+.4f}]  "
              f"{c['p_value']:.4g}")

    # --- Save ---
    out = {
        "baseline": args.baseline,
        "mode": args.mode,
        "metric_key": args.metric_key,
        "summaries": summaries,
        "comparisons": comparisons,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.out_json}")

    _forest_plot(sorted(comparisons, key=lambda x: x["delta_mean"], reverse=True),
                 baseline=args.baseline, out_png=args.out_png)


if __name__ == "__main__":
    main()
