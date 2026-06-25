"""
Aggregate per-sample explain metrics + quality metrics into a master CSV
comparing all variants (1a, 1b, 1c_concat, 1c_spade).

Reads `sample_NN_metrics.json` from each variant's `explain/` directory,
optionally combines with `quality.json` from `quality_metrics.py` output,
and writes a single CSV with one row per variant.

Usage:
    python -m src.Generator.aggregate_metrics \\
        --explain_dirs runs/exp1a/explain runs/exp1b/explain \\
                       runs/exp1c_concat/explain runs/exp1c_spade/explain \\
        --quality_jsons runs/exp1a/quality.json runs/exp1b/quality.json \\
                        runs/exp1c_concat/quality.json runs/exp1c_spade/quality.json \\
        --out master_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ORGAN_KEYS = ["uterus", "ov_L", "ov_R", "em"]
ALL_LABEL_KEYS = ["outside_body", "uterus", "ov_L", "ov_R", "em", "body_other"]


def _nanmean(values: list[float]) -> float:
    valid = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(sum(valid) / len(valid)) if valid else float("nan")


def aggregate_explain_dir(explain_dir: Path) -> dict:
    """Read all sample_NN_metrics.json files in a directory and return
    mean values across samples for each metric."""
    jsons = sorted(explain_dir.glob("sample_*_metrics.json"))
    if not jsons:
        return {"_note": f"no JSON files found in {explain_dir}"}

    per_sample = []
    for jp in jsons:
        with open(jp) as f:
            per_sample.append(json.load(f))

    variant = per_sample[0].get("meta", {}).get("variant", explain_dir.parent.name)
    n_samples = len(per_sample)

    out: dict = {
        "variant": variant,
        "n_samples": n_samples,
        "ckpt_step": per_sample[0].get("meta", {}).get("ckpt_step"),
        "guidance_scale": per_sample[0].get("meta", {}).get("guidance_scale"),
        "num_inference_steps": per_sample[0].get("meta", {}).get("num_inference_steps"),
    }

    # CLR per organ
    for organ in ORGAN_KEYS:
        vals = [s.get("CLR_per_channel", {}).get(organ) for s in per_sample]
        out[f"CLR_{organ}"] = _nanmean(vals)

    # AILM per label channel
    for k in ALL_LABEL_KEYS:
        vals = [s.get("AILM_per_channel", {}).get(k) for s in per_sample]
        out[f"AILM_{k}"] = _nanmean(vals)

    # Attribution sparsity per channel (mean)
    sp_vals = []
    for s in per_sample:
        sps = s.get("attribution_sparsity_per_channel", {})
        sp_vals.extend([v for v in sps.values() if v is not None])
    out["sparsity_mean"] = _nanmean(sp_vals)

    # SPADE OSI (1b/1c_spade only)
    osi_max_organ_vals = [
        s.get("OSI_summary", {}).get("mean_max_organ_corr") for s in per_sample
    ]
    osi_body_vals = [
        s.get("OSI_summary", {}).get("mean_body_corr") for s in per_sample
    ]
    out["OSI_max_organ_corr"] = _nanmean(osi_max_organ_vals)
    out["OSI_body_corr"] = _nanmean(osi_body_vals)

    return out


def load_quality_json(path: Path) -> dict:
    with open(path) as f:
        q = json.load(f)
    flat = {
        "fid": q.get("fid"),
        "hist_kl": q.get("hist_kl"),
    }
    lpips_nn = q.get("lpips_nn") or {}
    flat["lpips_nn_min"] = lpips_nn.get("min")
    flat["lpips_nn_mean"] = lpips_nn.get("mean")
    flat["lpips_nn_max"] = lpips_nn.get("max")
    return flat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--explain_dirs", nargs="+", required=True,
                        help="One or more explain/ directories containing sample_NN_metrics.json")
    parser.add_argument("--quality_jsons", nargs="*", default=[],
                        help="Optional: matching quality.json files (must align with --explain_dirs order)")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    rows: list[dict] = []
    for i, ed in enumerate(args.explain_dirs):
        row = aggregate_explain_dir(Path(ed))
        if i < len(args.quality_jsons):
            qpath = Path(args.quality_jsons[i])
            if qpath.exists():
                row.update(load_quality_json(qpath))
            else:
                print(f"[warn] quality JSON not found: {qpath}")
        rows.append(row)

    all_keys: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)
    # Put 'variant' first
    if "variant" in all_keys:
        all_keys.remove("variant")
        all_keys = ["variant"] + all_keys

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[aggregate] wrote {out_path}")

    print("\n=== Summary ===")
    print(f"{'variant':<18} {'CLR_ut':>8} {'CLR_em':>8} {'AILM_ut':>9} "
          f"{'OSI_org':>9} {'OSI_body':>9} {'FID':>7} {'LPIPSm':>8}")
    for r in rows:
        def fmt(v):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "    n/a"
            return f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"
        print(f"{str(r.get('variant', '?')):<18} "
              f"{fmt(r.get('CLR_uterus')):>8} {fmt(r.get('CLR_em')):>8} "
              f"{fmt(r.get('AILM_uterus')):>9} "
              f"{fmt(r.get('OSI_max_organ_corr')):>9} {fmt(r.get('OSI_body_corr')):>9} "
              f"{fmt(r.get('fid')):>7} {fmt(r.get('lpips_nn_mean')):>8}")


if __name__ == "__main__":
    main()
