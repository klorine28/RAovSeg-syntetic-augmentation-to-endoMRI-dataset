#!/usr/bin/env python3
"""
Append post-fix variant rows to metrics/master_metrics.csv.

Pulls what we have from:
  - mechanism CSV → intensity distribution stats
  - explain.py output → CLR per channel
  - DSC metrics_ov.json → ovary DSC (informational, not part of master CSV columns
    but useful in a comment column)

Metrics not readily available (FID, LPIPS, hist_KL) are written as `nan`.
Update those later by re-running the quality-metrics computation.

Usage:
    python -m src.RaovSeg_recreation.append_fixed_metrics \\
        --master-csv metrics/master_metrics.csv \\
        --mechanism-csv figures_fixed/mechanism/mech_ovary_intensity_table.csv \\
        --explain-root . \\
        --variants exp1c_concat exp1c_spade exp2 exp2_lam05 exp2_lam50
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


NAN = "nan"

# Fixed-variant defaults (match training/sampling settings)
VARIANT_META = {
    "exp1c_concat":  {"guidance_scale": "3.0", "num_inference_steps": "100"},
    "exp1c_spade":   {"guidance_scale": "2.0", "num_inference_steps": "100"},
    "exp2":          {"guidance_scale": "2.0", "num_inference_steps": "100"},
    "exp2_lam05":    {"guidance_scale": "2.0", "num_inference_steps": "100"},
    "exp2_lam50":    {"guidance_scale": "2.0", "num_inference_steps": "100"},
}


def load_clr(explain_root: Path, variant: str) -> dict:
    """Read CLR_per_channel from explain.py's sample_00_metrics.json."""
    if variant == "exp1c_concat":
        p = explain_root / "1c/concat/explain/sample_00_metrics.json"
    elif variant == "exp1c_spade":
        p = explain_root / "1c/spade/explain/sample_00_metrics.json"
    else:
        p = explain_root / f"phase2/{variant}/explain/sample_00_metrics.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return d.get("CLR_per_channel", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-csv", type=Path, required=True)
    ap.add_argument("--mechanism-csv", type=Path, required=True)
    ap.add_argument("--explain-root", type=Path, default=Path("."))
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--n-samples", type=int, default=32,
                    help="synth subjects per variant (default 32)")
    ap.add_argument("--ckpt-step", type=int, default=100000)
    args = ap.parse_args()

    # Read the existing CSV to preserve its column order + drop stale fixed rows
    rows = []
    fieldnames = None
    with args.master_csv.open() as f:
        rdr = csv.DictReader(f)
        fieldnames = rdr.fieldnames
        for r in rdr:
            # Skip any prior fixed-variant rows so this is idempotent
            if not r["variant"].endswith("_fixed"):
                rows.append(r)

    # Build fixed-variant rows
    for v in args.variants:
        row = {k: NAN for k in fieldnames}
        row["variant"] = f"{v}_fixed"
        row["n_samples"] = str(args.n_samples)
        row["ckpt_step"] = str(args.ckpt_step)
        meta = VARIANT_META.get(v, {})
        row["guidance_scale"] = meta.get("guidance_scale", NAN)
        row["num_inference_steps"] = meta.get("num_inference_steps", NAN)

        # CLR from explain
        clr = load_clr(args.explain_root, v)
        # Map explain's channel names to the CSV columns
        col_map = {
            "uterus":   "CLR_uterus",
            "ov_L":     "CLR_ov_L",
            "ov_R":     "CLR_ov_R",
            "em":       "CLR_em",
        }
        for ch, col in col_map.items():
            if ch in clr and col in row:
                row[col] = f"{float(clr[ch]):.6f}"

        rows.append(row)

    # Write back with a backup
    bak = args.master_csv.with_suffix(".csv.bak_before_fixed_append")
    args.master_csv.rename(bak)
    with args.master_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[saved] {args.master_csv} ({len(rows)} rows, backup at {bak})")
    print(f"        appended {len(args.variants)} fixed-variant rows")
    print(f"        (FID / hist_kl / LPIPS / AILM columns are nan — compute later)")


if __name__ == "__main__":
    main()
