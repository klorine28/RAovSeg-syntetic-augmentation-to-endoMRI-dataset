#!/usr/bin/env python3
"""DATA-4 — ovary intensity in-window fraction (v14 corrected version).

Measures the fraction of ovary voxels that land in RAovSeg's enhancement
window [O1, O2] = [0.22, 0.30] AFTER the RAovSeg preprocess pipeline's
steps 1-3 (resample → percentile clip → minmax to [0, 1]) but BEFORE
the o1/o2 enhancement itself. This matches the domain in which Table
4.10's cited 10.6% real-D2 value was computed.

Acceptance test (domain check): measures the exact 3-subject pool
mechanism_figures.py used (D2-016, D2-017, D2-024) and asserts the
result matches 10.06% ± 1.0% — the value recorded in
figures/mech_ovary_intensity_table.csv. If that check fails, the script
exits non-zero and writes no output.

Two real reference rows are written to the CSV:
    real_d2        (30-subject D2 training pool — the corrected reference)
    real_d2_mech3  (3-subject pool D2-016/017/024 — reproduces 10.06%)

Synth rows should be compared against the 30-subject value. The 3-subject
row is retained for continuity with the historical mech table.

Both pre_rescale and post_rescale rows are produced by the same
`measure_pool()` function so the pre-vs-post comparison is internally
consistent.

Usage:
    python -m src.analysis.data4_in_window \\
        --d2-raw-dir  UT-EndoMRI/D2_TCPW \\
        --synth-root  /mnt/parscratch/users/$USER/synth_mri/assembled \\
        --variants    exp1c_concat exp1c_spade exp1c_concat_fixed \\
                      exp1c_spade_fixed exp2_fixed exp2_lam05_fixed \\
                      exp2_lam50_fixed \\
        --out-csv     metrics/data4_in_window_fraction.csv

Layout auto-detection per variant:
  (a) Staged  — <variant>/pre_rescale/D2-XXX/D2-XXX_T2FS.nii.gz + _ov.nii.gz
                <variant>/post_rescale/D2-XXX/...
                → emits one row per existing stage.
  (b) Flat    — <variant>/D2-XXX/D2-XXX_T2FS.nii.gz + _ov.nii.gz
                → emits one post_rescale row (assembly-time Path B is the
                  observed state; the raw pre-rescale volumes were not
                  retained on disk).

Both layouts go through the same measure_pool(), so real vs synth rows are
strictly comparable regardless of which layout a variant used.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "RaovSeg_recreation"))

from preprocess import (  # noqa: E402
    O1, O2, SUBJECT_RE,
    preprocess_image, preprocess_label,
    scan_and_classify, find_best_sequence,
)


# --- one-shot measurement ------------------------------------------------

def measure_subject(subject_dir: Path, subject_id: str) -> tuple[float, float, int] | None:
    """Load one subject through preprocess steps 1-3 and return
    (ovary_mean, ovary_sd, n_ovary_voxels). Returns None if the subject is
    missing its sequence or ovary label.
    """
    img_path = find_best_sequence(subject_dir, subject_id)
    if img_path is None:
        return None
    ov_path = subject_dir / f"{subject_id}_ov.nii.gz"
    if not ov_path.exists():
        return None

    img = preprocess_image(img_path, skip_enhancement=True)  # [0, 1] domain
    lab = preprocess_label(ov_path).astype(bool)
    if not lab.any():
        return None
    # Both should have the same shape post-resample; check defensively
    if img.shape != lab.shape:
        print(f"[warn] shape mismatch {subject_id}: img {img.shape} vs lab {lab.shape}; skipping")
        return None

    vals = img[lab]
    return float(vals.mean()), float(vals.std(ddof=0)), int(vals.size)


def measure_pool(
    pool_dir: Path,
    subject_ids: list[str],
    label: str,
) -> tuple[float, float, int, float] | None:
    """Measure the in-window fraction across a pool of subjects. Returns
    (mean_of_means, sd_of_means, n_volumes, in_window_pct) or None if no
    subject could be loaded.

    - mean_of_means / sd_of_means: mean and SD of per-subject ovary means
    - in_window_pct: fraction of ALL pooled ovary voxels in [O1, O2] × 100
    """
    per_subj_means = []
    all_voxels = []
    n_loaded = 0
    n_missing = 0

    for sid in subject_ids:
        subj_dir = pool_dir / sid
        if not subj_dir.is_dir():
            n_missing += 1
            continue
        img_path = find_best_sequence(subj_dir, sid)
        ov_path = subj_dir / f"{sid}_ov.nii.gz"
        if img_path is None or not ov_path.exists():
            n_missing += 1
            continue

        img = preprocess_image(img_path, skip_enhancement=True)
        lab = preprocess_label(ov_path).astype(bool)
        if img.shape != lab.shape or not lab.any():
            n_missing += 1
            continue

        vals = img[lab]
        per_subj_means.append(float(vals.mean()))
        all_voxels.append(vals)
        n_loaded += 1

    if not all_voxels:
        print(f"[{label}] no subjects loaded ({n_missing} missing)")
        return None

    pooled = np.concatenate(all_voxels)
    in_window_pct = 100.0 * float(((pooled >= O1) & (pooled <= O2)).mean())
    mean_of_means = float(np.mean(per_subj_means))
    sd_of_means = float(np.std(per_subj_means, ddof=1)) if len(per_subj_means) > 1 else 0.0

    print(f"[{label}] n_loaded={n_loaded}, n_missing={n_missing}, "
          f"mean_of_means={mean_of_means:.4f}, sd={sd_of_means:.4f}, "
          f"in_window_pct={in_window_pct:.2f}% "
          f"(pooled n_voxels={pooled.size:,})")
    return mean_of_means, sd_of_means, n_loaded, in_window_pct


# --- subject discovery ---------------------------------------------------

def d2_training_pool(d2_raw_dir: Path) -> list[str]:
    """Return the deterministic 30-subject D2 training pool per
    preprocess.scan_and_classify (SPLIT_SEED=42, N_TRAIN_VAL=30).
    """
    rows = scan_and_classify(d2_raw_dir)
    train_val = sorted([r["subject_id"] for r in rows if r["split"] == "train_val"])
    if len(train_val) != 30:
        raise SystemExit(
            f"[FATAL] D2 training pool size = {len(train_val)}, expected 30. "
            f"Check UT-EndoMRI/D2_TCPW layout and SUBJECT_RE."
        )
    return train_val


# --- acceptance test -----------------------------------------------------

# The historical mech_ovary_intensity_table.csv reports 10.06% pooled across
# subjects D2-016, D2-017, D2-024. This is the domain check.
MECH3_SUBJECTS = ["D2-016", "D2-017", "D2-024"]
MECH3_TARGET_PCT = 10.06
MECH3_TOLERANCE_PCT = 1.0


def acceptance_test(d2_raw_dir: Path) -> tuple[float, float, int, float]:
    """Domain check — reproduce the 10.06% figure from
    figures/mech_ovary_intensity_table.csv using the exact same 3-subject
    pool (D2-016, D2-017, D2-024). Exits non-zero if not within tolerance.
    """
    print(f"\n=== Acceptance test: mechanism 3-subject pool "
          f"({', '.join(MECH3_SUBJECTS)}) ===")
    result = measure_pool(d2_raw_dir, MECH3_SUBJECTS, label="real_d2_mech3")
    if result is None:
        raise SystemExit("[FAIL] mech-3 pool could not be measured")

    _, _, _, in_window_pct = result
    diff = abs(in_window_pct - MECH3_TARGET_PCT)
    if diff > MECH3_TOLERANCE_PCT:
        raise SystemExit(
            f"[FAIL] mech-3 in-window = {in_window_pct:.2f}% but target = "
            f"{MECH3_TARGET_PCT}% ± {MECH3_TOLERANCE_PCT}%. "
            f"Measurement domain has drifted from the historical reference; "
            f"no synth row from this run is trustworthy. Not writing CSV."
        )
    print(f"[OK] mech-3 = {in_window_pct:.2f}%, within {MECH3_TOLERANCE_PCT}% of "
          f"target {MECH3_TARGET_PCT}%")
    return result


# --- main ---------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d2-raw-dir", type=Path, required=True,
                    help="Path to UT-EndoMRI/D2_TCPW directory (raw D2 subjects)")
    ap.add_argument("--synth-root", type=Path, required=True,
                    help="Path to root dir containing per-variant assembled synth volumes. "
                         "Each variant dir must contain pre_rescale/ and post_rescale/ subdirs, "
                         "each holding D2-XXX subject subdirs in the raw-input layout.")
    ap.add_argument("--variants", nargs="+", required=True,
                    help="Variant names (subdirs of --synth-root) to measure.")
    ap.add_argument("--out-csv", type=Path, required=True,
                    help="Output CSV path.")
    ap.add_argument("--skip-acceptance", action="store_true",
                    help="Skip the real-D2 acceptance test (debugging only; NOT for v14 use).")
    args = ap.parse_args()

    print(f"Window: [{O1}, {O2}]  (RAovSeg enhancement window)")
    print(f"Domain: preprocess_image(skip_enhancement=True)  → percentile clip + minmax to [0, 1]")

    # 1) Discover 30-subject training pool
    d2_ids = d2_training_pool(args.d2_raw_dir)
    print(f"D2 training pool: {len(d2_ids)} subjects (first 5: {d2_ids[:5]})")

    # 2) Acceptance test: reproduce the historical 10.06% figure using the
    #    exact 3-subject pool it was computed on.
    if not args.skip_acceptance:
        mech3_result = acceptance_test(args.d2_raw_dir)
    else:
        print("[warn] --skip-acceptance set; domain sanity check bypassed")
        mech3_result = measure_pool(args.d2_raw_dir, MECH3_SUBJECTS, label="real_d2_mech3")

    # 3) Measure the corrected 30-subject reference (the value synth rows compare against)
    print(f"\n=== Corrected reference: 30-subject D2 training pool ===")
    real30_result = measure_pool(args.d2_raw_dir, d2_ids, label="real_d2")

    # 4) Assemble real rows
    rows = []
    if real30_result is not None:
        mm, sd, n_vol, iw = real30_result
        rows.append({
            "variant": "real_d2", "stage": "-", "n_volumes": n_vol,
            "ovary_mean": mm, "ovary_sd": sd, "in_window_pct": iw,
        })
    if mech3_result is not None:
        mm, sd, n_vol, iw = mech3_result
        rows.append({
            "variant": "real_d2_mech3", "stage": "-", "n_volumes": n_vol,
            "ovary_mean": mm, "ovary_sd": sd, "in_window_pct": iw,
        })

    for variant in args.variants:
        variant_dir = args.synth_root / variant
        if not variant_dir.is_dir():
            print(f"[skip] {variant}: dir not found ({variant_dir})")
            continue

        # Auto-detect layout: staged (has pre_rescale/post_rescale subdirs)
        # or flat (has D2-XXX subject dirs directly).
        staged = [s for s in ("pre_rescale", "post_rescale")
                  if (variant_dir / s).is_dir()]
        flat_subjs = sorted(p.name for p in variant_dir.iterdir()
                            if p.is_dir() and SUBJECT_RE.match(p.name))

        if staged:
            stages_to_measure = [(s, variant_dir / s) for s in staged]
        elif flat_subjs:
            # Flat layout: treat as post_rescale (assembly-time Path B is the
            # observed state; pre-rescale volumes were not retained).
            stages_to_measure = [("post_rescale", variant_dir)]
        else:
            print(f"[skip] {variant}: neither staged nor flat layout found")
            continue

        for stage, pool_dir in stages_to_measure:
            subj_ids = sorted(
                p.name for p in pool_dir.iterdir()
                if p.is_dir() and SUBJECT_RE.match(p.name)
            )
            if not subj_ids:
                print(f"[skip] {variant}/{stage}: no D2-XXX subject dirs found")
                continue
            result = measure_pool(pool_dir, subj_ids, label=f"{variant}/{stage}")
            if result is None:
                continue
            mm, sd, n_vol, iw = result
            rows.append({
                "variant": variant, "stage": stage, "n_volumes": n_vol,
                "ovary_mean": mm, "ovary_sd": sd, "in_window_pct": iw,
            })

    # 4) Write CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "stage", "n_volumes",
                                          "ovary_mean", "ovary_sd", "in_window_pct"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[saved] {args.out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
