"""DATA-3: detection vs delineation per (variant, seed, subject).

Emits metrics/data3_detection_vs_delineation.csv. Handles BOTH the older
predictions/ dir (used by augmented raovseg_aug_* runs) and the newer
predictions_ov/ dir (used by raovseg_real_only_*).
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    runs = Path("/mnt/parscratch/users/ijp25lg/synth_mri/runs")
    rows = []

    for run_dir in sorted(runs.glob("raov*_seed*")):
        m = re.match(r"(?P<variant>.+?)_seed(?P<seed>\d+)$", run_dir.name)
        if not m:
            continue
        variant = m.group("variant")
        seed = m.group("seed")

        pred_dir = None
        for candidate in ("predictions_ov", "predictions"):
            cand_dir = run_dir / candidate
            if cand_dir.exists() and any(cand_dir.glob("*_pred.npy")):
                pred_dir = cand_dir
                break
        if pred_dir is None:
            continue

        test_dir = run_dir / "processed" / "test"
        if not test_dir.exists():
            continue

        for pred_path in sorted(pred_dir.glob("*_pred.npy")):
            subj = pred_path.stem.replace("_pred", "")
            gt_path = test_dir / subj / "ov_label.npy"
            if not gt_path.exists():
                continue
            pred = np.load(pred_path).astype(bool)
            gt = np.load(gt_path).astype(bool)
            rows.append({
                "variant": variant,
                "seed": seed,
                "subject_id": subj,
                "gt_present": int(gt.any()),
                "detected": int(pred.any()),
                "n_pred_voxels": int(pred.sum()),
                "n_gt_voxels": int(gt.sum()),
                "pred_dir_name": pred_dir.name,
            })

    if not rows:
        print("no rows to write")
        return

    out = Path("metrics/data3_detection_vs_delineation.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[saved] {out} ({len(rows)} rows)\n")

    det = defaultdict(list)
    for r in rows:
        if r["gt_present"]:
            det[r["variant"]].append(r["detected"])
    print("Detection rate (of gt-present pairs):")
    for v, lst in sorted(det.items()):
        rate = 100 * sum(lst) / len(lst) if lst else 0.0
        print(f"  {v:<40}  {sum(lst)}/{len(lst)}  ({rate:.1f}%)")


if __name__ == "__main__":
    main()
