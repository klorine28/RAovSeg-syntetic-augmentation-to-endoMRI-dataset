"""
Postprocessing closing-iterations sweep for RAovSeg evaluation.

Reuses the trained ResClass and AttUSeg models — no retraining. Iterates over
a list of closing_iterations values, runs the full evaluation each time, and
writes per-subject DSC + per-config summary to a CSV.

Output: data/predictions/sweep_postproc_results.csv

CSV columns:
  closing_iterations, subject_id, dsc_full, dsc_no_pp, dsc_no_rc

Plus a summary block at the end with mean ± std per config.
"""

import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import torch

# Reuse the loaders and predict function from evaluate.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load_resclass, load_attuseg, DEVICE

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "RAovSeg"))
from RAovSeg_tools import postprocess_, dsc_cal_np


CLOSING_ITERATIONS_GRID = [0, 1, 3, 5, 10]


@torch.no_grad()
def predict_subject_cached(image, label, resclass, attuseg, closing_iters: int):
    """Run pipeline once per subject for a single closing_iters value.

    Returns dsc_full, dsc_no_pp, dsc_no_rc.
    """
    n_slices = image.shape[0]

    # Cache classifier scores and segmentation outputs once per subject so we
    # don't pay 3x the cost per closing_iters value (segmentation is the same;
    # only the postproc step changes).
    cls_keep = np.zeros(n_slices, dtype=bool)
    seg_outputs = np.zeros_like(label, dtype=np.float32)

    for s in range(n_slices):
        img_slice = torch.from_numpy(image[s][np.newaxis, np.newaxis, ...]).float().to(DEVICE)
        cls_keep[s] = torch.sigmoid(resclass(img_slice).squeeze()).item() > 0.5
        seg_outputs[s] = (torch.sigmoid(attuseg(img_slice)) > 0.5).cpu().numpy().squeeze()

    # Configuration A: full pipeline (resclass gate + postproc)
    pred_full = np.where(cls_keep[:, None, None], seg_outputs, 0).astype(np.float32)
    if closing_iters > 0:
        pred_full = postprocess_(pred_full, closing_iterations=closing_iters).astype(np.float32)
    dsc_full = dsc_cal_np(pred_full, label)

    # Configuration B: no postproc (resclass gate, no closing)
    pred_no_pp = np.where(cls_keep[:, None, None], seg_outputs, 0).astype(np.float32)
    dsc_no_pp = dsc_cal_np(pred_no_pp, label)

    # Configuration C: no resclass (segment all slices, with postproc)
    pred_no_rc = seg_outputs.astype(np.float32)
    if closing_iters > 0:
        pred_no_rc = postprocess_(pred_no_rc, closing_iterations=closing_iters).astype(np.float32)
    dsc_no_rc = dsc_cal_np(pred_no_rc, label)

    return dsc_full, dsc_no_pp, dsc_no_rc


def main():
    parser = argparse.ArgumentParser(description="Sweep closing_iterations for RAovSeg")
    parser.add_argument("--test-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data" / "processed" / "test")
    parser.add_argument("--models-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "models")
    parser.add_argument("--output-csv", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data" / "predictions" / "sweep_postproc_results.csv")
    parser.add_argument("--grid", type=int, nargs="+", default=CLOSING_ITERATIONS_GRID,
                        help="closing_iterations values to sweep")
    args = parser.parse_args()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load models once
    resclass = load_resclass(args.models_dir / "resclass_best.pth")
    attuseg = load_attuseg(args.models_dir / "attuseg_best.pth")
    print(f"Models loaded. Device: {DEVICE}")
    print(f"Sweeping closing_iterations = {args.grid}\n")

    # Collect all (closing_iters, subject_id) results
    rows = []
    summary = {}  # closing_iters -> dict of mean/std for each ablation

    test_subjects = sorted([d for d in args.test_dir.iterdir() if d.is_dir()])

    for closing_iters in args.grid:
        print(f"=== closing_iterations = {closing_iters} ===")
        full_list, no_pp_list, no_rc_list = [], [], []

        for subj_dir in test_subjects:
            img_path = subj_dir / "image.npy"
            lbl_path = subj_dir / "ov_label.npy"
            if not (img_path.exists() and lbl_path.exists()):
                continue

            image = np.load(img_path)
            label = np.load(lbl_path)
            dsc_full, dsc_no_pp, dsc_no_rc = predict_subject_cached(
                image, label, resclass, attuseg, closing_iters
            )

            rows.append({
                "closing_iterations": closing_iters,
                "subject_id": subj_dir.name,
                "dsc_full": f"{dsc_full:.4f}",
                "dsc_no_pp": f"{dsc_no_pp:.4f}",
                "dsc_no_rc": f"{dsc_no_rc:.4f}",
            })
            full_list.append(dsc_full)
            no_pp_list.append(dsc_no_pp)
            no_rc_list.append(dsc_no_rc)
            print(f"  {subj_dir.name}: full={dsc_full:.4f} | no_pp={dsc_no_pp:.4f} | no_rc={dsc_no_rc:.4f}")

        summary[closing_iters] = {
            "full_mean": np.mean(full_list), "full_std": np.std(full_list),
            "no_pp_mean": np.mean(no_pp_list), "no_pp_std": np.std(no_pp_list),
            "no_rc_mean": np.mean(no_rc_list), "no_rc_std": np.std(no_rc_list),
            "n": len(full_list),
        }
        print(f"  -> mean DSC: full={summary[closing_iters]['full_mean']:.4f} "
              f"± {summary[closing_iters]['full_std']:.4f} (n={summary[closing_iters]['n']})\n")

    # Write per-subject CSV
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["closing_iterations", "subject_id",
                                                "dsc_full", "dsc_no_pp", "dsc_no_rc"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-subject results written: {args.output_csv}")

    # Write summary CSV alongside
    summary_csv = args.output_csv.with_name(args.output_csv.stem + "_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["closing_iterations", "n",
                         "full_mean", "full_std",
                         "no_pp_mean", "no_pp_std",
                         "no_rc_mean", "no_rc_std"])
        for ci, s in summary.items():
            writer.writerow([ci, s["n"],
                             f"{s['full_mean']:.4f}", f"{s['full_std']:.4f}",
                             f"{s['no_pp_mean']:.4f}", f"{s['no_pp_std']:.4f}",
                             f"{s['no_rc_mean']:.4f}", f"{s['no_rc_std']:.4f}"])
    print(f"Summary written: {summary_csv}")

    # Pretty-print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'closing_iters':<15} {'full DSC':<20} {'no_pp DSC':<20} {'no_rc DSC':<20}")
    for ci in args.grid:
        s = summary[ci]
        print(f"{ci:<15} "
              f"{s['full_mean']:.4f} ± {s['full_std']:.4f}  "
              f"{s['no_pp_mean']:.4f} ± {s['no_pp_std']:.4f}  "
              f"{s['no_rc_mean']:.4f} ± {s['no_rc_std']:.4f}")
    print(f"\nPaper benchmarks: full=0.290, no_postprocess=0.235, no_resclass=0.013")


if __name__ == "__main__":
    main()