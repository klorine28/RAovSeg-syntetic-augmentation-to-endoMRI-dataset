"""
Priority 1: ResClass threshold sweep, selected on validation set.

Paper says only "BCEWithLogitsLoss" — does not specify the binary decision
threshold for ResClass at inference. We sweep thresholds on the validation
set, pick the best, then run final test evaluation with that fixed threshold.

This is methodologically clean: threshold is a free parameter the paper
does not specify; we tune it on validation data, never test.

Output:
  data/predictions/threshold_sweep_val.csv      (per-subject val results)
  data/predictions/threshold_sweep_summary.csv  (mean DSC per threshold)
  data/predictions/threshold_sweep_test.csv     (final test eval at best threshold)
"""

import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import torch

# Reuse the loaders and split logic from training scripts
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load_resclass, load_attuseg, DEVICE
from train_resclass import split_subjects as split_subjects_resclass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "RAovSeg"))
from RAovSeg_tools import postprocess_, dsc_cal_np


THRESHOLD_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
TRAIN_RATIO_RESCLASS = 0.6  # must match train_resclass.TRAIN_RATIO
SPLIT_SEED = 42             # must match train_resclass.split_subjects seed


@torch.no_grad()
def predict_subject_with_threshold(image, label, resclass, attuseg,
                                   threshold: float, closing_iters: int = 10):
    """Run pipeline once per subject for a single ResClass threshold.

    Caches segmentation outputs so we don't re-run AttUSeg for every threshold.
    Returns dsc with full pipeline (resclass at threshold + postproc).
    """
    n_slices = image.shape[0]

    # Cache classifier scores (sigmoid) and segmentation outputs once
    cls_scores = np.zeros(n_slices, dtype=np.float32)
    seg_outputs = np.zeros_like(label, dtype=np.float32)

    for s in range(n_slices):
        img_slice = torch.from_numpy(image[s][np.newaxis, np.newaxis, ...]).float().to(DEVICE)
        cls_scores[s] = torch.sigmoid(resclass(img_slice).squeeze()).item()
        seg_outputs[s] = (torch.sigmoid(attuseg(img_slice)) > 0.5).cpu().numpy().squeeze()

    # Apply threshold
    cls_keep = cls_scores > threshold
    prediction = np.where(cls_keep[:, None, None], seg_outputs, 0).astype(np.float32)

    if closing_iters > 0:
        prediction = postprocess_(prediction, closing_iterations=closing_iters).astype(np.float32)

    return dsc_cal_np(prediction, label), cls_scores, seg_outputs


def evaluate_at_threshold(subjects_dir, subject_ids, resclass, attuseg,
                          threshold, closing_iters):
    """Evaluate one threshold value across a list of subjects. Returns list of DSCs."""
    dscs = []
    for sid in subject_ids:
        subj_dir = subjects_dir / sid
        img_path = subj_dir / "image.npy"
        lbl_path = subj_dir / "ov_label.npy"
        if not (img_path.exists() and lbl_path.exists()):
            continue
        image = np.load(img_path)
        label = np.load(lbl_path)
        dsc, _, _ = predict_subject_with_threshold(
            image, label, resclass, attuseg, threshold, closing_iters
        )
        dscs.append((sid, dsc))
    return dscs


def main():
    parser = argparse.ArgumentParser(description="ResClass threshold sweep on validation set")
    parser.add_argument("--processed-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "data" / "processed")
    parser.add_argument("--models-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "models")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "data" / "predictions")
    parser.add_argument("--closing-iters", type=int, default=10,
                        help="Postprocessing closing iterations (kept fixed across sweep)")
    parser.add_argument("--grid", type=float, nargs="+", default=THRESHOLD_GRID,
                        help="ResClass thresholds to sweep")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Reconstruct the train/val split that train_resclass used
    train_val_dir = args.processed_dir / "train_val"
    train_subjects, val_subjects = split_subjects_resclass(
        train_val_dir, TRAIN_RATIO_RESCLASS, seed=SPLIT_SEED
    )
    test_dir = args.processed_dir / "test"
    test_subjects = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])

    print(f"Validation subjects ({len(val_subjects)}): {val_subjects}")
    print(f"Test subjects ({len(test_subjects)}): {test_subjects}")
    print(f"Closing iterations (fixed): {args.closing_iters}")
    print(f"Threshold grid: {args.grid}\n")

    # Load models once
    resclass = load_resclass(args.models_dir / "resclass_best.pth")
    attuseg = load_attuseg(args.models_dir / "attuseg_best.pth")
    print(f"Models loaded. Device: {DEVICE}\n")

    # ===== VALIDATION SWEEP =====
    print("=" * 60)
    print("VALIDATION SWEEP — selecting threshold")
    print("=" * 60)
    val_rows = []
    val_summary = {}
    for thr in args.grid:
        print(f"\nThreshold = {thr:.2f}")
        results = evaluate_at_threshold(
            train_val_dir, val_subjects, resclass, attuseg, thr, args.closing_iters
        )
        for sid, dsc in results:
            val_rows.append({"threshold": thr, "subject_id": sid, "dsc": f"{dsc:.4f}"})
            print(f"  {sid}: DSC = {dsc:.4f}")

        dscs = [d for _, d in results]
        val_summary[thr] = {
            "mean": float(np.mean(dscs)),
            "std": float(np.std(dscs)),
            "n": len(dscs),
        }
        print(f"  -> mean DSC = {val_summary[thr]['mean']:.4f} ± {val_summary[thr]['std']:.4f}")

    # Pick the best threshold by validation mean DSC
    best_thr = max(val_summary, key=lambda k: val_summary[k]["mean"])
    best_val_dsc = val_summary[best_thr]["mean"]
    print(f"\n*** Best threshold on validation: {best_thr:.2f} "
          f"(val DSC = {best_val_dsc:.4f}) ***\n")

    # Write validation per-subject CSV
    val_csv = args.output_dir / "threshold_sweep_val.csv"
    with open(val_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "subject_id", "dsc"])
        writer.writeheader()
        writer.writerows(val_rows)
    print(f"Per-subject validation results: {val_csv}")

    # Write summary CSV
    summary_csv = args.output_dir / "threshold_sweep_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "n", "val_mean_dsc", "val_std_dsc"])
        for thr in sorted(val_summary.keys()):
            s = val_summary[thr]
            writer.writerow([thr, s["n"], f"{s['mean']:.4f}", f"{s['std']:.4f}"])
    print(f"Validation summary: {summary_csv}")

    # ===== FINAL TEST EVALUATION AT BEST THRESHOLD =====
    print("\n" + "=" * 60)
    print(f"FINAL TEST EVALUATION (threshold={best_thr:.2f})")
    print("=" * 60)

    test_results = evaluate_at_threshold(
        test_dir, test_subjects, resclass, attuseg, best_thr, args.closing_iters
    )
    test_dscs = [d for _, d in test_results]
    test_mean = float(np.mean(test_dscs))
    test_std = float(np.std(test_dscs))

    test_rows = [{"threshold": best_thr, "subject_id": sid, "dsc": f"{dsc:.4f}"}
                 for sid, dsc in test_results]

    for sid, dsc in test_results:
        print(f"  {sid}: DSC = {dsc:.4f}")
    print(f"\n  Test mean DSC = {test_mean:.4f} ± {test_std:.4f} "
          f"(n={len(test_dscs)}, threshold={best_thr:.2f})")

    test_csv = args.output_dir / "threshold_sweep_test.csv"
    with open(test_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "subject_id", "dsc"])
        writer.writeheader()
        writer.writerows(test_rows)
    print(f"Final test results: {test_csv}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Best threshold on validation: {best_thr:.2f}")
    print(f"Validation DSC at best threshold: {best_val_dsc:.4f}")
    print(f"Test DSC at best threshold: {test_mean:.4f} ± {test_std:.4f}")
    print(f"Paper benchmark (full pipeline): 0.290")


if __name__ == "__main__":
    main()