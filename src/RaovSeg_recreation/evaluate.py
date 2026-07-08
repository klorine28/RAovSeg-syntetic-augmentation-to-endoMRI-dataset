"""
Evaluation pipeline: ResClass -> AttUSeg -> Post-processing -> DSC.
Runs the full RAovSeg pipeline on test subjects.

Paper benchmarks (Dataset 2, n=8 test subjects):
  - Full pipeline (preprocess + ResClass + AttUSeg + postprocess): DSC = 0.290
  - Without postprocessing: DSC = 0.235
  - Without ResClass (AttUSeg only): DSC = 0.013
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from monai.networks.nets import AttentionUnet

# src/ is on sys.path when running `python src/evaluate.py`, so the sibling
# import works without further setup.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_resclass import TwoBlockResNet

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "RAovSeg"))
from RAovSeg_tools import postprocess_, dsc_cal_np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Validation-tuned: paper does not specify the ResClass binary threshold.
# We selected 0.6 on the validation set across a sweep of {0.2..0.7}.
# See sweep_threshold.py for the full sweep.
RESCLASS_THRESHOLD = 0.6
CLOSING_ITERATIONS = 10  # paper does not specify; tuned default


def load_resclass(model_path: Path) -> nn.Module:
    model = TwoBlockResNet(in_channels=1, num_classes=1, dropout=0.2)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()
    return model


def load_attuseg(model_path: Path) -> nn.Module:
    model = AttentionUnet(
        spatial_dims=2, in_channels=1, out_channels=1,
        channels=(16, 32, 64, 128), strides=(2, 2, 2), dropout=0.2,
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()
    return model


@torch.no_grad()
def predict_subject(image: np.ndarray, label: np.ndarray,
                    resclass: nn.Module, attuseg: nn.Module,
                    use_resclass: bool = True, use_postprocess: bool = True,
                    resclass_threshold: float = RESCLASS_THRESHOLD,
                    closing_iterations: int = CLOSING_ITERATIONS):
    """Run full pipeline on one subject. Returns DSC and prediction volume."""
    n_slices = image.shape[0]
    prediction = np.zeros_like(label, dtype=np.float32)

    for s in range(n_slices):
        img_slice = torch.from_numpy(image[s][np.newaxis, np.newaxis, ...]).float().to(DEVICE)

        # Stage 1: classify slice
        if use_resclass:
            logit = resclass(img_slice).squeeze()
            has_ovary = torch.sigmoid(logit).item() > resclass_threshold
            if not has_ovary:
                continue

        # Stage 2: segment
        seg_output = attuseg(img_slice)
        seg_binary = (torch.sigmoid(seg_output) > 0.5).cpu().numpy().squeeze()
        prediction[s] = seg_binary

    # Post-processing
    if use_postprocess:
        prediction = postprocess_(prediction, closing_iterations=closing_iterations).astype(np.float32)

    dsc = dsc_cal_np(prediction, label)
    return dsc, prediction


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAovSeg pipeline on test set")
    parser.add_argument("--test-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data" / "processed" / "test")
    parser.add_argument("--models-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "models")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data" / "predictions")
    parser.add_argument("--resclass-threshold", type=float, default=RESCLASS_THRESHOLD,
                        help="ResClass binary threshold (paper unspecified; tuned on val)")
    parser.add_argument("--closing-iterations", type=int, default=CLOSING_ITERATIONS,
                        help="Postprocessing closing iterations (paper unspecified)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Clean stale *_pred.npy files from prior runs so the predictions dir
    # matches the current test split. Sweep CSVs (sweep_*.csv,
    # threshold_sweep_*.csv) and any other files are preserved.
    stale = list(args.output_dir.glob("*_pred.npy"))
    if stale:
        print(f"Removing {len(stale)} stale prediction file(s) from {args.output_dir}")
        for f in stale:
            f.unlink()

    # Load models
    resclass = load_resclass(args.models_dir / "resclass_best.pth")
    attuseg = load_attuseg(args.models_dir / "attuseg_best.pth")
    print(f"Models loaded. Device: {DEVICE}")
    print(f"ResClass threshold: {args.resclass_threshold}")
    print(f"Closing iterations: {args.closing_iterations}\n")

    # Evaluate each test subject
    results = {"full": [], "no_postprocess": [], "no_resclass": []}

    for subj_dir in sorted(args.test_dir.iterdir()):
        if not subj_dir.is_dir():
            continue

        img_path = subj_dir / "image.npy"
        lbl_path = subj_dir / "ov_label.npy"

        if not img_path.exists() or not lbl_path.exists():
            print(f"{subj_dir.name}: SKIP (no label)")
            continue

        image = np.load(img_path)
        label = np.load(lbl_path)

        # Full pipeline
        dsc_full, pred = predict_subject(image, label, resclass, attuseg,
                                         use_resclass=True, use_postprocess=True,
                                         resclass_threshold=args.resclass_threshold,
                                         closing_iterations=args.closing_iterations)
        results["full"].append(dsc_full)

        # Ablation: no post-processing
        dsc_nopp, _ = predict_subject(image, label, resclass, attuseg,
                                      use_resclass=True, use_postprocess=False,
                                      resclass_threshold=args.resclass_threshold)
        results["no_postprocess"].append(dsc_nopp)

        # Ablation: no ResClass
        dsc_norc, _ = predict_subject(image, label, resclass, attuseg,
                                      use_resclass=False, use_postprocess=True,
                                      closing_iterations=args.closing_iterations)
        results["no_resclass"].append(dsc_norc)

        # Save prediction
        np.save(args.output_dir / f"{subj_dir.name}_pred.npy", pred)

        print(f"{subj_dir.name}: DSC full={dsc_full:.4f} | no_pp={dsc_nopp:.4f} | no_rc={dsc_norc:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for key, values in results.items():
        if values:
            print(f"  {key:20s}: mean DSC = {np.mean(values):.4f} ± {np.std(values):.4f} (n={len(values)})")

    print(f"\nPaper benchmarks: full=0.290, no_postprocess=0.235, no_resclass=0.013")


if __name__ == "__main__":
    main()
