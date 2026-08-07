"""
Evaluation pipeline: ResClass -> AttUSeg -> Post-processing -> full metric bundle.
Runs the full RAovSeg pipeline on test subjects.

Paper benchmarks (Dataset 2, n=8 test subjects):
  - Full pipeline (preprocess + ResClass + AttUSeg + postprocess): DSC = 0.290
  - Without postprocessing: DSC = 0.235
  - Without ResClass (AttUSeg only): DSC = 0.013

Metric bundle per subject (see metrics.py):
  DSC, IoU, sensitivity, precision, HD95 (voxels), volume error, volume_pred, volume_gt.

Aggregate reporting includes bootstrap 95% CI on each metric across the test cohort.

--target selects which organ label file to score against
  (default `ov` -> ov_label.npy; use `ut` for uterus once preprocess emits ut_label.npy).
"""

import sys
import json
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
from metrics import METRIC_KEYS, compute_metric_bundle, summary_row

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "RAovSeg"))
from RAovSeg_tools import postprocess_

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
def predict_subject(image: np.ndarray, resclass: nn.Module, attuseg: nn.Module,
                    use_resclass: bool = True, use_postprocess: bool = True,
                    resclass_threshold: float = RESCLASS_THRESHOLD,
                    closing_iterations: int = CLOSING_ITERATIONS) -> np.ndarray:
    """Run the pipeline on one subject and return the predicted binary volume."""
    n_slices = image.shape[0]
    prediction = np.zeros_like(image, dtype=np.float32)

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

    return prediction


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAovSeg pipeline on test set")
    parser.add_argument("--test-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data" / "processed" / "test")
    parser.add_argument("--models-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "models")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data" / "predictions")
    parser.add_argument("--target", type=str, default="ov",
                        help="Label channel to score against (ov | ut | em | cy). "
                             "Uses <target>_label.npy in each subject dir.")
    parser.add_argument("--resclass-threshold", type=float, default=RESCLASS_THRESHOLD,
                        help="ResClass binary threshold (paper unspecified; tuned on val)")
    parser.add_argument("--closing-iterations", type=int, default=CLOSING_ITERATIONS,
                        help="Postprocessing closing iterations (paper unspecified)")
    parser.add_argument("--metrics-out", type=Path, default=None,
                        help="Optional path to dump per-subject metrics as JSON.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_file = f"{args.target}_label.npy"

    # Clean stale *_pred.npy files from prior runs so the predictions dir
    # matches the current test split. Sweep CSVs (sweep_*.csv,
    # threshold_sweep_*.csv) and any other files are preserved.
    stale = list(args.output_dir.glob("*_pred.npy"))
    if stale:
        print(f"Removing {len(stale)} stale prediction file(s) from {args.output_dir}")
        for f in stale:
            f.unlink()

    # Load models — prefer target-suffixed checkpoints (new layout),
    # fall back to un-suffixed (legacy — ovary-only runs before --target existed).
    def _resolve(name: str) -> Path:
        target_path = args.models_dir / f"{name}_best_{args.target}.pth"
        legacy_path = args.models_dir / f"{name}_best.pth"
        if target_path.exists():
            return target_path
        if legacy_path.exists() and args.target == "ov":
            print(f"[eval] using legacy checkpoint: {legacy_path.name}")
            return legacy_path
        raise FileNotFoundError(
            f"No checkpoint for '{name}' target='{args.target}' in {args.models_dir}. "
            f"Looked for {target_path.name} then {legacy_path.name}."
        )

    resclass = load_resclass(_resolve("resclass"))
    attuseg = load_attuseg(_resolve("attuseg"))
    print(f"Models loaded. Device: {DEVICE}")
    print(f"Target: {args.target} (label file: {label_file})")
    print(f"ResClass threshold: {args.resclass_threshold}")
    print(f"Closing iterations: {args.closing_iterations}\n")

    # Per-mode per-subject metric records.
    modes = {"full": [], "no_postprocess": [], "no_resclass": []}

    for subj_dir in sorted(args.test_dir.iterdir()):
        if not subj_dir.is_dir():
            continue

        img_path = subj_dir / "image.npy"
        lbl_path = subj_dir / label_file
        if not img_path.exists() or not lbl_path.exists():
            print(f"{subj_dir.name}: SKIP (no {label_file})")
            continue

        image = np.load(img_path)
        label = np.load(lbl_path)

        # Full pipeline
        pred = predict_subject(image, resclass, attuseg,
                               use_resclass=True, use_postprocess=True,
                               resclass_threshold=args.resclass_threshold,
                               closing_iterations=args.closing_iterations)
        m_full = compute_metric_bundle(pred, label)
        m_full["subject"] = subj_dir.name
        modes["full"].append(m_full)

        # Ablation: no post-processing
        pred_nopp = predict_subject(image, resclass, attuseg,
                                    use_resclass=True, use_postprocess=False,
                                    resclass_threshold=args.resclass_threshold)
        m_nopp = compute_metric_bundle(pred_nopp, label)
        m_nopp["subject"] = subj_dir.name
        modes["no_postprocess"].append(m_nopp)

        # Ablation: no ResClass
        pred_norc = predict_subject(image, resclass, attuseg,
                                    use_resclass=False, use_postprocess=True,
                                    closing_iterations=args.closing_iterations)
        m_norc = compute_metric_bundle(pred_norc, label)
        m_norc["subject"] = subj_dir.name
        modes["no_resclass"].append(m_norc)

        # Save prediction from the full pipeline
        np.save(args.output_dir / f"{subj_dir.name}_pred.npy", pred)

        print(f"{subj_dir.name}: "
              f"DSC full={m_full['dsc']:.4f} | no_pp={m_nopp['dsc']:.4f} | no_rc={m_norc['dsc']:.4f}  "
              f"| HD95={m_full['hd95_mm']:.1f} mm  volErr={m_full['volume_error']:+.2f}")

    # Aggregate summary
    print("\n" + "=" * 78)
    print(f"RESULTS SUMMARY  (target={args.target})")
    print("=" * 78)
    header = f"{'mode':<20} {'metric':<15} {'n':>3}  {'mean':>7}  {'std':>7}  {'CI95_lo':>7}  {'CI95_hi':>7}"
    print(header)
    print("-" * len(header))

    aggregate = {}
    for mode_name, records in modes.items():
        aggregate[mode_name] = {}
        for key in METRIC_KEYS:
            values = [r[key] for r in records]
            row = summary_row(values, name=f"{mode_name}.{key}")
            aggregate[mode_name][key] = row
            print(f"{mode_name:<20} {key:<15} {row['n']:>3}  "
                  f"{row['mean']:>7.4f}  {row['std']:>7.4f}  "
                  f"{row['ci95_lo']:>7.4f}  {row['ci95_hi']:>7.4f}")
        print()

    print(f"Paper benchmarks: full=0.290, no_postprocess=0.235, no_resclass=0.013")

    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        # Persist the settings that were APPLIED at eval time. Downstream
        # analysis needs these to flag e.g. "ovary-tuned enhancement was
        # applied while scoring uterus" — non-obvious from the metrics alone.
        out = {
            "target": args.target,
            "applied_settings": {
                "resclass_threshold": args.resclass_threshold,
                "closing_iterations": args.closing_iterations,
                "resclass_threshold_note": (
                    "0.6 validated on OVARY only (see sweep_threshold.py); "
                    "used unchanged for uterus/em/cy — see M1 in audit."
                    if args.target != "ov" else "0.6 tuned on ovary validation set"
                ),
                "enhancement_window_note": (
                    "Enhancement window [0.22, 0.30] applied at preprocess time "
                    "was tuned by Liang et al. for OVARY tissue intensity. Any "
                    "non-ovary target scores are computed after that ovary-specific "
                    "saturation has been applied to the input images."
                ),
                "hd95_spacing_zyx_mm": [6.0, 0.35, 0.35],
            },
            "per_subject": {mode: records for mode, records in modes.items()},
            "aggregate": aggregate,
        }
        with args.metrics_out.open("w") as f:
            json.dump(out, f, indent=2)
        print(f"[eval] wrote {args.metrics_out}")


if __name__ == "__main__":
    main()
