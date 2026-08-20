"""
Preprocessing pipeline for UT-EndoMRI dataset.
Replicates the RAovSeg preprocessing from Liang et al. (2025).

Steps:
  1. Resample to 512x512, spacing (0.35, 0.35, 6.0) mm
  2. Percentile clip (1st-99th)
  3. Min-max normalization to [0, 1]
  4. Custom intensity enhancement (default o1=0.22, o2=0.30 — paper text;
     override with --o1/--o2 for band-sweep experiments)
"""

import sys
import os
import csv
import re
import random
import shutil
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# Add RAovSeg tools to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "RAovSeg"))
from RAovSeg_tools import ImgResample, ImgNorm, preprocess_

# --- Config ---
# Paper text: "Each slice was resampled to a height and width of 512 pixels."
# In-plane only — z is preserved at native count. The authors' tutorial.py
# passes out_size=(512,512,38), forcing z-resampling, but the paper text is
# authoritative and the native counts vary subject-to-subject (32–44 in our
# data); we compute out_size per subject from the native z-count.
OUT_SPACING = (0.35, 0.35, 6.0)
OUT_XY = 512
PERCENTILE_LOW = 1
PERCENTILE_HIGH = 99
# Paper text: "intensity of the ovaries in our dataset ranges from 0.22 to 0.3.
# In this study, we set..." — so o1=0.22. The repo's tutorial.py uses 0.24,
# which contradicts the paper text; we follow the paper.
O1 = 0.22
O2 = 0.3

# Paper (s41597-025-05623-3) inclusion: endometriosis + T2FS MRI + manual ovary seg.
# Paper exclusion: "(1) patients with obvious endometriomas and (2) patients with cysts."
# The 38 D2 subjects passing these criteria match the paper's "38 subjects ... included
# for pipeline development" exactly. The dataset README's D2-000..007 / D2-008..037 ID
# ranges do NOT correspond to the paper's split (they sum to 21, not 38, after exclusion);
# we therefore enumerate the 38 from the data and split 30 train/val + 8 test
# deterministically. Authors did not publish a canonical subject-ID list.
N_TRAIN_VAL = 30
N_TEST = 8
SPLIT_SEED = 42

MRI_SEQUENCES = ["T2FS", "T2", "T1FS", "T1"]
SUBJECT_RE = re.compile(r"^D2-(\d{3})$")


def _out_size_for(image: "sitk.Image") -> tuple[int, int, int]:
    """Compute resampling out_size: 512x512 in-plane, native z preserved."""
    native_z = int(image.GetSize()[2])
    return (OUT_XY, OUT_XY, native_z)


def preprocess_image(img_path: Path, skip_enhancement: bool = False,
                     o1: float = O1, o2: float = O2) -> np.ndarray:
    """Load, resample (in-plane only), normalize, and enhance a single MRI volume.

    If `skip_enhancement` is True, returns after the percentile-clip + minmax
    step, WITHOUT the RAovSeg custom ovary-intensity enhancement. Used by
    Option C of the augmentation experiments to test whether the enhancement
    step is what's hurting synth utility.

    `o1`/`o2` set the enhancement window (defaults from module constants,
    which reproduce Liang et al.'s [0.22, 0.30]). Override for band-sweep
    experiments.
    """
    img = sitk.ReadImage(str(img_path), sitk.sitkFloat64)
    img = ImgResample(img, out_spacing=OUT_SPACING, out_size=_out_size_for(img),
                      is_label=False, pad_value=0)
    img_array = sitk.GetArrayFromImage(img)
    img_array = ImgNorm(img_array, norm_type="percentile_clip",
                        percentile_low=PERCENTILE_LOW, percentile_high=PERCENTILE_HIGH)
    if skip_enhancement:
        return img_array.astype(np.float32)
    img_enhanced = preprocess_(img_array, o1=o1, o2=o2)
    return img_enhanced


def preprocess_label(label_path: Path) -> np.ndarray:
    """Load and resample a label volume (nearest-neighbor interpolation, in-plane only)."""
    lbl = sitk.ReadImage(str(label_path), sitk.sitkInt32)
    lbl = ImgResample(lbl, out_spacing=OUT_SPACING, out_size=_out_size_for(lbl),
                      is_label=True, pad_value=0)
    return sitk.GetArrayFromImage(lbl)


def find_best_sequence(subject_dir: Path, subject_id: str) -> Path | None:
    """Find the best available MRI sequence (T2FS preferred)."""
    for seq in MRI_SEQUENCES:
        path = subject_dir / f"{subject_id}_{seq}.nii.gz"
        if path.exists():
            return path
    return None


def process_subject(subject_dir: Path, output_dir: Path, subject_id: str,
                    skip_enhancement: bool = False,
                    o1: float = O1, o2: float = O2):
    """Process a single subject: image + ovary label.

    `skip_enhancement` — see preprocess_image docstring. Used by Option C
    of the augmentation experiments to skip the o1/o2 enhancement for
    synthetic subjects (D2-9XX) while keeping it for real ones.

    `o1`/`o2` — see preprocess_image docstring; used by the band-sweep
    experiment to move the enhancement window off the published default.
    """
    img_path = find_best_sequence(subject_dir, subject_id)
    if img_path is None:
        print(f"  SKIP {subject_id}: no MRI sequence found")
        return

    # Process image
    print(f"  Image: {img_path.name}"
          + ("  (enhancement SKIPPED)" if skip_enhancement else ""))
    img_enhanced = preprocess_image(img_path, skip_enhancement=skip_enhancement,
                                    o1=o1, o2=o2)

    # Save
    subj_out = output_dir / subject_id
    subj_out.mkdir(parents=True, exist_ok=True)
    np.save(subj_out / "image.npy", img_enhanced.astype(np.float32))

    # Save every available organ label. Downstream scripts pick by --target.
    # `ov_label.npy` is required by Liang's original pipeline; the others
    # unlock parallel-target training (uterus, em, cy) without a re-preprocess.
    emitted = []
    for target, suffix in (("ov", "_ov"), ("ut", "_ut"), ("em", "_em"), ("cy", "_cy")):
        label_path = subject_dir / f"{subject_id}{suffix}.nii.gz"
        if not label_path.exists():
            continue
        label_array = preprocess_label(label_path)
        np.save(subj_out / f"{target}_label.npy", label_array.astype(np.int8))
        emitted.append(f"{target}({int(np.sum(label_array > 0))} pos vox)")
    if emitted:
        print(f"  Labels: {', '.join(emitted)}")
    else:
        print(f"  Labels: none")

    print(f"  Output shape: {img_enhanced.shape}")


def inspect_subject(subject_dir: Path, subject_id: str) -> dict:
    """Record what files a subject dir contains. Does not yet assign a split."""
    if not SUBJECT_RE.match(subject_id):
        return {"subject_id": subject_id, "sequence": "", "has_ov": 0,
                "has_cy": 0, "has_em": 0, "split": "skip", "included": 0,
                "reason": "non-D2 subject id"}

    seq_path = find_best_sequence(subject_dir, subject_id)
    sequence = seq_path.name.replace(f"{subject_id}_", "").replace(".nii.gz", "") if seq_path else ""
    has_ov = (subject_dir / f"{subject_id}_ov.nii.gz").exists()
    has_cy = (subject_dir / f"{subject_id}_cy.nii.gz").exists()
    has_em = (subject_dir / f"{subject_id}_em.nii.gz").exists()

    row = {
        "subject_id": subject_id,
        "sequence": sequence,
        "has_ov": int(has_ov),
        "has_cy": int(has_cy),
        "has_em": int(has_em),
    }

    # Apply paper's exclusion criteria (cy/em) + inclusion (T2FS+ov)
    if not seq_path:
        row.update(split="skip", included=0, reason="no MRI sequence")
    elif not has_ov:
        row.update(split="skip", included=0, reason="no ov label")
    elif has_cy or has_em:
        # Paper excludes patients with obvious endometriomas or cysts
        row.update(split="excluded", included=0, reason="has cy/em")
    else:
        row.update(split="pending", included=0, reason="awaiting split assignment")
    return row


def scan_and_classify(data_dir: Path) -> list[dict]:
    """Walk the data dir, inspect each D2 subject, and assign 30/8 train_val/test
    among those passing paper inclusion criteria.

    The paper publishes counts (30+8=38) but no canonical subject-ID list, so we
    deterministically split with SPLIT_SEED. Subjects with cy/em are tagged
    `excluded`; subjects missing T2FS or ov are tagged `skip`.

    Returns rows with an added "source_dir" field so callers (specifically
    main(), after scanning an --extra-train-dir for synthetic subjects) know
    which directory each subject lives in.
    """
    rows = []
    for subj_dir in sorted(data_dir.iterdir()):
        if not subj_dir.is_dir() or not SUBJECT_RE.match(subj_dir.name):
            continue
        row = inspect_subject(subj_dir, subj_dir.name)
        row["source_dir"] = str(data_dir)
        rows.append(row)

    # Deterministic 30/8 split among inclusion-passing subjects
    eligible = [r["subject_id"] for r in rows if r["split"] == "pending"]
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(eligible)
    train_val_set = set(eligible[:N_TRAIN_VAL])
    test_set = set(eligible[N_TRAIN_VAL:N_TRAIN_VAL + N_TEST])

    for r in rows:
        if r["split"] != "pending":
            continue
        sid = r["subject_id"]
        if sid in train_val_set:
            r.update(split="train_val", included=1, reason="ok")
        elif sid in test_set:
            r.update(split="test", included=1, reason="ok")
        else:
            r.update(split="skip", included=0, reason="surplus past 30+8")
    return rows


def scan_extra_train_dir(extra_dir: Path) -> list[dict]:
    """Scan an additional directory for synthetic subjects and force them into
    the train_val split. Used for augmentation experiments.

    Same inclusion rules as the main path (must have a sequence file + ov label,
    no cy/em). But these subjects bypass the 30/8 shuffle — they all go into
    train_val so the 8 sacred D2 test subjects from the main dir are preserved.
    """
    rows = []
    for subj_dir in sorted(extra_dir.iterdir()):
        if not subj_dir.is_dir() or not SUBJECT_RE.match(subj_dir.name):
            continue
        row = inspect_subject(subj_dir, subj_dir.name)
        # Force the split: if the subject passes inclusion, it joins train_val.
        if row["split"] == "pending":
            row.update(split="train_val", included=1, reason="extra-train-dir (synthetic)")
        row["source_dir"] = str(extra_dir)
        rows.append(row)
    return rows


def write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["subject_id", "sequence", "has_ov", "has_cy", "has_em",
              "split", "included", "reason"]
    with open(path, "w", newline="") as f:
        # extrasaction="ignore" so the per-row source_dir field
        # (added when --extra-train-dir is used) doesn't crash DictWriter
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    by_split: dict[str, list[str]] = {}
    for r in rows:
        by_split.setdefault(r["split"], []).append(r["subject_id"])
    print("\n=== Split summary (paper inclusion criteria + deterministic 30/8) ===")
    for split in ("train_val", "test", "excluded", "skip"):
        ids = by_split.get(split, [])
        print(f"  {split:10s} ({len(ids):2d}): {', '.join(ids) if ids else '-'}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess UT-EndoMRI for RAovSeg pipeline")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "UT-EndoMRI" / "D2_TCPW")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data" / "processed")
    parser.add_argument("--extra-train-dir", type=Path, default=None,
                        help="Optional second data dir whose subjects (matching D2-XXX naming) "
                             "are forced into the train_val split — used for synthetic data "
                             "augmentation experiments. Real test split (8 sacred subjects) is "
                             "preserved unchanged.")
    parser.add_argument("--skip-enhancement-for-prefix", default=None,
                        help="Option C: skip the o1/o2 ovary-intensity enhancement step for "
                             "subjects whose ID starts with this prefix (e.g. 'D2-9' for synthetic "
                             "subjects). Real subjects still get enhancement. Used to test whether "
                             "the enhancement step is what's hurting synth utility.")
    parser.add_argument("--o1", type=float, default=O1,
                        help=f"Lower bound of the enhancement window (default {O1}, "
                             f"reproducing Liang et al.). Band-sweep experiment overrides this.")
    parser.add_argument("--o2", type=float, default=O2,
                        help=f"Upper bound of the enhancement window (default {O2}). "
                             f"Band-sweep experiment overrides this.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan and write manifest only; skip preprocessing")
    args = parser.parse_args()

    if not (0.0 <= args.o1 < args.o2 <= 1.0):
        parser.error(f"require 0 <= --o1 ({args.o1}) < --o2 ({args.o2}) <= 1")

    data_dir = args.data_dir
    output_dir = args.output_dir

    # Clean previous split outputs before re-running so stale subject dirs from
    # an earlier (different) split don't linger in train_val/, test/, excluded/.
    if not args.dry_run:
        for sub in ("train_val", "test", "excluded"):
            target = output_dir / sub
            if target.exists():
                print(f"Cleaning {target}")
                shutil.rmtree(target)

    rows = scan_and_classify(data_dir)
    if args.extra_train_dir is not None:
        if not args.extra_train_dir.exists():
            raise FileNotFoundError(f"--extra-train-dir not found: {args.extra_train_dir}")
        extra_rows = scan_extra_train_dir(args.extra_train_dir)
        # Sanity: extra subject IDs must not collide with main subject IDs
        main_ids = {r["subject_id"] for r in rows}
        for r in extra_rows:
            if r["subject_id"] in main_ids:
                raise ValueError(
                    f"--extra-train-dir subject id {r['subject_id']} collides with the main "
                    f"--data-dir. Synthetic subjects should use D2-9XX naming (or another "
                    f"range disjoint from real subjects)."
                )
        print(f"Adding {sum(1 for r in extra_rows if r['included'])} subjects from "
              f"--extra-train-dir into train_val (out of {len(extra_rows)} scanned)")
        rows.extend(extra_rows)
    manifest_path = output_dir / "manifest.csv"
    write_manifest(rows, manifest_path)
    print(f"Wrote manifest: {manifest_path} ({len(rows)} subjects scanned)")
    print_summary(rows)

    if args.dry_run:
        print("\nDry run — skipping preprocessing.")
        return

    skip_prefix = args.skip_enhancement_for_prefix
    if skip_prefix:
        print(f"[preprocess] Option C: skipping enhancement for subjects with prefix '{skip_prefix}'")
    if (args.o1, args.o2) != (O1, O2):
        print(f"[preprocess] enhancement window OVERRIDE: o1={args.o1}, o2={args.o2} "
              f"(default was o1={O1}, o2={O2})")
    for r in rows:
        if not r["included"]:
            continue
        sid = r["subject_id"]
        split = r["split"]
        src_dir = Path(r.get("source_dir", str(data_dir)))
        skip_enh = bool(skip_prefix and sid.startswith(skip_prefix))
        print(f"\nProcessing {sid} [{split}] from {src_dir} (sequence: {r['sequence']}):")
        process_subject(src_dir / sid, output_dir / split, sid,
                        skip_enhancement=skip_enh, o1=args.o1, o2=args.o2)

    print("\nDone.")


if __name__ == "__main__":
    main()
