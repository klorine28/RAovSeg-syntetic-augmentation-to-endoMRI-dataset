"""
Preprocessing pass for the GENERATOR (Exp 1a, 1b, 1c, 2, 3).

Separate from RAovSeg preprocessing. The differences:

  RAovSeg pipeline                       Generator pipeline (this script)
  ----------------                       --------------------------------
  Output: .npy                           Output: .nii.gz (preserves affine)
  Single-channel binary (ovary only)     6-channel one-hot
                                            ch0=outside_body (air),
                                            ch1=uterus,
                                            ch2=L-ov, ch3=R-ov,
                                            ch4=endo, ch5=body_other
                                                       (inside body, no target)
  Ovary intensity enhancement APPLIED    Enhancement NOT applied
  (img → enhanced[0.22, 0.3] → 1, ...)   (plain [0, 1] normalised)
  Excludes ovary-free slices in train    Keeps all slices (full anatomy)

The 6th channel (body silhouette minus target organs) is derived from the
normalised T2FS image via threshold > 0.05, morphological closing, and hole
filling. It gives the generator explicit "inside body, fill with plausible
non-target tissue" conditioning, eliminating the edge-noise artefact seen
with only the 5-channel labels.

Why no enhancement here:
  Enhancement is RAovSeg-specific. If the generator trained on enhanced
  images, downstream RAovSeg would apply enhancement a SECOND time when
  ingesting the synthetic pool → double-clamp, broken inference.

L/R ovary split:
  UT-EndoMRI ships a single _ov.nii.gz with both ovaries combined.
  We split it via connected-components: largest CCs sorted by mean
  x-centroid → leftmost gets L-channel, rightmost gets R-channel.
  Fallback to pure midline split when only one CC exists (one ovary
  absent, or two ovaries fused into one CC by thin connectivity).
  Per-subject decisions are logged.

Source data layout (D2_TCPW, confirmed via ls):
    UT-EndoMRI/D2_TCPW/D2-XXX/
        D2-XXX_T2FS.nii.gz       # used
        D2-XXX_T1FS.nii.gz       # ignored
        D2-XXX_T1.nii.gz         # ignored
        D2-XXX_T2.nii.gz         # ignored
        D2-XXX_ut.nii.gz         # uterus
        D2-XXX_ov.nii.gz         # combined ovaries (will be split)
        D2-XXX_em.nii.gz         # endometrioma (only 11 subjects)

Output layout:
    preprocessed/D2/<subject_id>/
        image_T2FS.nii.gz                  # (Z, H, W) float32 [0,1], 0.35x0.35x6mm
        label_T2FS.nii.gz                  # 4D vector NIfTI, 5 components, for dataloader
        label_T2FS_bg.nii.gz               # per-class binaries, for ITK-SNAP / Slicer QA
        label_T2FS_uterus.nii.gz
        label_T2FS_ov_L.nii.gz
        label_T2FS_ov_R.nii.gz
        label_T2FS_em.nii.gz
    preprocessed/D2/preprocess_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label as cc_label
from scipy.ndimage import binary_closing, binary_fill_holes


SUBJECT_DIR_RE = re.compile(r"^D2-\d{3}$")

# --------------------------------------------------------------------------- #
# Constants matching the existing RAovSeg recreation pipeline
# --------------------------------------------------------------------------- #
TARGET_SIZE = 512
TARGET_SPACING = (0.35, 0.35, 6.0)  # (sx, sy, sz) in mm — matches RAovSeg recreation
SEQUENCE = "T2FS"

# Threshold (on normalised [0,1] image) for the body silhouette mask.
# Pelvic T2FS: air is near 0 after percentile-clip + minmax, body tissue is
# meaningfully brighter. 0.05 reliably catches body without including air.
BODY_THRESHOLD = 0.05

# Channel layout in the output 6-channel label.
# bg is "outside body" (air). body is "inside body, not a target organ".
# Target organs (uterus, L-ov, R-ov, em) take priority over body where they
# overlap. One-hot semantics: exactly one channel == 1 per voxel.
CH_BG, CH_UTERUS, CH_OV_L, CH_OV_R, CH_EM, CH_BODY = 0, 1, 2, 3, 4, 5
CH_NAMES = {CH_BG: "outside_body", CH_UTERUS: "uterus", CH_OV_L: "ov_L",
            CH_OV_R: "ov_R", CH_EM: "em", CH_BODY: "body_other"}
N_CHANNELS = 6


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _read(path: Path, pixel_type=sitk.sitkFloat32) -> sitk.Image:
    return sitk.ReadImage(str(path), pixel_type)


def _resample_image_to_target(img: sitk.Image) -> sitk.Image:
    """Resample to TARGET_SIZE in-plane and TARGET_SPACING. Linear interp."""
    orig_size = img.GetSize()
    orig_spacing = img.GetSpacing()
    # Z extent (mm) preserved — recompute Z size at new spacing
    z_extent_mm = orig_size[2] * orig_spacing[2]
    new_z_size = max(int(round(z_extent_mm / TARGET_SPACING[2])), 1)
    new_size = [TARGET_SIZE, TARGET_SIZE, new_z_size]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(TARGET_SPACING)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    return resampler.Execute(img)


def _resample_label_to_image(label: sitk.Image, ref: sitk.Image) -> sitk.Image:
    """Resample a label volume into the *image's* post-resample frame.

    This is the audit's point 4 fix: use the resampled image as the reference
    so labels and image share size/spacing/direction/origin exactly. Avoids
    silent misalignment when the rater drew on a re-oriented copy.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(label)


def _body_silhouette(image_zyx: np.ndarray) -> np.ndarray:
    """Extract a body-vs-air binary mask from the normalised image.

    Pipeline per slice: threshold > BODY_THRESHOLD, morphological closing
    (3 iterations) to bridge intra-body discontinuities, fill internal
    holes. Returns (Z, H, W) uint8 binary.

    This is a cheap, deterministic proxy — UT-EndoMRI doesn't ship body
    silhouette annotations. For pelvic T2FS after percentile-clip and
    minmax, air is reliably below 0.05 and the body is contiguous, so
    this is robust without any per-subject tuning.
    """
    mask = image_zyx > BODY_THRESHOLD
    closed = np.stack([binary_closing(s, iterations=3) for s in mask])
    filled = np.stack([binary_fill_holes(s) for s in closed])
    return filled.astype(np.uint8)


def _base_preprocess_image(image_path: Path) -> sitk.Image:
    """Load → clip 1st-99th percentile → normalise [0,1] → resample. NO enhancement."""
    img = _read(image_path, sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(img)  # (Z, Y, X)

    p1, p99 = np.percentile(arr, [1, 99])
    arr = np.clip(arr, p1, p99)
    arr = (arr - p1) / max(p99 - p1, 1e-6)

    norm = sitk.GetImageFromArray(arr.astype(np.float32))
    norm.CopyInformation(img)
    return _resample_image_to_target(norm)


def _split_ovary_lr(ov_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Split combined ovary mask into L and R using 3D connected components.

    Strategy:
      1. Find connected components in 3D (binary `ov_arr`).
      2. If 0 CCs → both channels empty (subject has no ovary; rare).
      3. If 1 CC → midline fallback. Split the SINGLE CC by image-x midline.
      4. If 2 CCs → the two largest are the ovaries. Sort by mean x-centroid.
                    Smaller mean x = "left in image space". Assign accordingly.
                    Tiny extra CCs (< 5% of largest) are merged into the
                    closest of the two by Euclidean centroid distance.
      5. If ≥3 CCs → take the 2 largest as the ovaries; merge stragglers as
                     in case 4.

    Returns (left_mask, right_mask, decision_string) where decision_string
    is one of: "no_ovary", "midline_fallback", "cc_2", "cc_n_merged".

    NOTE on "left vs right":
      The convention here is purely image-space — leftmost in the array
      x-axis goes to L. Anatomical L vs R depends on patient orientation
      (LPS vs RAS) and direction cosines. For the generator's purposes
      (SPADE conditioning + flip augmentation), image-space L/R is what
      matters: the channels are anatomically meaningful as long as they
      are consistent across subjects, and they are if we always work in
      the resampled image's frame. The scanner/orientation convention can
      be reconciled at downstream evaluation time if needed.
    """
    if ov_arr.sum() == 0:
        empty = np.zeros_like(ov_arr, dtype=np.uint8)
        return empty, empty.copy(), "no_ovary"

    # 3D 6-connectivity (faces only). 26-connectivity tends to fuse separate
    # ovaries through thin diagonal noise; 6 is conservative.
    structure = np.zeros((3, 3, 3), dtype=int)
    structure[1, 1, :] = 1
    structure[1, :, 1] = 1
    structure[:, 1, 1] = 1

    labeled, n_cc = cc_label(ov_arr > 0, structure=structure)

    if n_cc == 1:
        # Midline fallback: split the single CC by x = W/2
        w = ov_arr.shape[-1]  # last axis is X (image columns)
        midline = w // 2
        single_mask = (labeled == 1)
        left_mask = single_mask.copy()
        left_mask[..., midline:] = False
        right_mask = single_mask.copy()
        right_mask[..., :midline] = False
        return (
            left_mask.astype(np.uint8),
            right_mask.astype(np.uint8),
            "midline_fallback",
        )

    # ≥2 CCs: pick the 2 largest, assign by x-centroid, merge stragglers
    cc_sizes = np.array(
        [(labeled == i).sum() for i in range(1, n_cc + 1)], dtype=np.int64
    )
    order = np.argsort(-cc_sizes)  # descending by size
    top1, top2 = order[0] + 1, order[1] + 1

    def x_centroid(cc_id: int) -> float:
        coords = np.argwhere(labeled == cc_id)
        return float(coords[:, -1].mean())

    def centroid_3d(cc_id: int) -> np.ndarray:
        coords = np.argwhere(labeled == cc_id)
        return coords.mean(axis=0)

    if x_centroid(top1) < x_centroid(top2):
        left_id, right_id = top1, top2
    else:
        left_id, right_id = top2, top1

    left_mask = (labeled == left_id)
    right_mask = (labeled == right_id)

    # Merge tiny stragglers (< 5% of larger ovary) into the nearest of the two
    decision = "cc_2"
    big_size = max(cc_sizes[left_id - 1], cc_sizes[right_id - 1])
    threshold = max(big_size * 0.05, 10)
    left_centroid = centroid_3d(left_id)
    right_centroid = centroid_3d(right_id)

    for cc_id in range(1, n_cc + 1):
        if cc_id in (left_id, right_id):
            continue
        if cc_sizes[cc_id - 1] >= threshold:
            # A genuinely large extra component — also merge to nearest, but flag
            decision = "cc_n_merged"
        c = centroid_3d(cc_id)
        d_left = np.linalg.norm(c - left_centroid)
        d_right = np.linalg.norm(c - right_centroid)
        if d_left <= d_right:
            left_mask = left_mask | (labeled == cc_id)
        else:
            right_mask = right_mask | (labeled == cc_id)
        if cc_sizes[cc_id - 1] >= threshold and decision == "cc_2":
            decision = "cc_n_merged"

    return left_mask.astype(np.uint8), right_mask.astype(np.uint8), decision


def _save_per_class_binaries(
    label_4d: np.ndarray,
    ref_image: sitk.Image,
    out_dir: Path,
    sequence: str,
) -> None:
    """Dump 5 separate binary NIfTIs for ITK-SNAP / Slicer QA."""
    for ch in range(label_4d.shape[0]):
        binary = label_4d[ch].astype(np.uint8)
        img = sitk.GetImageFromArray(binary)
        img.CopyInformation(ref_image)
        out_path = out_dir / f"label_{sequence}_{CH_NAMES[ch]}.nii.gz"
        sitk.WriteImage(img, str(out_path))


# --------------------------------------------------------------------------- #
# Per-subject pipeline
# --------------------------------------------------------------------------- #
def process_subject(
    subj_id: str,
    raw_subject_dir: Path,
    out_dir: Path,
    has_em_expected: bool,
    save_per_class: bool = True,
) -> dict:
    """Process one subject. Returns a summary dict for the run log."""
    img_path = raw_subject_dir / f"{subj_id}_{SEQUENCE}.nii.gz"
    ut_path = raw_subject_dir / f"{subj_id}_ut.nii.gz"
    ov_path = raw_subject_dir / f"{subj_id}_ov.nii.gz"
    em_path = raw_subject_dir / f"{subj_id}_em.nii.gz"

    if not img_path.exists():
        return {"status": "skipped", "reason": f"missing {img_path.name}"}
    if not ut_path.exists():
        return {"status": "skipped", "reason": f"missing {ut_path.name}"}
    if not ov_path.exists():
        return {"status": "skipped", "reason": f"missing {ov_path.name}"}
    if has_em_expected and not em_path.exists():
        # Loud failure — manifest lied or file was deleted
        raise FileNotFoundError(
            f"{subj_id}: manifest expected has_em=1 but {em_path} is missing"
        )

    # 1) Image: clip→normalise→resample to TARGET_SPACING
    image = _base_preprocess_image(img_path)

    # 2) Labels: resample each into the IMAGE'S frame (audit point 4 fix)
    ut_resampled = _resample_label_to_image(_read(ut_path, sitk.sitkUInt8), image)
    ov_resampled = _resample_label_to_image(_read(ov_path, sitk.sitkUInt8), image)
    em_resampled = (
        _resample_label_to_image(_read(em_path, sitk.sitkUInt8), image)
        if em_path.exists() else None
    )

    ut_arr = (sitk.GetArrayFromImage(ut_resampled) > 0).astype(np.uint8)
    ov_arr = (sitk.GetArrayFromImage(ov_resampled) > 0).astype(np.uint8)
    em_arr = (
        (sitk.GetArrayFromImage(em_resampled) > 0).astype(np.uint8)
        if em_resampled is not None else np.zeros_like(ut_arr)
    )

    # 3) Split ovary into L/R
    ov_L, ov_R, lr_decision = _split_ovary_lr(ov_arr)

    # 4) Body silhouette from the normalised image
    image_arr_for_body = sitk.GetArrayFromImage(image)
    body_arr = _body_silhouette(image_arr_for_body)

    # 5) Build 6-channel one-hot label, resolve overlaps with priority
    z_size = image.GetSize()[2]
    label_4d = np.zeros((N_CHANNELS, z_size, TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
    label_4d[CH_UTERUS] = ut_arr
    label_4d[CH_OV_L] = ov_L
    label_4d[CH_OV_R] = ov_R
    label_4d[CH_EM] = em_arr

    # Priority for overlap resolution among target organs: endo > L-ov > R-ov > uterus.
    # Endometriomas are physically inside ovary contours, so they win those.
    priority = [CH_EM, CH_OV_L, CH_OV_R, CH_UTERUS]
    overlap_voxels = 0
    for i, hi in enumerate(priority):
        for lo in priority[i + 1:]:
            both = (label_4d[hi] > 0) & (label_4d[lo] > 0)
            n_overlap = int(both.sum())
            if n_overlap > 0:
                label_4d[lo][both] = 0
                overlap_voxels += n_overlap

    # 6) Compose body and bg channels so the 6-channel label stays one-hot:
    #   - target organ channels (1..4) keep their voxels as resolved above
    #   - body channel (5) = inside body AND not in any target organ
    #   - bg channel (0)   = outside body silhouette (air / non-tissue)
    target_organs = (
        label_4d[CH_UTERUS] | label_4d[CH_OV_L]
        | label_4d[CH_OV_R] | label_4d[CH_EM]
    ).astype(bool)
    body_bool = body_arr.astype(bool)
    label_4d[CH_BODY] = (body_bool & ~target_organs).astype(np.uint8)
    label_4d[CH_BG] = (~body_bool).astype(np.uint8)

    # 5) Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(out_dir / f"image_{SEQUENCE}.nii.gz"))

    # Combined 4D vector NIfTI for the dataloader.
    # SimpleITK stores leading dim as the vector component → on disk shape
    # becomes (Z, H, W, 5) when read back, which dataset.py handles.
    label_vec = sitk.GetImageFromArray(label_4d)
    sitk.WriteImage(label_vec, str(out_dir / f"label_{SEQUENCE}.nii.gz"))

    if save_per_class:
        _save_per_class_binaries(label_4d, image, out_dir, SEQUENCE)

    # 6) Summary stats
    n_ovary_slices = int(((label_4d[CH_OV_L] | label_4d[CH_OV_R]).any(axis=(1, 2))).sum())
    return {
        "status": "ok",
        "z": z_size,
        "spacing": TARGET_SPACING,
        "n_ovary_slices": n_ovary_slices,
        "lr_decision": lr_decision,
        "n_voxels": {
            "uterus": int(label_4d[CH_UTERUS].sum()),
            "ov_L": int(label_4d[CH_OV_L].sum()),
            "ov_R": int(label_4d[CH_OV_R].sum()),
            "em": int(label_4d[CH_EM].sum()),
            "body_other": int(label_4d[CH_BODY].sum()),
            "outside_body": int(label_4d[CH_BG].sum()),
        },
        "overlap_voxels_resolved": overlap_voxels,
        "has_em_file": em_path.exists(),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True,
                        help="Root with one dir per subject, e.g. UT-EndoMRI/D2_TCPW")
    parser.add_argument("--out_root", required=True,
                        help="Output root, e.g. preprocessed/D2")
    parser.add_argument("--split_file", required=True,
                        help="Generator split JSON (from build_generator_split.py)")
    parser.add_argument("--manifest", required=True,
                        help="RAovSeg manifest.csv, used for has_em validation")
    parser.add_argument("--no_per_class", action="store_true",
                        help="Skip per-class binary NIfTIs (saves disk, kills viewer QA)")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)

    with open(args.split_file) as f:
        splits = json.load(f)
    subjects = sorted(set(splits["train"]) | set(splits["test"]))
    print(f"[preprocess] {len(subjects)} subjects "
          f"({len(splits['train'])} train, {len(splits['test'])} test)")

    # Clean stale subject dirs from previous runs so out_root only contains
    # subjects that belong to the current split. Only directories matching
    # the D2-XXX pattern are touched; preprocess_summary.json and any other
    # sibling files are preserved.
    if out_root.exists():
        keep = set(subjects)
        stale_dirs = [
            d for d in out_root.iterdir()
            if d.is_dir() and SUBJECT_DIR_RE.match(d.name) and d.name not in keep
        ]
        if stale_dirs:
            print(f"[preprocess] removing {len(stale_dirs)} stale subject "
                  f"dir(s) not in current split: "
                  f"{sorted(d.name for d in stale_dirs)}")
            for d in stale_dirs:
                shutil.rmtree(d)

    import pandas as pd
    manifest = pd.read_csv(args.manifest).set_index("subject_id")

    summary = []
    for i, subj in enumerate(subjects, 1):
        subj_dir = raw_root / subj
        if not subj_dir.exists():
            print(f"[{i:>3}/{len(subjects)}] {subj}: MISSING in {raw_root}")
            summary.append({"subject": subj, "status": "missing"})
            continue

        has_em_expected = bool(manifest.loc[subj, "has_em"]) if subj in manifest.index else False

        try:
            result = process_subject(
                subj_id=subj,
                raw_subject_dir=subj_dir,
                out_dir=out_root / subj,
                has_em_expected=has_em_expected,
                save_per_class=not args.no_per_class,
            )
            print(f"[{i:>3}/{len(subjects)}] {subj}: {result['status']} "
                  f"z={result.get('z')} ov_slices={result.get('n_ovary_slices')} "
                  f"lr={result.get('lr_decision')} "
                  f"L={result['n_voxels']['ov_L'] if 'n_voxels' in result else '?'} "
                  f"R={result['n_voxels']['ov_R'] if 'n_voxels' in result else '?'}")
            summary.append({"subject": subj, **result})
        except Exception as e:
            print(f"[{i:>3}/{len(subjects)}] {subj}: ERROR {type(e).__name__}: {e}")
            summary.append({"subject": subj, "status": "error", "error": str(e)})
            # Don't bail — process the rest, surface all problems at the end

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "preprocess_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    n_ok = sum(1 for s in summary if s.get("status") == "ok")
    n_err = sum(1 for s in summary if s.get("status") == "error")
    n_missing = sum(1 for s in summary if s.get("status") == "missing")

    # Aggregate L/R decisions for visibility
    lr_counts: dict[str, int] = {}
    for s in summary:
        if s.get("status") == "ok":
            lr_counts[s["lr_decision"]] = lr_counts.get(s["lr_decision"], 0) + 1

    print(f"\n[preprocess] DONE: {n_ok} ok, {n_err} errors, {n_missing} missing")
    print(f"[preprocess] L/R split decisions: {lr_counts}")
    print(f"[preprocess] summary written to {out_root}/preprocess_summary.json")


if __name__ == "__main__":
    main()
