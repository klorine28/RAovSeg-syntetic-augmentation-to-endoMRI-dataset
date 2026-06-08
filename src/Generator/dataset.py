"""
2D slice dataset for D2 T2FS, with paired 5-channel label maps.

Layout assumed (produced by src.Generator.preprocess_for_generator):

    preprocessed/D2/
        D2-001/
            image_T2FS.nii.gz       # (Z, H, W) float32 in [0, 1]
            label_T2FS.nii.gz       # 4D vector NIfTI, on-disk shape (Z, H, W, 5) uint8
            label_T2FS_bg.nii.gz    # per-class binaries (ignored by this loader)
            ...

Slices are extracted on-the-fly from cached 3D volumes. We pre-scan all volumes
once at construction time to build the (subject, z) index plus an "is_ovary"
flag for the weighted sampler.

NOTE: Ovary intensity enhancement is NOT applied here. Generators learn in
plain [0,1] space; enhancement is downstream RAovSeg-specific.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

SEQUENCE = "T2FS"


@dataclass
class SliceIndex:
    subject: str
    z: int
    has_ovary: bool


def _load_image_nifti(path: Path) -> np.ndarray:
    """Load a 3D image NIfTI as (Z, H, W) float32."""
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img).astype(np.float32)


def _load_label_nifti(path: Path, num_channels: int) -> np.ndarray:
    """Load the 5-channel vector label NIfTI as (C, Z, H, W) uint8.

    SimpleITK stores vector NIfTIs with the component dim last, so reading
    `label_T2FS.nii.gz` gives shape (Z, H, W, C). We transpose to (C, Z, H, W).

    Backwards compatible: also accepts (Z, H, W) integer labels (one-hot'd) or
    already-(C, Z, H, W) volumes, in case preprocessing is changed later.
    """
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)

    if arr.ndim == 3:
        # Integer-valued labels → one-hot
        out = np.stack([(arr == c).astype(np.uint8) for c in range(num_channels)], axis=0)
        return out
    if arr.ndim == 4:
        if arr.shape[-1] == num_channels:
            return np.transpose(arr, (3, 0, 1, 2)).astype(np.uint8)
        if arr.shape[0] == num_channels:
            return arr.astype(np.uint8)
    raise ValueError(f"Unexpected label shape {arr.shape} at {path}")


class D2SliceDataset(Dataset):
    """2D axial slice dataset for D2 T2FS + 5-channel labels."""

    def __init__(
        self,
        preprocessed_root: str | Path,
        split_file: str | Path,
        split: str = "train",
        sequence: str = SEQUENCE,
        num_label_channels: int = 5,
        image_size: int = 512,
    ):
        self.root = Path(preprocessed_root)
        self.sequence = sequence
        self.num_label_channels = num_label_channels
        self.image_size = image_size

        with open(split_file) as f:
            splits = json.load(f)
        self.subjects: List[str] = splits[split]

        # Cache full volumes in RAM. ~30 subjects × ~512×512×~30 float32 ≈ 1GB
        # for images + 5x uint8 labels ≈ 1.5GB total. Fine on Stanage.
        self.images: dict[str, np.ndarray] = {}
        self.labels: dict[str, np.ndarray] = {}
        self.index: List[SliceIndex] = []

        skipped = []
        for subj in self.subjects:
            img_path = self.root / subj / f"image_{sequence}.nii.gz"
            lbl_path = self.root / subj / f"label_{sequence}.nii.gz"
            if not img_path.exists() or not lbl_path.exists():
                skipped.append(subj)
                continue

            img = _load_image_nifti(img_path)                            # (Z, H, W)
            lbl = _load_label_nifti(lbl_path, num_label_channels)        # (C, Z, H, W)

            assert img.shape[-2:] == (image_size, image_size), (
                f"{subj} image is {img.shape[-2:]}, expected ({image_size}, {image_size}). "
                "Re-run preprocessing."
            )
            assert lbl.shape[1:] == img.shape, (
                f"{subj} label/image shape mismatch: {lbl.shape} vs {img.shape}"
            )

            self.images[subj] = img
            self.labels[subj] = lbl

            for z in range(img.shape[0]):
                # has_ovary = any voxel in either L (ch 2) or R (ch 3)
                has_ov = bool(lbl[2, z].any() or lbl[3, z].any())
                self.index.append(SliceIndex(subject=subj, z=z, has_ovary=has_ov))

        if skipped:
            print(f"[D2SliceDataset/{split}] WARNING skipped (missing files): {skipped}")

        n_ov = sum(1 for s in self.index if s.has_ovary)
        n_subj = len(self.images)
        print(
            f"[D2SliceDataset/{split}] {n_subj} subjects loaded, "
            f"{len(self.index)} slices ({n_ov} with ovary, "
            f"{len(self.index) - n_ov} without)"
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        s = self.index[idx]
        img = self.images[s.subject][s.z]                  # (H, W)
        lbl = self.labels[s.subject][:, s.z]               # (C, H, W)

        # To [-1, 1] for diffusion (standard DDPM convention)
        img = img * 2.0 - 1.0

        img_t = torch.from_numpy(img).unsqueeze(0).float()      # (1, H, W)
        lbl_t = torch.from_numpy(lbl.astype(np.float32))        # (C, H, W)
        return {
            "image": img_t,
            "label": lbl_t,
            "subject": s.subject,
            "z": s.z,
            "has_ovary": s.has_ovary,
        }

    def make_weighted_sampler(self, ovary_weight: float) -> WeightedRandomSampler:
        """Oversample ovary-containing slices by `ovary_weight`x relative to others."""
        weights = torch.tensor(
            [ovary_weight if s.has_ovary else 1.0 for s in self.index],
            dtype=torch.double,
        )
        return WeightedRandomSampler(
            weights=weights, num_samples=len(weights), replacement=True
        )
