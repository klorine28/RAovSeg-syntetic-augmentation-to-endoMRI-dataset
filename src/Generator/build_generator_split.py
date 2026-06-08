"""
Build the generator's train/test split from the RAovSeg manifest.

Source of truth: data/processed/manifest.csv (written by src/preprocess.py).
Columns: subject_id, sequence, has_ov, has_cy, has_em, split, included, reason

Generator split rules (option (c) from the design discussion):
    train = (RAovSeg's 30 train_val subjects) ∪ (all subjects with has_em==1)
            minus any that overlap with RAovSeg's test set
    test  = RAovSeg's test set, IDENTICAL (sacred, never trained on)

Why include the endometrioma subjects in train:
    The 30 RAovSeg-aligned subjects have has_em=0, so channel 4 of the
    generator's label tensor would always be empty. Adding the 11 endo
    subjects gives the generator examples for the endometrioma channel,
    which matters for the SPADE conditioning in Exp 1b/1c. Doing this here
    rather than later means the same training data is used across 1a/1b/1c,
    keeping the ablation clean.

Output:
    splits/d2_generator_split.json
    {
        "train":  ["D2-001", ...],
        "test":   ["D2-005", ...],
        "_meta": {
            "source_manifest": "...",
            "n_train": N,
            "n_test": 8,
            "n_with_em_in_train": K,
            "raovseg_train_subjects": [...],
            "endo_subjects_added": [...],
            "endo_subjects_dropped_in_test": [...]
        }
    }

Usage:
    python -m src.Generator.build_generator_split \
        --manifest data/processed/manifest.csv \
        --raw_root UT-EndoMRI/D2_TCPW \
        --out_file splits/d2_generator_split.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


# Files the generator preprocess REQUIRES per subject. Endometrioma (`em`) is
# optional — channel 4 just stays empty when the file is absent. Cyst is
# unused by the generator entirely.
REQUIRED_SUFFIXES = ["T2FS", "ut", "ov"]


def missing_required_files(raw_root: Path, subject_id: str) -> list[str]:
    """Return the list of REQUIRED suffixes that don't exist on disk for this
    subject. Empty list means subject is fully equipped for preprocessing."""
    subj_dir = raw_root / subject_id
    return [
        suffix for suffix in REQUIRED_SUFFIXES
        if not (subj_dir / f"{subject_id}_{suffix}.nii.gz").exists()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to RAovSeg manifest.csv")
    parser.add_argument("--raw_root", required=True, help="Root of UT-EndoMRI/D2_TCPW")
    parser.add_argument("--out_file", required=True, help="Output split JSON path")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    raw_root = Path(args.raw_root)
    out_path = Path(args.out_file)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run src/preprocess.py first."
        )

    df = pd.read_csv(manifest_path)
    required_cols = {"subject_id", "split", "included", "has_em"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    # RAovSeg's splits
    raovseg_train = sorted(
        df[(df["included"] == 1) & (df["split"] == "train_val")]["subject_id"].tolist()
    )
    raovseg_test = sorted(
        df[(df["included"] == 1) & (df["split"] == "test")]["subject_id"].tolist()
    )

    # All subjects with endometriomas. Read from disk to be authoritative —
    # don't trust manifest if disk says different.
    em_files_on_disk = sorted(raw_root.glob("*/D2-*_em.nii.gz"))
    em_subjects_disk = sorted({p.parent.name for p in em_files_on_disk})

    em_subjects_manifest = sorted(df[df["has_em"] == 1]["subject_id"].tolist())
    if set(em_subjects_disk) != set(em_subjects_manifest):
        only_disk = set(em_subjects_disk) - set(em_subjects_manifest)
        only_manifest = set(em_subjects_manifest) - set(em_subjects_disk)
        print(
            "[WARN] manifest/disk disagree on endometrioma subjects:\n"
            f"  on disk only: {sorted(only_disk)}\n"
            f"  in manifest only: {sorted(only_manifest)}\n"
            "  Using DISK as source of truth."
        )
    em_subjects = em_subjects_disk

    # Build generator train set: RAovSeg train ∪ endo subjects, minus test
    test_set = set(raovseg_test)
    train_candidates = set(raovseg_train) | set(em_subjects)

    endo_dropped_in_test = sorted(set(em_subjects) & test_set)
    if endo_dropped_in_test:
        print(
            f"[INFO] {len(endo_dropped_in_test)} endo subject(s) overlap with "
            f"RAovSeg's test set and will NOT be added to train: "
            f"{endo_dropped_in_test}"
        )

    # Drop train candidates that lack any required file. Without this the
    # generator preprocess would attempt them, skip mid-run, and leave the
    # split file claiming a higher n_train than actually got preprocessed.
    train_dropped: dict[str, list[str]] = {}
    for sid in sorted(train_candidates):
        miss = missing_required_files(raw_root, sid)
        if miss:
            train_dropped[sid] = miss
    if train_dropped:
        print(f"[INFO] dropping {len(train_dropped)} train candidate(s) "
              f"missing required files (T2FS/ut/ov):")
        for sid, miss in train_dropped.items():
            print(f"  {sid}: missing {miss}")
        train_candidates -= set(train_dropped.keys())

    # Same check for the test set, but don't drop — RAovSeg's test set is
    # sacred. Just warn loudly so we know about it.
    test_warnings: dict[str, list[str]] = {}
    for sid in sorted(test_set):
        miss = missing_required_files(raw_root, sid)
        if miss:
            test_warnings[sid] = miss
    if test_warnings:
        print(f"[WARN] {len(test_warnings)} test subject(s) missing required "
              f"files (keeping anyway, test set is sacred):")
        for sid, miss in test_warnings.items():
            print(f"  {sid}: missing {miss}")

    gen_train = sorted(train_candidates - test_set)
    gen_test = sorted(test_set)

    endo_added = sorted(set(em_subjects) - set(raovseg_train) - test_set
                        - set(train_dropped.keys()))

    n_em_in_train = sum(1 for s in gen_train if s in set(em_subjects))

    out = {
        "train": gen_train,
        "test": gen_test,
        "_meta": {
            "source_manifest": str(manifest_path),
            "raw_root": str(raw_root),
            "required_suffixes": REQUIRED_SUFFIXES,
            "n_train": len(gen_train),
            "n_test": len(gen_test),
            "n_with_em_in_train": n_em_in_train,
            "raovseg_train_subjects": raovseg_train,
            "endo_subjects_on_disk": em_subjects,
            "endo_subjects_added_to_train": endo_added,
            "endo_subjects_dropped_in_test": endo_dropped_in_test,
            "train_subjects_dropped_missing_files": train_dropped,
            "test_subjects_missing_files_kept": test_warnings,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[split] wrote {out_path}")
    print(f"[split]   generator train: {len(gen_train)} subjects "
          f"({n_em_in_train} with endometrioma)")
    print(f"[split]   generator test:  {len(gen_test)} subjects (same as RAovSeg)")
    print(f"[split]   endo subjects added beyond RAovSeg train: {len(endo_added)}")
    print(f"[split]   train candidates dropped (missing files): {len(train_dropped)}")


if __name__ == "__main__":
    main()
