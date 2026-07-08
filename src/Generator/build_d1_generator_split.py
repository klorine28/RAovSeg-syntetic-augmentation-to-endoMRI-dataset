"""
Build the D1 train/val split for the Phase 2 generator.

D1_MHS has no RAovSeg manifest (D1 is not used in RAovSeg train/test), so the
split logic is much simpler than the D2 case:

  1. Enumerate subjects on disk in raw_root (D1-XXX).
  2. Keep subjects that have T2 + uterus (any rater) + ovary (any rater).
  3. Hold out a small val set for periodic sample generation during training;
     everything else goes into train.

The val set is NOT a sacred test set (D1 subjects are never used in RAovSeg
evaluation), just a small pool used by `inference_validate.py` for sample
grids during training.

Output layout matches d2_generator_split.json:
    {
        "train": ["D1-000", ...],
        "test":  ["D1-023", ...],   # named "test" for dataset.py compatibility
        "_meta": {...}
    }

Usage:
    python -m src.Generator.build_d1_generator_split \
        --raw_root UT-EndoMRI/D1_MHS \
        --out_file data/splits/d1_generator_split.json \
        --n_val 5 \
        --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


SUBJECT_DIR_RE = re.compile(r"^D1-\d{3}$")


def _has_any_rater(subj_dir: Path, subj_id: str, organ: str) -> bool:
    for r in ("r1", "r2", "r3"):
        if (subj_dir / f"{subj_id}_{organ}_{r}.nii.gz").exists():
            return True
    return False


def _missing_required(subj_dir: Path, subj_id: str) -> list[str]:
    """List of required-file suffix labels that are absent for this subject."""
    missing = []
    if not (subj_dir / f"{subj_id}_T2.nii.gz").exists():
        missing.append("T2")
    if not _has_any_rater(subj_dir, subj_id, "ut"):
        missing.append("ut_r*")
    if not _has_any_rater(subj_dir, subj_id, "ov"):
        missing.append("ov_r*")
    return missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True,
                        help="Root of UT-EndoMRI/D1_MHS")
    parser.add_argument("--out_file", required=True,
                        help="Output split JSON path")
    parser.add_argument("--n_val", type=int, default=5,
                        help="Held-out subjects for validation sampling (default 5)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Shuffle seed for the train/val partition (default 0)")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_path = Path(args.out_file)

    all_subjs = sorted(
        p.name for p in raw_root.iterdir()
        if p.is_dir() and SUBJECT_DIR_RE.match(p.name)
    )
    print(f"[split] found {len(all_subjs)} D1 subject dirs")

    usable, dropped = [], {}
    for subj in all_subjs:
        miss = _missing_required(raw_root / subj, subj)
        if miss:
            dropped[subj] = miss
        else:
            usable.append(subj)

    if dropped:
        print(f"[split] dropping {len(dropped)} subject(s) missing required files:")
        for s, m in dropped.items():
            print(f"  {s}: missing {m}")
    print(f"[split] {len(usable)} usable D1 subjects")

    if len(usable) < args.n_val + 1:
        raise SystemExit(
            f"only {len(usable)} usable subjects; can't hold out {args.n_val} for val"
        )

    rng = random.Random(args.seed)
    shuffled = usable[:]
    rng.shuffle(shuffled)
    val = sorted(shuffled[:args.n_val])
    train = sorted(shuffled[args.n_val:])

    out = {
        "train": train,
        "test": val,   # named "test" for dataset.py compatibility
        "_meta": {
            "raw_root": str(raw_root),
            "cohort": "D1",
            "n_train": len(train),
            "n_test": len(val),
            "seed": args.seed,
            "n_val_requested": args.n_val,
            "subjects_dropped_missing_files": dropped,
            "note": (
                "D1 has per-rater masks; preprocess_for_generator.py falls "
                "back r1 → r2 → r3. Subjects without any ovary rater are "
                "excluded here (Decision B1)."
            ),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[split] wrote {out_path}")
    print(f"[split]   train: {len(train)} subjects")
    print(f"[split]   test (val pool): {len(val)} subjects — {val}")


if __name__ == "__main__":
    main()
