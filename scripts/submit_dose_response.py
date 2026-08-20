#!/usr/bin/env python3
"""
Exp 1 fan-out — retarget the exp1c_spade_fixed synth volumes to 6 new t
values, then submit 6 t x 3 seeds = 18 RAovSeg training jobs.

Run on the HPC login node (has access to /mnt/parscratch synth_volumes).

Steps per t value:
  1. Skip if the retargeted synth dir already exists (idempotent — re-runs
     the coordinator won't re-retarget).
  2. Call scripts/retarget_ovary_intensity.py to produce it.
  3. sbatch 3 seed jobs against it via run_raovseg_aug_dose_response.sh.

The t=0.26 condition (nominal middle of the enhancement window) is the
existing exp1c_spade_fixed baseline — skipped by default because you already
have those three seeds under `runs/raovseg_aug_spade_seed{0,1,2}` (see
metrics/master_metrics.csv). Pass --include-control if you want it re-run
on the same node/env as the sweep.

Usage:
    cd /mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset
    python scripts/submit_dose_response.py --dry-run     # inspect
    python scripts/submit_dose_response.py               # go
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


# t values from SIDE_EXPERIMENT_DESIGN.md §Exp 1. Middle row (0.26) is the
# existing baseline; the others are new.
T_VALUES = [0.05, 0.10, 0.16, 0.26, 0.36, 0.45, 0.60]
CONTROL_T = 0.26

SLURM_SCRIPT = Path(__file__).resolve().parent / "run_raovseg_aug_dose_response.sh"
RETARGET_SCRIPT = Path(__file__).resolve().parent / "retarget_ovary_intensity.py"


def t_tag(t: float) -> str:
    """0.05 -> '005', 0.26 -> '026', 0.60 -> '060'."""
    return f"{int(round(t * 100)):03d}"


def default_scratch_base() -> Path:
    """Guess the HPC scratch root from $USER. Overridable via CLI."""
    user = os.environ.get("USER", "unknown")
    return Path(f"/mnt/parscratch/users/{user}/synth_mri")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--t-values", type=float, nargs="+", default=None,
                        help=f"Override the t sweep (default: {T_VALUES})")
    parser.add_argument("--include-control", action="store_true",
                        help=f"Also submit t={CONTROL_T} (the existing baseline). "
                             "Skipped by default.")
    parser.add_argument("--scratch-base", type=Path, default=None,
                        help="HPC scratch root (default: /mnt/parscratch/users/$USER/synth_mri)")
    parser.add_argument("--src-synth-dir-name", default="exp1c_spade_fixed",
                        help="Name of the source synth dir under $scratch_base/synth_volumes "
                             "to retarget from (default: exp1c_spade_fixed)")
    parser.add_argument("--python", default="python",
                        help="Python interpreter to invoke the retarget script (default: python). "
                             "On Stanage login you may need `python` after `source activate synth_mri`.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not SLURM_SCRIPT.exists():
        print(f"ERROR: SLURM script not found at {SLURM_SCRIPT}", file=sys.stderr)
        return 1
    if not RETARGET_SCRIPT.exists():
        print(f"ERROR: retarget script not found at {RETARGET_SCRIPT}", file=sys.stderr)
        return 1

    scratch_base = args.scratch_base or default_scratch_base()
    synth_root = scratch_base / "synth_volumes"
    src_synth = synth_root / args.src_synth_dir_name
    if not args.dry_run and not src_synth.exists():
        print(f"ERROR: source synth dir not found: {src_synth}", file=sys.stderr)
        print("       Point --scratch-base at the right root, or --src-synth-dir-name "
              "at the right dir under synth_volumes/.", file=sys.stderr)
        return 1

    t_values = args.t_values if args.t_values is not None else T_VALUES
    if not args.include_control:
        t_values = [t for t in t_values if abs(t - CONTROL_T) > 1e-6]

    submitted = 0
    for t in t_values:
        tag = t_tag(t)
        out_synth = synth_root / f"{args.src_synth_dir_name}_t{tag}"

        # --- Retarget once per t (shared across seeds) ---
        if out_synth.exists():
            print(f"[dose] t={t:.2f} — {out_synth.name} already exists, skipping retarget")
        else:
            retarget_cmd = [
                args.python, str(RETARGET_SCRIPT),
                "--src-dir", str(src_synth),
                "--out-dir", str(out_synth),
                "--target-normalized", str(t),
            ]
            print(f"\n[dose] t={t:.2f} — retargeting → {out_synth}")
            print("  " + " ".join(retarget_cmd))
            if not args.dry_run:
                subprocess.check_call(retarget_cmd)

        # --- Fan out 3 seeds ---
        for seed in args.seeds:
            jobname = f"raov_dose_t{tag}_s{seed}"
            sbatch_cmd = [
                "sbatch",
                f"--export=ALL,T_TAG={tag},SEED={seed}",
                f"--job-name={jobname}",
                str(SLURM_SCRIPT),
            ]
            print("  " + " ".join(sbatch_cmd))
            if not args.dry_run:
                subprocess.check_call(sbatch_cmd)
                submitted += 1

    if args.dry_run:
        print(f"\n[dry-run] would retarget {len(t_values)} t-values and submit "
              f"{len(t_values) * len(args.seeds)} jobs")
    else:
        print(f"\nSubmitted {submitted} jobs across {len(t_values)} t-values.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
