#!/usr/bin/env python3
"""
Exp 2 fan-out — submit the 5 band conditions x 3 seeds = 15 SLURM jobs.

Skips the published-control condition [0.22, 0.30] by default because that
condition is identical to Exp 0a and you already have it in
`hpc_pulled/`. Pass --include-control to run it anyway (useful if you want
a fresh baseline on the same node/env as the sweep).

Usage (from HPC login node):
    cd /mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset
    python scripts/submit_band_sweep.py                 # 4 conditions x 3 seeds = 12 jobs
    python scripts/submit_band_sweep.py --dry-run       # print, don't submit
    python scripts/submit_band_sweep.py --include-control  # 15 jobs

Cost: 12 jobs x ~35 min ~= 7 GPU-hours (matches Exp 2 estimate in the design doc).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# (o1, o2, skip_enh) — skip_enh=True means disable enhancement entirely.
CONDITIONS = [
    (0.22, 0.30, False),   # published control — skipped by default
    (0.15, 0.25, False),
    (0.30, 0.42, False),
    (0.42, 0.56, False),
    (0.00, 0.00, True),    # no enhancement (floor)
]

SLURM_SCRIPT = Path(__file__).resolve().parent / "run_raovseg_band_sweep.sh"


def build_sbatch_cmd(o1: float, o2: float, seed: int, skip_enh: bool) -> list[str]:
    exports = [f"ALL,SEED={seed}"]
    if skip_enh:
        exports.append("SKIP_ENH=1")
    else:
        exports.append(f"O1={o1}")
        exports.append(f"O2={o2}")
    export_str = ",".join(exports)
    # SLURM job name — surfaces the condition in squeue output.
    if skip_enh:
        jobname = f"raov_band_noenh_s{seed}"
    else:
        jobname = f"raov_band_o{int(round(o1*100)):03d}_o{int(round(o2*100)):03d}_s{seed}"
    return [
        "sbatch",
        f"--export={export_str}",
        f"--job-name={jobname}",
        str(SLURM_SCRIPT),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--include-control", action="store_true",
                        help="Also submit the [0.22, 0.30] published band. "
                             "By default we skip it because Exp 0a already gives "
                             "you three seeds under that band.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the sbatch commands, don't submit")
    args = parser.parse_args()

    if not SLURM_SCRIPT.exists():
        print(f"ERROR: SLURM script not found at {SLURM_SCRIPT}", file=sys.stderr)
        return 1

    conditions = [c for c in CONDITIONS
                  if args.include_control or c != (0.22, 0.30, False)]

    submitted = 0
    for o1, o2, skip in conditions:
        for seed in args.seeds:
            cmd = build_sbatch_cmd(o1, o2, seed, skip)
            print(" ".join(cmd))
            if not args.dry_run:
                subprocess.check_call(cmd)
                submitted += 1
    if args.dry_run:
        print(f"\n[dry-run] would submit {len(conditions) * len(args.seeds)} jobs")
    else:
        print(f"\nSubmitted {submitted} jobs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
