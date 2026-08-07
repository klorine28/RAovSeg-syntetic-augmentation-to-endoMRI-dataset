#!/usr/bin/env python3
"""
Tier 1 sweep coordinator — batched Optuna + SLURM.

Runs on a login/interactive node (survives disconnect if launched under tmux
or screen). Owns the Optuna study on disk (SQLite). Each iteration:

    1. Ask Optuna for BATCH_SIZE new configs.
    2. For each config, sbatch 3 jobs (one per seed) via tier1_run_config.sh
       — env vars carry the sampled params into the sbatch script.
    3. Poll sacct until every submitted job reaches a terminal state.
    4. For each finished config, read metrics_ov.json and metrics_ut.json from
       every seed, compute mean DSC per target, and objective = min(mean_ov,
       mean_ut). Feed back to Optuna.
    5. Loop until TOTAL_TRIALS reached.

Sampler:
    QMC (10 warm-up trials) then TPE. Provides space-filling coverage before
    exploiting.

Resume:
    load_if_exists=True — killing and restarting this script picks up where it
    left off. In-flight trials (still queued in SLURM) will be re-parsed on
    restart if their metrics.json exists; otherwise Optuna's `ask` will just
    generate replacement suggestions.

Usage (from HPC login node, in tmux):
    cd /mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset
    python scripts/tier1_sweep_coordinator.py \\
        --total-trials 40 --batch-size 10 --n-seeds 3

Cost estimate:
    ~120 SLURM jobs total (40 trials × 3 seeds), ~4-6 GPU-h each ⇒
    ~500-700 GPU-hours.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import optuna
    from optuna.samplers import QMCSampler, TPESampler
except ImportError:
    sys.exit("optuna not installed. `pip install optuna` in the synth_mri env.")


TERMINAL_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
    "NODE_FAIL", "BOOT_FAIL", "OUT_OF_MEMORY", "PREEMPTED",
}


def build_sampler(n_warmup: int, seed: int = 42):
    """QMC warm-up then TPE — Bergstra-style refinement.

    Optuna doesn't have a native "swap samplers at trial N" primitive, so we
    return a TPESampler with n_startup_trials set to the QMC-warmup count.
    TPE's own initial-trials fallback samples uniformly rather than QMC, but
    for our 3+ continuous + 3 categorical mixed space that gap is small and
    it avoids maintaining two study handles.

    If space-filling matters more than tunability, swap in QMCSampler(
    scramble=True) instead and set n_startup_trials=0.
    """
    return TPESampler(n_startup_trials=n_warmup, seed=seed)


def suggest_params(trial: "optuna.Trial") -> dict:
    """Define the Tier 1 search space."""
    return {
        "ovary_target":       trial.suggest_float("ovary_target", 0.20, 0.32),
        "iscs_alpha":         trial.suggest_float("iscs_alpha", 0.0, 1.0),
        # Upper bound capped at 4.0 (not 8.0). Above ~4 px, the lowpassed
        # shared noise deviates far enough from white N(0, I) that the
        # DDPM (trained on white-noise reverse-diffusion) may produce
        # artefacts. 4 is enough to bias the shared component toward
        # low-frequency structure without breaking the noise assumption.
        "iscs_lowpass_sigma": trial.suggest_float("iscs_lowpass_sigma", 0.0, 4.0),
        "z_smooth_sigma":     trial.suggest_float("z_smooth_sigma", 0.0, 1.5),
        "body_mask":          trial.suggest_categorical("body_mask", [0, 1]),
        "hist_match":         trial.suggest_categorical("hist_match", [0, 1]),
        "skip_enh":           trial.suggest_categorical("skip_enh", [0, 1]),
    }


def sbatch_submit(script: Path, env_overrides: dict) -> int:
    """Submit one sbatch script with env overrides. Returns SLURM jobid.

    Stanage's SLURM defaults to --export=NONE for job scripts, so we must
    explicitly pass each variable via --export. Format is comma-separated
    KEY=VALUE, prefixed with ALL to also inherit non-SLURM shell env
    (module state, PATH, etc.).
    """
    env = os.environ.copy()
    for k, v in env_overrides.items():
        env[k] = str(v)
    export_pairs = ",".join(f"{k}={v}" for k, v in env_overrides.items())
    r = subprocess.run(
        ["sbatch", "--parsable", f"--export=ALL,{export_pairs}", str(script)],
        env=env, capture_output=True, text=True, check=True,
    )
    return int(r.stdout.strip().split(";")[0])


def sacct_states(jids: list[int]) -> dict[int, str]:
    """Return the latest SLURM state per jobid via sacct.

    We only care about the parent job (not steps or batch entries), so we
    strip anything after "." on the JobID column.
    """
    if not jids:
        return {}
    r = subprocess.run(
        ["sacct", "-X", "-j", ",".join(str(j) for j in jids),
         "--format=JobID,State", "--parsable2", "--noheader"],
        capture_output=True, text=True, check=True,
    )
    out: dict[int, str] = {}
    for line in r.stdout.splitlines():
        if "|" not in line:
            continue
        jid_field, state_field = line.split("|", 1)
        try:
            jid = int(jid_field.split(".")[0])
        except ValueError:
            continue
        state = state_field.split()[0]  # e.g. "CANCELLED by 12345"
        out[jid] = state
    return out


def wait_for_batch(jids: list[int], poll_seconds: int = 300) -> dict[int, str]:
    """Poll sacct until every jobid is in a terminal state. Returns final states."""
    remaining = set(jids)
    final: dict[int, str] = {}
    while remaining:
        states = sacct_states(list(remaining))
        for jid, st in states.items():
            if st in TERMINAL_STATES:
                final[jid] = st
                remaining.discard(jid)
        # sacct sometimes takes a moment to know about a very new job; be forgiving.
        if remaining:
            logging.info(f"  [wait] {len(remaining)}/{len(jids)} still running, sleeping {poll_seconds}s")
            time.sleep(poll_seconds)
    return final


def parse_trial_seed(sweep_root: Path, trial_id: int, seed: int) -> dict | None:
    """Read metrics_ov.json + metrics_ut.json from one (trial, seed) dir.

    Returns None if the DONE marker or either metrics file is missing.
    """
    base = sweep_root / f"trial_{trial_id}" / f"seed_{seed}"
    if not (base / "DONE").exists():
        return None
    try:
        m_ov = json.load((base / "metrics_ov.json").open())
        m_ut = json.load((base / "metrics_ut.json").open())
    except FileNotFoundError:
        return None

    def _get(m, mode, key):
        return m["aggregate"][mode][key]["mean"]

    return {
        "ov_dsc": _get(m_ov, "full", "dsc"),
        "ov_iou": _get(m_ov, "full", "iou"),
        "ov_hd95": _get(m_ov, "full", "hd95_mm"),
        "ut_dsc": _get(m_ut, "full", "dsc"),
        "ut_iou": _get(m_ut, "full", "iou"),
        "ut_hd95": _get(m_ut, "full", "hd95_mm"),
    }


def reconcile_orphan_trials(study: "optuna.Study", sweep_root: Path,
                            n_seeds: int) -> int:
    """On startup, find Optuna trials in RUNNING state (i.e. asked-for but not
    told-back) and either resolve them from on-disk metrics or mark them FAIL.

    An "orphan" trial is a study.ask() that never received study.tell() —
    happens when the coordinator was killed after submitting jobs but before
    the batch wait completed. Without this, restarting the coordinator would
    ask for NEW trials on top of the still-running old ones, wasting GPU-h.

    For each RUNNING trial:
        - If sweep_root/trial_<N>/aggregate.json exists, the trial completed
          on a previous coord run; re-tell the objective and mark COMPLETE.
        - If all n_seeds seed dirs have DONE files (parseable metrics), we
          can compute and tell.
        - Otherwise the trial's SLURM jobs are still in flight or died
          silently; mark FAIL so Optuna moves on. (User can manually
          resubmit if they want to keep the data.)

    Returns the number of orphan trials resolved.
    """
    running = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
    if not running:
        return 0

    logging.warning(f"reconciling {len(running)} RUNNING trial(s) from previous coordinator run")
    resolved = 0
    seeds = list(range(n_seeds))
    for t in running:
        # study.tell() takes the trial NUMBER (int), not a FrozenTrial object.
        # Also, `objective_min_dsc` may be null in aggregate.json when the
        # coordinator used --objective mean|weighted; fall back to objective_value.
        trial_num = t.number
        agg_path = sweep_root / f"trial_{trial_num}" / "aggregate.json"
        if agg_path.exists():
            try:
                agg = json.loads(agg_path.read_text())
                obj = agg.get("objective_value")
                if obj is None:
                    obj = agg.get("objective_min_dsc")
                if obj is None:
                    raise ValueError("no objective_value or objective_min_dsc in aggregate.json")
                study.tell(trial_num, float(obj))
                logging.info(f"  trial {trial_num}: resolved from aggregate.json → objective={float(obj):.4f}")
                resolved += 1
                continue
            except Exception as e:
                logging.warning(f"  trial {trial_num}: aggregate.json unreadable ({e}); trying seed metrics")

        agg = aggregate_trial(sweep_root, trial_num, seeds)
        if agg is not None:
            obj = min(agg["ov_dsc_mean"], agg["ut_dsc_mean"])
            study.tell(trial_num, obj)
            logging.info(f"  trial {trial_num}: resolved from seed metrics → objective={obj:.4f}")
            resolved += 1
            continue

        # Nothing usable on disk — mark FAIL so Optuna can move on
        study.tell(trial_num, state=optuna.trial.TrialState.FAIL)
        logging.warning(f"  trial {trial_num}: no usable metrics on disk; marked FAIL")

    return resolved


def warm_start_from_old_study(new_study: "optuna.Study", old_db_path: Path,
                              old_study_name: str, top_n: int,
                              param_bounds: dict) -> int:
    """Enqueue the top-N COMPLETE trials from an OLD study into a new study.

    Reads old study db, sorts trials by objective (higher = better), takes
    top-N by params (not values — values are discarded because they were
    measured against a different generator checkpoint), and enqueues them
    on the new study. The new sampler will test these params first before
    inventing new ones.

    Filters out any trial whose params fall outside the CURRENT search
    space (e.g. the old sweep had iscs_lowpass_sigma up to 8.0 but we
    later capped at 4.0). Skipped trials are logged.

    Returns the number of trials successfully enqueued.
    """
    if not old_db_path.exists():
        raise FileNotFoundError(f"old study db not found at {old_db_path}")

    old_storage = f"sqlite:///{old_db_path}"
    old_study = optuna.load_study(study_name=old_study_name, storage=old_storage)
    completes = [t for t in old_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completes.sort(key=lambda t: t.value if t.value is not None else -float("inf"), reverse=True)

    logging.info(f"warm-start: found {len(completes)} complete trials in old study; "
                 f"selecting top {top_n} by objective")

    enqueued = 0
    for t in completes[:top_n]:
        # Check every param is within the current search-space bounds
        params = t.params
        oob = []
        for k, (lo, hi) in param_bounds.items():
            v = params.get(k)
            if v is None:
                continue
            if isinstance(v, (int, float)) and not (lo <= v <= hi):
                oob.append(f"{k}={v} not in [{lo}, {hi}]")
        if oob:
            logging.warning(f"  skip old trial #{t.number} (obj={t.value:.4f}): "
                            f"out-of-bounds — {'; '.join(oob)}")
            continue
        new_study.enqueue_trial(params, skip_if_exists=False)
        logging.info(f"  enqueued from old trial #{t.number} (old obj={t.value:.4f}): {params}")
        enqueued += 1
    logging.info(f"warm-start: {enqueued} trials enqueued")
    return enqueued


def check_early_stop(study: "optuna.Study", patience: int, delta: float,
                     min_trials: int) -> bool:
    """Return True if the sweep has stopped improving.

    Rule: after at least `min_trials` COMPLETE trials, if the best objective
    hasn't improved by more than `delta` over the last `patience` trials,
    return True. Only considers COMPLETE trials in chronological trial-number
    order.
    """
    completes = sorted(
        (t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
         and t.value is not None),
        key=lambda t: t.number,
    )
    if len(completes) < min_trials:
        return False

    values = [t.value for t in completes]
    best_now = max(values)
    best_before_patience = max(values[:-patience]) if len(values) > patience else values[0]
    improvement = best_now - best_before_patience
    if improvement < delta:
        logging.warning(
            f"early stop: best objective {best_now:.4f} improved by only "
            f"{improvement:+.4f} vs the state {patience} trials ago "
            f"({best_before_patience:.4f}); delta threshold = {delta}. Stopping."
        )
        return True
    return False


def aggregate_trial(sweep_root: Path, trial_id: int, seeds: list[int]) -> dict | None:
    """Mean each metric across seeds. None if no seed produced usable metrics."""
    per_seed = [parse_trial_seed(sweep_root, trial_id, s) for s in seeds]
    per_seed = [p for p in per_seed if p is not None]
    if not per_seed:
        return None

    def _mean(k):
        vals = [p[k] for p in per_seed if p[k] is not None and not np.isnan(p[k])]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "n_seeds": len(per_seed),
        "ov_dsc_mean": _mean("ov_dsc"),
        "ut_dsc_mean": _mean("ut_dsc"),
        "ov_iou_mean": _mean("ov_iou"),
        "ut_iou_mean": _mean("ut_iou"),
        "ov_hd95_mean": _mean("ov_hd95"),
        "ut_hd95_mean": _mean("ut_hd95"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-db", type=Path,
                    default=Path("sweep/tier1_study.db"),
                    help="SQLite path for the Optuna study.")
    ap.add_argument("--study-name", type=str, default="tier1_synth_params")
    ap.add_argument("--sweep-root", type=Path,
                    default=Path("/mnt/parscratch/users") / os.environ.get("USER", "user") / "synth_mri" / "sweep" / "tier1",
                    help="Where the sbatch template writes per-trial dirs.")
    ap.add_argument("--sbatch-script", type=Path,
                    default=Path("scripts/tier1_run_config.sh"))
    ap.add_argument("--total-trials", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--n-seeds", type=int, default=3,
                    help="Seeds per trial. 3 is the practical minimum; the "
                         "project's own variance study (Phase 1 §4.4) showed "
                         "n=3 means overstated the true effect by ~22%% vs "
                         "n=8. Consider --n-seeds 5 if compute allows (~1.7x cost).")
    ap.add_argument("--warmup-trials", type=int, default=10,
                    help="How many trials to sample uniformly before TPE takes over.")
    ap.add_argument("--objective", type=str, default="min",
                    choices=("min", "mean", "weighted"),
                    help="How to combine per-target DSC into the Optuna objective: "
                         "'min' = min(ov_dsc, ut_dsc) — robust to weak target; "
                         "'mean' = (ov + ut) / 2 — equal weight; "
                         "'weighted' = 0.6 * ov + 0.4 * ut — bias toward ovary "
                         "(the harder target). Default 'min'; switch to 'mean' or "
                         "'weighted' if smoke trials show one target consistently "
                         "dominates (making 'min' degenerate into single-target opt).")
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true",
                    help="Suggest params and print what would be submitted; do not sbatch.")

    # Warm-start options (seed the new sweep with params from a previous sweep)
    ap.add_argument("--warm-start-from", type=Path, default=None,
                    help="Path to an OLD Optuna study db (SQLite). Its top-N params "
                         "will be enqueued in the new study before TPE explores. "
                         "Values are discarded — only params are transferred.")
    ap.add_argument("--warm-start-study-name", type=str, default="tier1_synth_params",
                    help="Name of the study inside --warm-start-from.")
    ap.add_argument("--warm-start-top-n", type=int, default=10,
                    help="How many top-scoring old trials to enqueue.")

    # Early stopping — stop when best objective plateaus
    ap.add_argument("--early-stop-patience", type=int, default=10,
                    help="Stop if best objective hasn't improved by --early-stop-delta "
                         "in the last N trials. 0 disables early stopping.")
    ap.add_argument("--early-stop-delta", type=float, default=0.005,
                    help="Minimum objective improvement over --early-stop-patience "
                         "trials to keep the sweep going. Default 0.005 DSC.")
    ap.add_argument("--early-stop-min-trials", type=int, default=15,
                    help="Don't consider stopping until at least this many trials complete. "
                         "Prevents stopping during warm-start / early exploration.")

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Dry-run stays in-memory so it can't corrupt the persistent study
    # (a killed dry-run would otherwise leave FAILed trials on disk).
    if args.dry_run:
        storage = None
        logging.info("[dry-run] using in-memory study; disk study untouched")
    else:
        args.study_db.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{args.study_db}"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=build_sampler(args.warmup_trials),
        load_if_exists=True,
    )
    n_done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    logging.info(f"loaded study: {len(study.trials)} trials on disk, {n_done} complete")

    seeds = list(range(args.n_seeds))
    if args.n_seeds < 5:
        logging.warning(
            f"n_seeds={args.n_seeds} is likely noisy. Project variance study "
            f"(Phase 1 §4.4) found n=3 means overstated the true effect by "
            f"~22%% vs n=8. TPE will chase noise for the first ~15 trials."
        )
    if not args.dry_run:
        args.sweep_root.mkdir(parents=True, exist_ok=True)
        # Resolve any RUNNING trials from a previous coordinator run
        # (crash recovery — see reconcile_orphan_trials docstring).
        resolved = reconcile_orphan_trials(study, args.sweep_root, args.n_seeds)
        if resolved > 0:
            n_done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
            logging.info(f"after reconciliation: {n_done} complete")

    # Warm-start: enqueue top-N params from an old study before TPE explores.
    # Only fires once per db (skips if the new study already has any trials).
    if args.warm_start_from is not None and len(study.trials) == 0:
        # Current search-space bounds (must match suggest_params). Used to
        # filter out old trials whose params fall outside the current range.
        param_bounds = {
            "ovary_target":       (0.20, 0.32),
            "iscs_alpha":         (0.0, 1.0),
            "iscs_lowpass_sigma": (0.0, 4.0),
            "z_smooth_sigma":     (0.0, 1.5),
        }
        warm_start_from_old_study(
            study, args.warm_start_from, args.warm_start_study_name,
            args.warm_start_top_n, param_bounds,
        )
    elif args.warm_start_from is not None:
        logging.info(f"warm-start requested but study already has {len(study.trials)} "
                     f"trials — skipping to avoid duplicate enqueue")

    while n_done < args.total_trials:
        remaining = args.total_trials - n_done
        this_batch = min(args.batch_size, remaining)
        logging.info(f"=== batch: {this_batch} trials (of {remaining} remaining) ===")

        ask_pack: list[tuple[optuna.Trial, dict]] = []
        for _ in range(this_batch):
            t = study.ask()
            params = suggest_params(t)
            ask_pack.append((t, params))
            logging.info(f"  suggest trial {t.number}: {params}")

        if args.dry_run:
            logging.info("[dry-run] not submitting; exiting after one batch")
            return

        submitted: list[tuple[int, int, int]] = []  # (jobid, trial_id, seed)
        for t, params in ask_pack:
            for seed in seeds:
                env = {
                    "TRIAL_ID": t.number,
                    "SEED": seed,
                    "OVARY_TARGET": params["ovary_target"],
                    "ISCS_ALPHA": params["iscs_alpha"],
                    "ISCS_LOWPASS_SIGMA": params["iscs_lowpass_sigma"],
                    "Z_SMOOTH_SIGMA": params["z_smooth_sigma"],
                    "BODY_MASK": params["body_mask"],
                    "HIST_MATCH": params["hist_match"],
                    "SKIP_ENH": params["skip_enh"],
                }
                jid = sbatch_submit(args.sbatch_script, env)
                submitted.append((jid, t.number, seed))
        logging.info(f"submitted {len(submitted)} jobs; polling sacct every {args.poll_seconds}s")

        final_states = wait_for_batch([j for j, _, _ in submitted], args.poll_seconds)

        # Report each trial back to Optuna based on aggregated per-seed metrics
        for t, params in ask_pack:
            trial_seeds = [seed for jid, tid, seed in submitted if tid == t.number]
            agg = aggregate_trial(args.sweep_root, t.number, trial_seeds)
            if agg is None:
                logging.warning(f"trial {t.number}: no usable metrics; marking FAIL")
                study.tell(t, state=optuna.trial.TrialState.FAIL)
                continue
            ov, ut = agg["ov_dsc_mean"], agg["ut_dsc_mean"]
            if args.objective == "min":
                objective = min(ov, ut)
            elif args.objective == "mean":
                objective = 0.5 * (ov + ut)
            elif args.objective == "weighted":
                objective = 0.6 * ov + 0.4 * ut
            else:
                raise ValueError(f"unknown --objective: {args.objective}")

            # Degeneration warning: if 'min' picks the same target consistently,
            # the sweep is effectively single-target.
            if args.objective == "min":
                gap = abs(ov - ut)
                if gap > 0.10:
                    logging.warning(
                        f"trial {t.number}: |ov - ut| = {gap:.3f} > 0.10 — "
                        f"consider --objective mean or weighted if this persists"
                    )
            # Persist per-trial aggregate summary next to the trial dir.
            trial_final_states = {
                int(jid): final_states.get(int(jid), "UNKNOWN")
                for jid, tid, _ in submitted if tid == t.number
            }
            summary_path = args.sweep_root / f"trial_{t.number}" / "aggregate.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps({
                "trial": t.number,
                "params": params,
                "n_seeds": agg["n_seeds"],
                **agg,
                "objective_mode": args.objective,
                "objective_value": objective,
                # keep the old key name for backwards compat with any reader
                # that was already looking for it
                "objective_min_dsc": objective if args.objective == "min" else None,
                "final_states": trial_final_states,
            }, indent=2, default=str))
            logging.info(
                f"trial {t.number}: ov_dsc={ov:.4f} ut_dsc={ut:.4f} "
                f"objective[{args.objective}]={objective:.4f}"
            )
            study.tell(t, objective)

        n_done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        logging.info(f"progress: {n_done}/{args.total_trials} complete")

        # Early stopping check — only after a full batch has landed so the
        # trial-number ordering is meaningful.
        if args.early_stop_patience > 0 and check_early_stop(
            study, args.early_stop_patience, args.early_stop_delta,
            args.early_stop_min_trials,
        ):
            logging.info(f"=== early stopping triggered at {n_done} trials ===")
            break

    logging.info("=== sweep complete ===")
    best = study.best_trial
    logging.info(f"best trial: #{best.number}  objective={best.value:.4f}  params={best.params}")


if __name__ == "__main__":
    main()
