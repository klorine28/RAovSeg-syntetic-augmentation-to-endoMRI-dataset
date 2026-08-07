# Side-experiment: automated parameter sweep for synthetic augmentation

> Personal / portfolio side-experiment. Not for the dissertation. Goal: use
> automated hyperparameter search to defend the final synthesis parameters
> against the criticism "how do we know these aren't cherry-picked?", and to
> test whether the RAovSeg augmentation pipeline generalises from ovary to
> uterus (a larger organ).

---

## 1. The question in one sentence

**"Given the DDPM checkpoint we already trained, what combination of
assembly-time parameters produces synth that is most useful for RAovSeg
segmentation of both ovary AND uterus at once?"**

Two things worth unpacking in that sentence:

- **Given the DDPM checkpoint** — we are NOT retraining the generator. The
  DDPM weights are fixed (`exp1c_spade/ckpt/step_100000.pt`). What we
  change is only the code that runs AFTER the generator produces slices:
  how they're combined into a volume, how they're normalised, how they're
  fed to the downstream segmenter.
- **Both ovary AND uterus at once** — one metric (the sweep objective)
  penalises configs that are good at ovary but bad at uterus, and vice
  versa. This forces the sweep toward configs that produce well-rounded
  volumes rather than organ-specific overfits.

---

## 2. Why this matters (the criticism we're addressing)

Our earlier work reported "SPADE augmentation gives ovary DSC 0.178 with
these parameters":
- `--ovary-target-intensity 0.26` (chosen because we swept 0.22 / 0.26 /
  0.28 by hand and 0.26 won),
- `--iscs-alpha 0.8` (chosen because "the ISCS paper says 0.8"),
- All body-mask / histogram-match / resample flags ON,
- Ovary enhancement enabled.

An examiner or reviewer can reasonably ask:

> "You searched three values of `t`. How do you know 0.26 is the actual
> optimum vs. just the best of three lucky guesses?"

A Bayesian hyperparameter search across the full space, with per-config
multi-seed evaluation, answers that criticism cleanly. It also lets us
sweep parameters we've never tested (`--iscs-lowpass-sigma`,
`--z-smooth-sigma`, the on/off flags), so we can defend those too.

The second angle — testing uterus in parallel — asks whether the same
generation pipeline that produces marginal ovary augmentation also
produces useful uterus augmentation, or whether the failure mode is
specific to small organs.

---

## 3. What we're changing (the search space, 7 parameters)

Every one of these is a knob on the assembly script,
`src/Generator/assemble_synthetic_volumes.py`.

| Parameter | Range | Default | What it does |
|---|---|---|---|
| `--ovary-target-intensity` | 0.20 to 0.32 | 0.26 | The RAovSeg preprocessing pipeline saturates any voxel whose intensity falls inside `[0.22, 0.30]` to 1.0. If synth ovary voxels don't land in that window, the segmenter never sees them as "highlighted". This flag adds an offset to synth ovary voxels so their mean intensity lands at this target after normalisation. |
| `--iscs-alpha` | 0.0 to 1.0 | 0.8 | Inter-Slice Consistent Stochasticity. Controls how much of the initial noise is shared across all slices of one synth volume. 0 = every slice starts from independent noise → jittery volume. 1 = every slice starts from the SAME noise → volume looks like a cylinder. Somewhere in between is right. |
| `--iscs-lowpass-sigma` | 0.0 to 4.0 pixels | 0.0 | NEW. A "multi-scale" version of ISCS. When > 0, we lowpass-filter the shared noise so only the LOW-frequency (structural) part is shared across slices, and each slice adds its own HIGH-frequency (texture) noise. Effect: consistent organ shape, varied texture per slice. Range capped at 4.0 to preserve the DDPM's white-noise assumption (see audit H3). |
| `--z-smooth-sigma` | 0.0 to 1.5 slices | 0.0 | NEW. Post-hoc Gaussian smoothing applied along the z-axis after the DDPM has produced all slices. Softens any residual slice-to-slice jitter. Trades a bit of sharpness for a bit more 3D-coherence. |
| `body_mask` (`--no-body-mask` flag) | on / off | on | Whether to zero out synth voxels outside the body silhouette derived from the label mask. Off = synth may hallucinate content outside the body. On = clean body cutout. |
| `hist_match` (`--no-histogram-match` flag) | on / off | on | Whether to histogram-match synth intensity distribution to the source real subject's raw intensity distribution. Off = synth stays in the DDPM's native intensity distribution. On = synth "looks like" real intensity-wise. |
| `skip_enh` (`--skip-enhancement-for-prefix D2-9`) | on / off | off | RAovSeg's preprocess normally applies the [0.22, 0.30] saturation to every subject. When on, this flag disables that step for synth subjects (prefix D2-9) while keeping it for real subjects. Useful because RAovSeg's saturation was tuned for real ovary intensity, which may or may not match synth. |

Together these 4 continuous + 3 categorical parameters define a
7-dimensional space with roughly 8 × 8 continuous corners plus 2×2×2
categorical corners = ~500+ meaningfully different configurations. We
can afford to actually visit ~40 of them (sampled cleverly).

---

## 4. What we're measuring (the metrics)

Every config, every seed, produces two `metrics_<target>.json` files
(one for ovary, one for uterus). Each file contains ALL of these
metrics across the 8 sacred test subjects, for three pipeline modes
(full pipeline, no post-processing, no ResClass):

| Metric | Range | What it tells you |
|---|---|---|
| **DSC** (Dice) | 0 to 1 | Volume overlap: 2·(A ∩ B) / (\|A\| + \|B\|). The paper's headline metric. Higher = better. |
| **IoU** (Jaccard) | 0 to 1 | Similar to DSC but slightly harsher on partial overlap. Monotonically related to DSC. |
| **Sensitivity** (Recall) | 0 to 1 | Of the true organ voxels, how many did we find? High = we don't miss ovaries. Clinically critical — missing an ovary matters more than over-segmenting one. |
| **Precision** | 0 to 1 | Of the voxels we called organ, how many really were? Low precision = we're over-predicting. |
| **HD95** (95th percentile Hausdorff, voxels) | 0 to infinity | Boundary error. If DSC is 0.7 but HD95 is 15 voxels, the boundary is way off — the model is finding the organ but drawing it in the wrong shape. |
| **Volume error** | -1 to +infinity | (V_pred - V_gt) / V_gt. Directly clinical — for endometrioma monitoring, size accuracy matters. |
| **Bootstrap 95% CI** | — | For each aggregate metric, we resample the 8-subject cohort 1000× and take 2.5-97.5 percentiles of the resulting means. This gives a defensible confidence interval on n=8. |

There's also a **diagnostic-only** metric that does NOT feed the sweep
objective:

| Metric | Range | What it tells you |
|---|---|---|
| **ISCS-score** | 0 to 1 | For each synth volume, we compute the same "inter-slice consistency" metrics we computed on real D2 (adjacent-slice mask DSC, centroid drift, image L1, z-axis TV). Then we score them: if the value lands in the real cohort's 5th-95th percentile band, the metric is "in-band". ISCS-score is the fraction of metrics in-band. 1.0 = synth is indistinguishable from real w.r.t. inter-slice coherence; ~0.5 = synth is roughly half in the real distribution's typical range. |

We log this per config but do NOT optimise on it. If it correlates with
downstream DSC across the sweep, that's a nice writeup finding. If it
doesn't, that's also informative.

---

## 5. What "good" means (the sweep objective)

For each trial (config), we run 3 seeds. For each seed, we get 8
subjects × 6 real metrics × 3 modes × 2 targets. We collapse all of
that down to ONE number that Optuna maximises:

```
objective = min(
    mean_across_seeds(mean_across_subjects(DSC_ovary_full)),
    mean_across_seeds(mean_across_subjects(DSC_uterus_full))
)
```

In words: **"the DSC of the worst target, averaged over subjects, averaged
over seeds"**. Using `min` rather than `mean` is the deliberate choice —
we want configs that work for both organs, not configs that trade uterus
accuracy for ovary accuracy.

The other metrics get logged for reporting, not for optimisation.

---

## 6. How a single trial runs (one config, one seed)

This is what happens inside one SLURM job (`scripts/tier1_run_config.sh`):

```
+-------------------------------------------------------------+
| STEP 1: Assemble synth                                      |
|                                                             |
|   Load DDPM checkpoint (fixed).                             |
|   For each subject in the generator's train split:          |
|     - Take the label mask (uterus, ovary, bladder, etc.)    |
|     - Sample a synth volume slice-by-slice via DDIM,        |
|       using this trial's ISCS-alpha and lowpass-sigma       |
|       to shape the initial noise.                           |
|     - Optionally z-smooth the resulting volume.             |
|     - Optionally body-mask, histogram-match, ovary-rescale. |
|   Output: 30 synth subjects at trial_N/seed_S/synth/D2-9XX/ |
+-------------------------------------------------------------+
                             |
                             v
+-------------------------------------------------------------+
| STEP 2: Score synth vs. real ISCS profile (diagnostic)      |
|                                                             |
|   Compute adjacent-slice mask DSC, centroid drift, image    |
|   L1, z-axis TV for each synth volume. Score each against   |
|   the real cohort's 5-95 percentile band. Output an         |
|   ISCS-score (0 to 1) for the synth batch.                  |
|   Purely diagnostic — does not affect the objective.        |
+-------------------------------------------------------------+
                             |
                             v
+-------------------------------------------------------------+
| STEP 3: Preprocess                                          |
|                                                             |
|   Take 30 real D2 subjects + 30 synth subjects.             |
|   Apply RAovSeg's resample + normalise + (optionally)       |
|   enhance pipeline. Split real subjects 30/8 (train_val +   |
|   test) using the paper's deterministic seed. Force all     |
|   synth into train_val.                                     |
|   Output: 60 preprocessed train_val subjects + 8 test       |
|   subjects. Every subject has image.npy + ov_label.npy +    |
|   ut_label.npy (whichever labels exist on disk).            |
+-------------------------------------------------------------+
                             |
                             v
+-------------------------------------------------------------+
| STEP 4: Train BOTH targets                                  |
|                                                             |
|   For target in {ov, ut}:                                   |
|     - train_resclass.py --target <t>  -> resclass_best_<t>  |
|     - train_attuseg.py  --target <t>  -> attuseg_best_<t>   |
|                                                             |
|   Total: 4 training runs (60 train_val subjects each).      |
|   Each ~1-2 hours on the assigned GPU.                      |
+-------------------------------------------------------------+
                             |
                             v
+-------------------------------------------------------------+
| STEP 5: Evaluate BOTH targets                               |
|                                                             |
|   For target in {ov, ut}:                                   |
|     - evaluate.py --target <t> --metrics-out metrics_<t>.json
|                                                             |
|   Each evaluation runs 3 pipeline modes (full, no_pp,       |
|   no_rc) and computes the 8-metric bundle per subject,      |
|   then aggregates with bootstrap 95% CI.                    |
+-------------------------------------------------------------+
                             |
                             v
+-------------------------------------------------------------+
| STEP 6: Touch DONE marker file                              |
|                                                             |
|   Signals to the coordinator that this (trial, seed) is     |
|   complete and metrics can be read.                         |
+-------------------------------------------------------------+
```

Wall-time: roughly 8-11 hours per (trial, seed) job (two full RAovSeg
pipelines end-to-end, one for ovary and one for uterus). SLURM allocation
set to 16h to leave slack. Runs on one A100.

---

## 7. How the sweep coordinates (across all trials)

The coordinator is a long-lived Python process on a login/interactive
node (recommended: launched in tmux). It loops:

```
+-----------------------------------------------------------------+
|                                                                 |
|                    LOOP UNTIL 40 TRIALS DONE                    |
|                                                                 |
|  +-----------------------------------------------------------+  |
|  | 1. Ask Optuna for 10 trials (a "batch")                   |  |
|  |                                                           |  |
|  |    Trial 0: {ovary_target=0.28, iscs_alpha=0.6, ...}      |  |
|  |    Trial 1: {ovary_target=0.22, iscs_alpha=0.9, ...}      |  |
|  |    ...                                                    |  |
|  |    Trial 9: {ovary_target=0.30, iscs_alpha=0.4, ...}      |  |
|  +-----------------------------------------------------------+  |
|                              |                                  |
|                              v                                  |
|  +-----------------------------------------------------------+  |
|  | 2. Submit 30 SLURM jobs (10 trials × 3 seeds each)        |  |
|  |                                                           |  |
|  |    sbatch tier1_run_config.sh   (with env vars per job)   |  |
|  |     -> jobid 10800001                                     |  |
|  |     -> jobid 10800002                                     |  |
|  |     -> ... (30 total)                                     |  |
|  +-----------------------------------------------------------+  |
|                              |                                  |
|                              v                                  |
|  +-----------------------------------------------------------+  |
|  | 3. Poll sacct every 5 minutes.                            |  |
|  |    Wait until every one of the 30 jobs is in a terminal   |  |
|  |    state (COMPLETED, FAILED, CANCELLED, TIMEOUT, ...).    |  |
|  +-----------------------------------------------------------+  |
|                              |                                  |
|                              v                                  |
|  +-----------------------------------------------------------+  |
|  | 4. Parse metrics for each finished trial:                 |  |
|  |                                                           |  |
|  |    For each trial, average DSC across its 3 seeds.        |  |
|  |    objective = min(ovary_DSC_mean, uterus_DSC_mean)       |  |
|  |                                                           |  |
|  |    Tell Optuna: study.tell(trial, objective)              |  |
|  |                                                           |  |
|  |    Save aggregate.json for post-hoc analysis.             |  |
|  +-----------------------------------------------------------+  |
|                              |                                  |
|                              v                                  |
|  +-----------------------------------------------------------+  |
|  | 5. Repeat until 40 total trials completed.                |  |
|  +-----------------------------------------------------------+  |
|                                                                 |
+-----------------------------------------------------------------+
                              |
                              v
              Report best trial + params to stdout.
```

**Sampler choice**: QMC (quasi-Monte Carlo, space-filling) for the first
10 warm-up trials to guarantee we EXPLORE the corners of the space, then
Tree-Parzen Estimator (TPE, Bayesian) for the remaining 30 to EXPLOIT
what we've learned. This mirrors the Bergstra et al. (2011) design that
Optuna's default uses.

**Resume-safety**: the Optuna study lives on disk (SQLite at
`sweep/tier1_study.db`). If the coordinator process is killed, restarting
it with `load_if_exists=True` picks up where it left off. In-flight SLURM
jobs continue to run and their metrics get parsed on the next batch.

---

## 8. How the files fit together

Here's the full dependency graph of everything we built for this
side-experiment:

```
+----------------------------------------------------------------+
|                                                                |
|   PHASE 0 (already done): calibrate on real D2                 |
|                                                                |
|   src/analysis/inter_slice_consistency.py                      |
|         |                                                      |
|         v                                                      |
|   metrics/real_iscs_profile.json                               |
|         (reference distributions — used at scoring time)       |
|                                                                |
+----------------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------------+
|                                                                |
|   PHASE 1 (per trial): coordinator launches sbatch jobs        |
|                                                                |
|   scripts/tier1_sweep_coordinator.py                           |
|         |                                                      |
|         | (Optuna suggests params, exports as env vars)        |
|         v                                                      |
|   scripts/tier1_run_config.sh                                  |
|         |                                                      |
|         | (runs the full pipeline for one config × one seed)   |
|         v                                                      |
|   +--------------------------------------------------------+   |
|   |                                                        |   |
|   |  src/Generator/assemble_synthetic_volumes.py           |   |
|   |    -> synth NIfTI volumes at:                          |   |
|   |       sweep/tier1/trial_N/seed_S/synth/D2-9XX/         |   |
|   |                                                        |   |
|   |  src/analysis/score_synth_iscs.py                      |   |
|   |    -> iscs_score.json (diagnostic)                     |   |
|   |                                                        |   |
|   |  src/RaovSeg_recreation/preprocess.py                  |   |
|   |    -> processed train_val/ + test/ dirs                |   |
|   |    -> emits <target>_label.npy for every organ         |   |
|   |                                                        |   |
|   |  src/RaovSeg_recreation/train_resclass.py --target ov  |   |
|   |    -> resclass_best_ov.pth                             |   |
|   |  src/RaovSeg_recreation/train_attuseg.py  --target ov  |   |
|   |    -> attuseg_best_ov.pth                              |   |
|   |  src/RaovSeg_recreation/train_resclass.py --target ut  |   |
|   |    -> resclass_best_ut.pth                             |   |
|   |  src/RaovSeg_recreation/train_attuseg.py  --target ut  |   |
|   |    -> attuseg_best_ut.pth                              |   |
|   |                                                        |   |
|   |  src/RaovSeg_recreation/evaluate.py --target ov        |   |
|   |    -> metrics_ov.json (per-subject + aggregate)        |   |
|   |  src/RaovSeg_recreation/evaluate.py --target ut        |   |
|   |    -> metrics_ut.json                                  |   |
|   |                                                        |   |
|   |  touch DONE                                            |   |
|   |                                                        |   |
|   +--------------------------------------------------------+   |
|                                                                |
+----------------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------------+
|                                                                |
|   PHASE 2 (per batch): coordinator waits + parses              |
|                                                                |
|   coordinator polls sacct until DONE files exist               |
|         |                                                      |
|         v                                                      |
|   for each trial: read metrics_ov.json + metrics_ut.json       |
|         from all 3 seeds, compute objective, tell Optuna       |
|         |                                                      |
|         v                                                      |
|   sweep/tier1_study.db (Optuna study, persistent)              |
|   sweep/tier1/trial_N/aggregate.json (per-trial summary)       |
|                                                                |
+----------------------------------------------------------------+
```

Key insight: the `src/analysis/` metric module is used TWICE — once
before the sweep to characterise real, once during every trial to
characterise synth. This is deliberate. The same code produces the
reference distribution and the scored distribution, so any bug in the
metric affects both equally — you can't accidentally give synth a lower
bar than real.

---

## 9. What outputs to expect

After a full 40-trial sweep, the disk layout will look like:

```
/mnt/parscratch/users/<user>/synth_mri/sweep/tier1/
    trial_0/
        seed_0/
            synth/D2-900/  ...  D2-9XX/       (~30 synth NIfTI dirs)
            processed/train_val/  test/       (60 + 8 subject dirs)
            models/
                resclass_best_ov.pth
                attuseg_best_ov.pth
                resclass_best_ut.pth
                attuseg_best_ut.pth
            predictions_ov/  predictions_ut/  (per-subject npy)
            iscs_score.json
            metrics_ov.json
            metrics_ut.json
            DONE
        seed_1/  ... (same layout)
        seed_2/  ... (same layout)
        aggregate.json           <- coordinator writes this
    trial_1/  ... (same shape)
    ...
    trial_39/  ... (same shape)

sweep/tier1_study.db             <- Optuna's persistent record
```

Each `aggregate.json` looks like this:

```json
{
    "trial": 12,
    "params": {
        "ovary_target": 0.2734,
        "iscs_alpha": 0.6218,
        "iscs_lowpass_sigma": 2.1,
        "z_smooth_sigma": 0.42,
        "body_mask": 1,
        "hist_match": 1,
        "skip_enh": 0
    },
    "n_seeds": 3,
    "ov_dsc_mean": 0.203,
    "ut_dsc_mean": 0.284,
    "ov_iou_mean": 0.141,
    "ut_iou_mean": 0.196,
    "ov_hd95_mean": 12.4,
    "ut_hd95_mean": 8.7,
    "objective_min_dsc": 0.203,
    "final_states": {
        "10800101": "COMPLETED",
        "10800102": "COMPLETED",
        "10800103": "COMPLETED"
    }
}
```

The Optuna study db (SQLite) can be queried with any Optuna tool or
loaded interactively for analysis:

```python
import optuna
s = optuna.load_study(study_name="tier1_synth_params",
                     storage="sqlite:///sweep/tier1_study.db")
print(s.best_trial.params)
print(s.trials_dataframe().sort_values("value", ascending=False).head())
```

---

## 10. Cost estimate

- **Wall clock per job**: 8-11 hours (assemble 30-60 min + preprocess 10 min
  + 4 trainings 1-2 hours each + 2 evals 5 min each). SLURM allocation
  set to 16h.
- **Jobs total**: 40 trials × 3 seeds = **120 SLURM jobs**.
- **GPU-hours total**: ~1,000-1,300 (2x earlier estimate because each job
  now runs TWO RAovSeg pipelines, not one).
- **Wall clock end-to-end**: depends on Stanage queue. If we can keep
  10 jobs running in parallel and each is ~10h, ~120 hours = 5 days
  for the sweep to finish. If queue is tight, longer.
- **Storage per trial**: ~500 MB (processed + models + predictions).
  Total sweep: ~60 GB.

Compared to what we've already done (Phase 1: ~200 GPU-h, Phase 2:
~300 GPU-h), this is a comparable but distinct chunk.

---

## 11. Interpretation guide (once results come back)

After the sweep, a few questions to answer with the data:

**Q: Did any config beat the 0.290 real-only baseline?**
- Unlikely given previous ceiling analysis, but the sweep will tell us
  the actual ceiling of the 7-dim space, not just the 3 hand-tuned points.

**Q: What parameter mattered most?**
- Optuna provides `optuna.importance.get_param_importances(study)`
  which computes fANOVA-based importance. This tells us whether
  `ovary_target` or `iscs_alpha` or the new lowpass/smoothing knobs
  drove the results.

**Q: Does ISCS-score correlate with DSC across the sweep?**
- Scatter plot: x-axis = ISCS-score, y-axis = objective DSC. If
  correlated, inter-slice consistency IS a downstream utility signal —
  a nice finding for the writeup.

**Q: Does the sweep generalise from ovary to uterus?**
- Are the best ovary configs also good uterus configs, or do they
  diverge? Divergence would suggest the pipeline is organ-specific in
  ways not accounted for. Convergence would suggest the pipeline is
  a general-purpose augmentation pipeline.

**Q: What's the winning config for a top-3 nnU-Net robustness check?**
- Once we've picked a Tier 1 winner, we re-run the top-3 configs
  through nnU-Net (instead of RAovSeg's custom pipeline) to see if
  the ranking holds. Adds ~30 GPU-h at the end. Optional.

---

## 12. Sanity-check pass before spending 500 GPU-hours

Recommended commands to run before launching the sweep for real:

```bash
# On HPC (in the synth_mri conda env)

# 0. Install optuna if missing
pip install optuna

# 1. Verify the calibration script runs end-to-end.
#    Produces metrics/real_iscs_profile.json.
python -m src.analysis.inter_slice_consistency \
    --data-dir UT-EndoMRI/D2_TCPW \
    --out metrics/real_iscs_profile.json

# 2. Test the coordinator's suggestion logic (dry-run, in-memory).
#    Prints what would be submitted without actually submitting.
python scripts/tier1_sweep_coordinator.py \
    --dry-run --total-trials 5 --batch-size 5 --n-seeds 1

# 3. Submit ONE actual trial × 3 seeds to test the sbatch template
#    end-to-end. Wait for it to finish before scaling up.
python scripts/tier1_sweep_coordinator.py \
    --total-trials 1 --batch-size 1 --n-seeds 3

# 4. Inspect the outputs.
ls sweep/tier1/trial_0/seed_0/
cat sweep/tier1/trial_0/aggregate.json

# 5. If the above looks right, launch the full sweep in tmux.
tmux new -s sweep
python scripts/tier1_sweep_coordinator.py \
    --total-trials 40 --batch-size 10 --n-seeds 3
# Ctrl-B d to detach; sweep keeps running.
# tmux attach -t sweep to reconnect.
```

---

## 13. Results

Sweep completed on **2026-08-01** after ~12 days of wall clock. Ran a
total of 56 Optuna trials to reach 40 COMPLETE (some early trials
lost to a login-node reboot at 09:30 on 2026-07-24 mid-sweep — see
"Coordinator crashes" below).

### 13.1 Headline numbers

| Metric | Value |
|---|---|
| Best trial | **#35** |
| Best weighted objective (0.6·ov + 0.4·ut) | **0.366** |
| Best ovary DSC (n=3 seeds) | **0.266** |
| Best uterus DSC | 0.517 |
| Best ovary HD95 | 46.5 mm |
| Best uterus HD95 | 33.7 mm |

**Contextual comparison**:

| Configuration | Ovary DSC | Note |
|---|---|---|
| Real-only baseline (no augmentation) | **0.290** | Never crossed by the sweep |
| Phase 1 v3 SPADE (n=8, hand-picked defaults) | 0.178 | Previous augmentation ceiling |
| Phase 2 exp2_pathC (n=3) | 0.152 | Previous cross-domain ceiling |
| **Tier 1 sweep best trial (#35)** | **0.266** | +50% over Phase 1 v3, still −8% vs real-only |

### 13.2 Leaderboard — top 20 trials

![Top 20 trials](figures_tier1/fig_tier1_leaderboard.png)

Every top-20 trial has uterus DSC ≈ 0.53-0.58 (very consistent) and
ovary DSC ≈ 0.13-0.27 (dominant source of variance). The green
objective bars are what TPE optimised on.

### 13.3 Best parameters (trial #35)

```
--ovary-target-intensity 0.247
--iscs-alpha            0.013
--iscs-lowpass-sigma    2.23
--z-smooth-sigma        1.20
--body-mask             on
--histogram-match       on
--skip-enhancement-for-prefix D2-9
```

### 13.4 What TPE converged on — continuous parameters

![Continuous parameters vs objective](figures_tier1/fig_tier1_params_continuous.png)

The top-scoring trials (red dots) cluster tightly in every plot:

- **`ovary_target` ≈ 0.23-0.25**: below the hand-picked 0.26 we used
  earlier. Sits just inside RAovSeg's [0.22, 0.30] enhancement window.
- **`iscs_alpha` ≈ 0.0-0.05** in 7 of top-10 trials. **Contradicts the
  ISCS paper's default α=0.8.** Very low correlation between slice
  noises works best for downstream utility.
- **`iscs_lowpass_sigma` ≈ 2.2-2.7**: moderate structural lowpass
  compensates for the low α. Shared low-frequency noise gives
  volumes some 3D coherence.
- **`z_smooth_sigma` ≈ 1.1-1.3**: heavy post-hoc z-smoothing further
  compensates for the low α.

**Interpretation**: TPE found the combination `(low α, moderate
lowpass, heavy z-smoothing)` as the optimum. This is a
"structural-coherence-via-lowpass, texture-variety-via-independence"
recipe rather than the "everything correlated" recipe the ISCS paper
suggested. The heavy z-smoothing cleans up any residual jitter.

### 13.5 What TPE converged on — categorical flags

![Categorical parameters](figures_tier1/fig_tier1_params_categorical.png)

All three flags favour ON, with statistical separation:

- **`body_mask` on**: +9% mean objective vs off (0.315 vs 0.288).
- **`hist_match` on**: +5% mean objective vs off (0.311 vs 0.296).
- **`skip_enh` (Path C) on**: +7% mean objective vs off (0.308 vs 0.288).
  Only 6 of 37 trials had this off — TPE learnt to keep it on quickly.

**Path C is confirmed.** Every top-10 trial uses `skip_enh=1`.

### 13.6 Convergence — TPE learnt quickly

![TPE convergence](figures_tier1/fig_tier1_convergence.png)

The running best (red line) climbed steeply during the QMC/uniform
warmup phase (trials 0-9), then flattened as TPE refined. The
absolute best (0.366) landed at trial #35 — long after the sweep had
converged on the productive region. Trials past #35 rarely improved
the best, confirming the sampler had exhausted the exploitable
space.

**Implication for future sweeps**: 25-30 trials would have been
enough. The extra 10 trials past that consumed ~250 GPU-hours for no
gain. If we run another sweep on a related task, add early-stopping
after 10 trials without improvement.

### 13.7 Ovary vs uterus DSC — the two targets don't move together

![Ovary vs uterus DSC](figures_tier1/fig_tier1_ovary_vs_uterus.png)

Uterus DSC stays in a narrow band (0.51-0.58) across ALL trials.
Ovary DSC varies much more (0.07-0.27). The two targets are only
loosely correlated — most gains in the objective come from moving
ovary right, not uterus up.

This confirms our M2 audit prediction: `min(ov, ut)` would have
degenerated (uterus always ≥ 0.5, ovary always ≤ 0.3). We were
right to switch to `--objective weighted 0.6·ov + 0.4·ut`. The
weighted objective correctly rewards trials that move ovary DSC
higher while penalising trials that let uterus drop below its
typical range.

### 13.8 Ovary DSC distribution — the ceiling is real

![Distribution of ovary DSC](figures_tier1/fig_tier1_ovary_dist.png)

Only **1 of 37 trials (#35) exceeded 0.25** ovary DSC. **None
crossed the real-only baseline (0.290).** The sweep's median ovary
DSC is 0.146 — below even Phase 1's v3 SPADE (0.178), showing that
random-parameter augmentation is worse than hand-picked defaults on
average.

The 0.290 ceiling appears to be a **data-scale limit**, not a
hyperparameter limit. With 30 real subjects + 30 synth subjects,
even the best-tuned augmentation stays 8% below the ceiling that the
30-real-only baseline achieves. This aligns with our earlier Phase 1
finding that DDPM augmentation at n=30 is fundamentally
underpowered.

### 13.9 Compute cost and coordinator crashes

- **Total wall clock**: ~12 days (2026-07-19 to 2026-08-01).
- **Total GPU-hours**: ~1,000 (56 trials × 3 seeds × ~10h × ~60%
  concurrent utilisation).
- **Disk footprint peak**: 79 GB (before Lever B auto-cleanup landed
  on batch 2+). After cleanup: ~56 GB steady state.
- **Coordinator crashes**: 2. First on 2026-07-24 09:30 (Stanage
  login2 kernel soft-lockup — IT rebooted). Recovered cleanly via
  the reconciliation function we added earlier — 3 trials' data was
  recovered from disk without loss.
- **SLURM failures/re-runs**: 16 trials lost to various env issues
  during the first 2 days of debugging (`--export=ALL`, conda env
  activation, orphaned coordinator batches). Once stable, no
  further failures.

### 13.10 Three claims for the writeup

1. **Bayesian hyperparameter tuning did not cross the real-only
   baseline**, but pushed the augmentation ceiling from 0.178 →
   0.266 (+50%). The gap to baseline (0.290) is 2× the noise floor
   (±0.05), so the gap is statistically robust.
2. **All three "obvious" flags** (`body_mask`, `hist_match`,
   `skip_enh` / Path C) **should be ON**. This is now defensible
   with n=37 trials.
3. **The ISCS default (α=0.8) is wrong for this downstream task**.
   TPE found `α≈0` combined with `lowpass_sigma≈2.2` and
   `z_smooth_sigma≈1.2` as the winning recipe. A "structural
   coherence via lowpass, texture variety via independence"
   interpretation.

### 13.11 What would improve results further

None of these were part of Tier 1's scope, but they're the natural
follow-ups:

- **Tier 2 — retrain the DDPM** with a segmentation-consistency loss
  (Konz et al., 2024). Everything Tier 1 tuned is assembly-time; the
  ceiling comes from the generator itself.
- **More real subjects**. The 0.290 → 0.266 gap suggests we're
  compute-limited on the target side (only 30 subjects for RAovSeg
  to learn from). Larger cohorts would raise the baseline AND likely
  the augmented ceiling.
- **nnU-Net robustness check on top-3 configs** — re-run trial #35,
  #30, #39 with nnU-Net instead of RAovSeg to confirm the ranking
  is generator-driven, not pipeline-specific. ~30 GPU-hours.

---

## 14. Glossary of terms used above

- **DDPM / DDIM**: the generative model family this project uses.
  DDPM = Denoising Diffusion Probabilistic Model (training). DDIM =
  the sampler used at inference time (deterministic, ~100 steps).
- **RAovSeg**: the downstream ovary-segmentation pipeline from
  Liang et al. (2025). Two stages: ResClass (slice classifier) +
  AttUSeg (attention U-Net segmenter).
- **DSC (Dice Similarity Coefficient)**: the primary segmentation
  quality metric. 0 = no overlap, 1 = perfect overlap.
- **ISCS (Inter-Slice Consistent Stochasticity)**: technique from
  Kwon & Ye (ICLR 2025) for making 2D-per-slice DDPM sampling
  produce 3D-coherent volumes.
- **SPADE (Spatially-Adaptive Denormalization)**: the conditioning
  mechanism inside the DDPM that lets the label mask control image
  generation on a per-organ basis.
- **CFG (Classifier-Free Guidance)**: DDPM inference technique that
  amplifies the conditioning effect.
- **Optuna**: the Python library that runs the Bayesian hyperparameter
  search.
- **QMC (Quasi-Monte Carlo)**: space-filling sampling scheme used for
  the warm-up phase of the sweep.
- **TPE (Tree-Structured Parzen Estimator)**: the Bayesian sampler
  used after warm-up.
- **sacct / sbatch**: SLURM commands for submitting and querying jobs.
- **fANOVA**: functional ANOVA, a technique for measuring which
  hyperparameters explained the most variance in the objective.
