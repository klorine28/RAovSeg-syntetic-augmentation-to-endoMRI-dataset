# Metric–DSC Correlation Analysis

> ⚠️ **AFFECTED BY THE POST-FIX UPDATE (2026-08-11)** — Half the DSC
> values used in this correlation analysis (1c_concat, 1c_spade, exp2,
> lam05, lam50) were measured against generators trained with the
> PatchGAN gradient-severance bug. Post-fix DSC jumps (1c_concat: 0.053
> → 0.202; exp2: 0.020 → 0.188) will substantially change any
> correlation reported here. The core structural claim ("metrics of
> realism don't predict downstream utility") likely survives — realism
> metrics barely change post-fix while DSC changes dramatically, so the
> utility-vs-realism divergence should be **stronger** after correction.
> But the specific correlation coefficients need re-computation once
> fixed-variant FID/LPIPS/hist_KL are recomputed. See dissertation §4.7
> and [LAMBDA_ABLATION_COLLAPSE.md](LAMBDA_ABLATION_COLLAPSE.md).

How synthetic-image quality metrics relate to downstream RAovSeg
segmentation performance. This is the empirical backbone of Chapter 4
§4.2.4's claim that **generic image-quality metrics do not predict
downstream utility for medical augmentation, but task-specific
mechanistic metrics do**.

The claim rests on measuring both sides — synth metrics AND downstream
DSC — for the same set of variants, and computing correlation
coefficients.

---

## 1. The question

Given N generator variants, each with:
- a set of quality metrics computed on its synth (X₁, X₂, …, Xₖ), and
- a downstream DSC score when its synth is used to augment RAovSeg (Y),

**which of the Xᵢ predict Y?**

Answer determines whether the field should adopt task-specific quality
measures or continue with generic ones (FID/LPIPS/hist_KL).

---

## 2. The metrics we correlate

### 2.1 Generic image-quality metrics

These treat synth as generic images with no task-awareness.

| Metric | What it measures | Source |
|---|---|---|
| **FID** | Frechet distance between InceptionV3 features of real vs synth image sets | `metrics/master_metrics.csv` (from earlier runs) |
| **LPIPS** | Learned perceptual patch similarity (network trained on human judgments) | `metrics/master_metrics.csv` |
| **hist_KL** | KL divergence between real and synth intensity histograms | `metrics/master_metrics.csv` |

Bias: all three were designed on natural images (Cityscapes, ImageNet)
and their transferability to medical MRI is unproven.

### 2.2 Task-specific mechanistic metrics

Designed with awareness of RAovSeg's pipeline.

| Metric | What it measures | Source |
|---|---|---|
| **in_window %** | % of synth ovary voxels landing in RAovSeg's enhancement window [0.22, 0.30] | `figures_fixed/mechanism/mech_ovary_intensity_table.csv` |
| **ovary mean intensity** | Post-normalization mean of ovary voxels | Same CSV |
| **CLR (Counterfactual Localization Ratio)** | Regenerate synth with an organ label removed; measure how much the OUTPUT changed IN that organ region vs elsewhere. Higher = generator uses labels per-organ (good). Lower = generator uses labels globally (bad). | `figures_fixed/clr/*/sample_00_metrics.json` (after running explain.py) |

Task-specific because they encode RAovSeg's inductive biases directly.

### 2.3 The ground-truth Y (DSC)

For each variant:
- Train RAovSeg on real D2 + variant's synth
- 3 seeds per variant (independent train/val splits + model init)
- Evaluate on 8 sacred test subjects
- Ovary DSC = 2·|A∩B| / (|A|+|B|), full pipeline
- Aggregate: mean and std across seeds
- Also: mean bootstrap 95% CI on the mean (via `evaluate.py --metrics-out`)

Source: `runs/raov_aug_<variant>_fixed_seed{0,1,2}/metrics_ov.json`

---

## 3. Correlation methods

Two coefficients, computed and reported for every (metric, DSC) pair:

### 3.1 Pearson r

Measures LINEAR association. Range [-1, +1].

```
r = cov(X, Y) / (σ_X · σ_Y)
```

- **r = +1**: perfectly linearly related, positive slope
- **r = 0**: no linear relationship
- **r = -1**: perfectly linearly related, negative slope

Best when: relationship is linear; both variables approximately normal
across variants.

### 3.2 Spearman ρ

Measures MONOTONIC association (rank-based). Range [-1, +1].

```
ρ = pearson(rank(X), rank(Y))
```

- Robust to outliers and non-linear-but-monotonic relationships.
- Doesn't assume normality.

Best when: only 5-6 data points (as we have), where a single outlier
could distort Pearson r; or when the relationship is non-linear (e.g.,
diminishing returns).

### 3.3 Which one matters for this analysis

**Report BOTH.** Look for the following patterns:

| Pearson r | Spearman ρ | Interpretation |
|---|---|---|
| High (>0.7) | High (>0.7) | Metric strongly predicts DSC |
| Low | High | Non-linear but monotonic — metric captures ordering only |
| High | Low | Suspicious — probably driven by 1-2 outliers |
| Low | Low | No relationship — metric doesn't matter for DSC |

### 3.4 Sample-size caveat

We have only **5 variants** (exp1c_spade, exp1c_concat, exp2,
exp2_lam05, exp2_lam50). Correlation coefficients on n=5 are noisy —
a single flip can move r from 0.9 to -0.3. Interpret with:

- **|r| > 0.9** on n=5 with p ≈ 0.03: solid signal
- **|r| ≈ 0.5-0.8** on n=5 with p > 0.1: suggestive, not conclusive
- **|r| < 0.5**: no evidence of relationship

Report p-values honestly. Never claim significance below n=8-10 without
noting the small sample.

### 3.5 Also report the confidence interval

Fisher z-transform confidence interval on r:

```
r ± 1.96 · SE(z)   where SE(z) = 1/√(n-3)
```

For n=5: CI is (r - 1.13, r + 1.13) — wider than the r range itself
in most cases. This makes explicit that any correlation on n=5 is
weak evidence.

---

## 4. Data-collection pipeline

The exact order of operations to get from "5 fresh checkpoints" to
"correlation coefficients":

```
1. Retrain the 5 generators (exp1c_concat/spade + exp2 + exp2_lam05 + exp2_lam50)
                            ↓
2. Assemble synth from each fixed checkpoint (5 sbatch jobs)
                            ↓
3. Compute mechanism figures — extracts in_window %, mean, median per variant
                            ↓
4. Run explain.py per variant — extracts CLR (optional, for the full 4-metric analysis)
                            ↓
5. Compute FID, hist_KL, LPIPS between real and each variant (from separate script or pre-existing metrics/master_metrics.csv)
                            ↓
6. Run RAovSeg augmentation, 3 seeds per variant — 15 sbatch jobs
                            ↓
7. Load 15 metrics_ov.json files, aggregate DSC per variant (mean, std)
                            ↓
8. Correlate X (each quality metric) against Y (DSC) → produce table + scatter plots
```

Each step is a real prerequisite. Missing step 5 = no FID/LPIPS/hist_KL
correlations; missing step 4 = no CLR correlation; missing step 6 = no
DSC at all.

---

## 5. Interpretation framework — three possible findings

After the numbers come in, the story lands as one of:

### Story A — "Task-specific metrics predict DSC; generic ones don't"

Signature:
- `in_window %` and/or `CLR` correlated with DSC (|ρ| > 0.7)
- `FID`, `LPIPS`, `hist_KL` NOT correlated (|ρ| < 0.4)

Publishing implication: recommend the medical augmentation field adopt
task-aware quality measures. Chapter 5 lead claim.

### Story B — "No metric predicts DSC; data scarcity dominates"

Signature:
- All correlations near zero (|ρ| < 0.4)
- DSC values are near a flat ceiling regardless of synth quality

Publishing implication: at n < 50 real subjects, synth quality is a
minor consideration compared to data-scale limits. Focus should shift
to collecting more real data, not tuning synth.

### Story C — "Partial correlation; multiple factors needed"

Signature:
- One or two metrics correlated but not perfectly
- DSC has variance that no single metric explains

Publishing implication: quality is multi-factorial. Recommend a
composite score (e.g., in_window × CLR) as the target for future work.

Which story lands is empirical. We designed the study to be able to
distinguish all three.

---

## 6. The analysis script

`src/analysis/metric_dsc_correlation.py` — reads the mechanism CSV +
the 15 DSC JSONs, computes Pearson r and Spearman ρ for each available
metric-vs-DSC pair, prints a summary table, and writes scatter plots
to disk. Usage documented in §7.

---

## 7. How to run the analysis

Once ALL 15 downstream jobs finish (see `squeue --me | grep raov`
returns empty and 15 `metrics_ov.json` files exist):

```bash
cd /mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset
export PYTHONNOUSERSITE=1
source /mnt/parscratch/users/$USER/anaconda/.envs/synth_mri/bin/activate

python -m src.analysis.metric_dsc_correlation \
    --mechanism-csv figures_fixed/mechanism/mech_ovary_intensity_table.csv \
    --dsc-root /mnt/parscratch/users/$USER/synth_mri/runs \
    --variants exp1c_concat exp1c_spade exp2 exp2_lam05 exp2_lam50 \
    --n-seeds 3 \
    --out-dir figures_fixed/correlation
```

If CLR data is available (from explain.py):

```bash
python -m src.analysis.metric_dsc_correlation \
    --mechanism-csv figures_fixed/mechanism/mech_ovary_intensity_table.csv \
    --dsc-root /mnt/parscratch/users/$USER/synth_mri/runs \
    --variants exp1c_concat exp1c_spade exp2 exp2_lam05 exp2_lam50 \
    --n-seeds 3 \
    --clr-root . \
    --out-dir figures_fixed/correlation
```

If a `metrics/master_metrics.csv` exists with FID / LPIPS / hist_KL:

```bash
python -m src.analysis.metric_dsc_correlation \
    --mechanism-csv figures_fixed/mechanism/mech_ovary_intensity_table.csv \
    --master-csv metrics/master_metrics.csv \
    --dsc-root /mnt/parscratch/users/$USER/synth_mri/runs \
    --variants exp1c_concat exp1c_spade exp2 exp2_lam05 exp2_lam50 \
    --n-seeds 3 \
    --out-dir figures_fixed/correlation
```

---

## 8. Expected output format

The script produces:

### 8.1 stdout summary table

```
=== Metric ↔ DSC correlation (n=5 variants) ===
metric              pearson_r   pearson_p   spearman_rho   spearman_p   interpretation
in_window_pct         +0.85       0.07         +0.90        0.04         solid predictor
ovary_mean            -0.42       0.48         -0.30        0.62         weak/absent
CLR                   +0.78       0.12         +0.80        0.10         suggestive
FID                   -0.15       0.81         -0.10        0.87         no evidence
LPIPS                 -0.22       0.72         -0.30        0.62         no evidence
hist_KL               -0.35       0.56         -0.30        0.62         weak/absent
```

### 8.2 Files written

- `figures_fixed/correlation/correlation_table.csv` — the summary above
  in machine-readable form
- `figures_fixed/correlation/scatter_<metric>.png` — one scatter plot
  per metric with linear fit, error bars on DSC, and per-variant labels
- `figures_fixed/correlation/summary_grid.png` — all scatter plots in
  one 2×3 grid for the dissertation figure

### 8.3 The narrative you can write from those outputs

Concrete example: if `in_window_pct` shows r=+0.85, p=0.07 on n=5:

> *"Across 5 generator variants, the fraction of synthetic ovary
> voxels landing within RAovSeg's [0.22, 0.30] intensity enhancement
> window was a strong predictor of downstream ovary DSC (Pearson
> r = +0.85, n = 5). Standard image-quality metrics (FID, LPIPS,
> hist_KL) showed no correlation with DSC (|r| < 0.4 for all three).
> This supports the claim that generic image-quality metrics, while
> useful in natural image synthesis, are not informative signals for
> medical augmentation utility. Task-specific mechanistic metrics
> that encode the downstream pipeline's inductive biases (in-window
> match, per-organ localization) should be preferred."*

---

## 9. Threats to the correlation interpretation

Small n = 5 is the biggest one. Others:

- **Same underlying DDPM architecture** — all 5 variants share the base
  DDPM. If a different family (e.g., LDM or StyleGAN) produced synth
  outside our observed range, correlations might not extrapolate.
- **Same downstream pipeline (RAovSeg)** — CLR and in-window are defined
  relative to RAovSeg's [0.22, 0.30] window and 2-stage pipeline. Different
  segmenter → different metric-DSC relationships.
- **Data scarcity dominates** — DSC ceiling at n = 30 real is ~0.29;
  metrics might show variance but downstream DSC is compressed
  against the ceiling, weakening the correlation signal.
- **Cross-experiment confounds** — Phase 1 and Phase 2 variants trained
  on different data (D2 T2FS-only vs D1 T2 → D2 T2FS). Mixing them
  in one correlation could hide within-phase relationships.

Mitigation: report Phase 1 (2 variants) and Phase 2 (3 variants)
correlations separately as sensitivity analyses, when feasible.

---

## 10. What we already know from the pre-fix data (context for the fresh run)

From Chapter 4 §4.2.4 of the existing dissertation draft, using
buggy-PatchGAN checkpoints:

| Metric | Pre-fix correlation with DSC | Notes |
|---|---|---|
| FID | Negative or absent | 1c_concat won FID but had worst DSC |
| LPIPS | Absent | 1c_spade won LPIPS, best DSC — but Phase 2 broke the pattern |
| hist_KL | Absent | 1c_concat won hist_KL, worst DSC |
| CLR | +0.9-ish, task-specific | SPADE variants beat concat in CLR, ordering matched DSC |
| in_window % | Positive, task-specific | Confirmed for SPADE; concat had low in_window AND low DSC |

The retrained run is a critical replication. If in_window and CLR still
correlate with DSC post-fix — but generic metrics still don't —
the finding is robust. If the fix reverses the story (e.g., concat_fixed
has high DSC despite low CLR), we learn something new about what
PatchGAN was actually doing.

---

## 11. Results — post-fix correlation (n=5 variants, 3 seeds each)

### 11.1 Per-variant table (input to the correlation)

| Variant | Ovary mean | In-window % | DSC (mean ± std, n=3) |
|---|---|---|---|
| exp1c_spade_fixed | 0.241 | 20.6% | **0.226 ± 0.012** |
| exp1c_concat_fixed | 0.246 | 54.8% | 0.202 ± 0.025 |
| exp2_fixed (λ=0.01) | 0.344 | 9.9% | 0.188 ± 0.065 |
| exp2_lam05_fixed (λ=0.05) | 0.322 | 19.7% | 0.173 ± 0.086 |
| exp2_lam50_fixed (λ=0.50) | 0.340 | 9.1% | 0.158 ± 0.147 |

Real-only baseline: 0.290 (unchanged, unaugmented). No variant crosses it.

### 11.2 Correlation coefficients

| Metric | Pearson r | Pearson p | Spearman ρ | Spearman p | Interpretation |
|---|---|---|---|---|---|
| `in_window_pct` | +0.425 | 0.48 | **+0.80** | 0.10 | Monotonic signal, non-linear |
| `ovary_mean` | **-0.85** | 0.07 | -0.70 | 0.19 | Strong linear, marginal p |
| `CLR` | — | — | — | — | Missing data (Phase 2 explain runs not completed) |

Standard metrics (FID / LPIPS / hist_KL) not recomputed for the fixed
variants — the `master_metrics.csv` rows we appended left those as
`nan`. To close the "generic vs task-specific" comparison, they would
need to be recomputed against the fixed synth. On the pre-fix data
they were flat correlations, so we expect no change.

### 11.3 Which story landed (from §5)

**Story A (task-specific metrics predict, generic don't) — partial support.**
Both task-specific intensity metrics point the expected direction
(ovary_mean negative, in_window positive). But the signal is noisier
than we'd hoped, primarily because n=5 with wide std on Phase 2 variants
(std=0.147 on lam50).

- `ovary_mean` has r=-0.85 with p=0.07 — borderline significant, right
  direction.
- `in_window_pct` has ρ=+0.80 — strong monotonic ordering but Pearson
  linear fit is weak (r=0.42) because the relationship saturates:
  moving from 10% to 55% in-window doesn't linearly buy DSC.

### 11.4 Anomaly worth explaining

`in_window_pct` predicted `exp2_lam05` (19.7%) would beat `exp2`
(9.9%). Actual DSC ordering: exp2 (0.188) > lam05 (0.173) > lam50
(0.158). All three within 1σ of each other though — lam05's std is
0.086 for n=3.

Two candidate explanations:
1. **In-window fraction is a necessary but not sufficient condition.**
   Once you're inside the window, other factors (per-organ localization,
   volumetric shape realism) take over. lam05's higher in-window match
   didn't rescue it from worse localization.
2. **n=3 seed noise** — with std=0.086 on lam05 vs 0.065 on exp2, the
   ordering could flip on a different seed draw. The variance study in
   Chapter 4 §4.4 predicted this kind of n=3 instability.

Distinguishing (1) from (2) needs more seeds; not affordable now.

### 11.5 Effect of the bug fix on the correlation

Pre-fix Phase 2 DSC was 0.020 ± 0.010 for all three λ (byte-identical
outputs). That gave the correlation NO Phase 2 variance to correlate
with — Phase 1 alone had n=2 which is uncorrelatable. So the pre-fix
correlation story was effectively n=2.

Post-fix Phase 2 DSC spans 0.158-0.188 (real variance), enabling
correlation across all 5 variants. **The correlation analysis was only
possible AFTER the PatchGAN bug was fixed.** This is itself a
retrospective validation of the bug — real hyperparameter response
now exists where before there was collapse.

### 11.6 Cautions for the writeup

- Report r AND ρ AND p together — a single number on n=5 can mislead.
- Note the wide 95% CI: Pearson r for ovary_mean is -0.85 with CI
  [-0.99, +0.14]. The lower bound touches "no correlation" territory.
- Frame findings as "consistent with Story A" rather than "proving
  Story A". The evidence is directional, not conclusive at n=5.
- **Do not claim generic metrics fail** without actually computing
  FID/LPIPS/hist_KL for the fixed variants. The pre-fix flat correlations
  are suggestive but not a fair comparison to the fixed-variant analysis.

### 11.7 Recommended sentence for Chapter 4 §4.2.4

Substitute for the pre-fix version:

> *"Across 5 retrained generator variants, ovary intensity distribution
> metrics predicted downstream DSC in the expected direction:
> ovary_mean vs DSC showed Pearson r = -0.85 (p = 0.07), and
> in_window fraction vs DSC showed Spearman ρ = +0.80 (p = 0.10).
> The relationship is monotonic but non-linear — higher in-window
> match yields diminishing DSC returns above ~20%, consistent with
> saturation of RAovSeg's enhancement step. Standard image-quality
> metrics (FID, LPIPS, hist_KL) were not recomputed for the fixed
> variants; pre-fix data showed them uncorrelated with DSC (Chapter
> 4 §4.2.4 pre-fix version)."*

