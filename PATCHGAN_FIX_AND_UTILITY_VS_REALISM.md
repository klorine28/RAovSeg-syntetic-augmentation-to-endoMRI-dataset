# The PatchGAN Fix and the Utility-vs-Realism Divergence

A story of a training bug, its accidental discovery, the retrain that
followed, and the empirical result that reframes what "quality" means
for medical image augmentation.

**Thesis of this document**:

> **Synthetic images used for medical segmentation augmentation do not
> need to look realistic — they need to be usable by the downstream
> segmenter's inductive biases.** These two goals CAN diverge, and in
> our data they DID diverge. The bug fix presented here provides the
> concrete evidence.

---

## 0. TL;DR

- A single `.detach()` in `train.py` was severing PatchGAN's gradient
  from the generator. Every 1c and Phase 2 run for months had trained
  with an idle discriminator, producing smooth "plausible-looking"
  synth that nonetheless got low downstream DSC (0.02-0.18).
- After the fix, the retrained generators produced synth that looks
  **less realistic** to human inspection but yields **substantially
  higher downstream DSC** across every variant (concat: 0.053 → 0.202,
  ~4×; Phase 2: 0.020 → 0.188, ~9×).
- The mechanism: PatchGAN's adversarial pressure pulls synth intensity
  distributions toward what fools a patch-level discriminator. Those
  distributions happen to also match RAovSeg's enhancement window —
  helping the segmenter — while producing higher-frequency textures
  that look artificial to humans.
- **Conclusion**: for downstream augmentation, "realism" as a human
  perceives it (FID, LPIPS, visual inspection) is the wrong quality
  target. The right target is task-aligned distributional match. The
  bug fix's before/after is the empirical proof.

> **Figures**: Eight figures are inserted at the relevant sections
> below (§4.1 DSC forest, §4.2 intensity histograms + overlay, §4.3
> good/bad/matched-anatomy grids, §5.1 diagnostic, §5.4 correlation
> grid + in-window scatter + CLR). All live under
> [`hpc_pulled/fixed_analysis/figures_fixed/`](hpc_pulled/fixed_analysis/figures_fixed/).
> In Notion the relative paths won't render — drag-drop each image
> from Finder at the marked location.

---

## 1. Background — what the bug was

Full technical narrative in
[`LAMBDA_ABLATION_COLLAPSE.md`](LAMBDA_ABLATION_COLLAPSE.md). One-paragraph
summary:

In `src/Generator/train.py` line 464, the single-step x̂₀ estimate for
the PatchGAN block was created with `eps_pred.detach().float()`. The
`.detach()` severed the computational graph at that point. Every
downstream tensor — including the discriminator's fake-logits and
the generator's adversarial loss `loss_g_adv` — became functionally a
leaf as far as `model.parameters()` were concerned. When the training
loop computed `loss_g_total = loss_diff + λ · loss_g_adv` and called
`.backward()`, only `loss_diff` produced non-zero gradients into the
generator. The `λ · loss_g_adv` term produced **exactly zero**
gradient, regardless of λ.

The discriminator's own path was fine — its parameters trained on
`x0_hat.detach()`. It became increasingly accurate at distinguishing
real from generated (D_acc(r/f) climbed toward 0.91/1.00). But that
accuracy never influenced the generator, because the graph pointing
back to G was cut at creation.

**Direct evidence**: a `torch.autograd.grad(λ·L_adv, model.parameters())`
diagnostic printed `|grad_lam_L_adv| = 0.000000e+00` at every training
step past λ warmup. Not a small number rounding to zero — literally
zero because there was no derivative path.

## 2. How the bug was found

Not by looking at the code. By noticing that **three λ variants
(exp2 at λ=0.01, exp2_lam05 at λ=0.05, exp2_lam50 at λ=0.5) produced
byte-identical synth NIfTI volumes**.

`md5sum` on the assembled synth showed the three "different-λ"
variants had identical hashes. Then `md5sum` on the state-dict tensors
of the three checkpoints showed the model weights were bit-identical
too. Then loss logs showed the three training runs had identical
`L_diff`, `L_adv`, and `L_D` values to four decimal places over 100k
steps on different GPUs and different days.

That's not statistical noise — that's the same computation running
three times. The λ parameter was entering the code but not the
gradient. From there it was a `grep -rn '\.detach()'` in the
adversarial block to find the culprit.

## 3. The fix

One line, in [train.py:464](src/Generator/train.py#L464):

```diff
- x0_hat = estimate_x0_from_eps(x_t, eps_pred.detach().float(), train_sched, t)
+ x0_hat = estimate_x0_from_eps(x_t, eps_pred.float(),           train_sched, t)
```

`eps_pred` is the generator's noise prediction. Removing `.detach()`
preserves the graph from `eps_pred` through `x0_hat` through
`discriminator(x0_hat, lbl)` through `loss_g_adv`. The gradient now
flows back into `model.parameters()` as intended.

The D-update path below already applied its own `x0_hat.detach()`
before feeding into the D forward pass, so the two update directions
stay correctly isolated. The pre-fix `.detach()` was doing the D-side
job (redundantly) while also breaking the G-side unintentionally.

## 4. What changed after the fix

All 5 affected training runs were retrained end-to-end
(exp1c_concat, exp1c_spade, exp2, exp2_lam05, exp2_lam50). Synth
volumes were re-assembled from the new checkpoints and re-run through
the RAovSeg augmentation pipeline at 3 seeds per variant.

### 4.1 The DSC table — pre-fix vs post-fix

| Variant | Pre-fix DSC | Post-fix DSC (n=3) | Δ | Note |
|---|---|---|---|---|
| **exp1c_concat** | 0.053 ± 0.056 | **0.202 ± 0.025** | **+0.149** | ~4× improvement |
| **exp1c_spade** | 0.178 ± 0.054 (n=8) | **0.226 ± 0.012** | +0.048 | +27%; new Phase 1 ceiling |
| **exp2** (λ=0.01) | 0.020 ± 0.010 | **0.188 ± 0.065** | +0.168 | ~9×; "collapse" was largely bug |
| **exp2_lam05** (λ=0.05) | 0.020 (bug artifact) | **0.173 ± 0.086** | first real | λ ablation now measurable |
| **exp2_lam50** (λ=0.50) | 0.020 (bug artifact) | **0.158 ± 0.147** | first real | Highest λ ≠ highest DSC |
| Real-only baseline | 0.290 | 0.290 | unchanged | Data-scarcity ceiling holds |

Every augmentation configuration improved by at least +0.048 DSC.
Concat and Phase 2 improved by 4–9×. **Real PatchGAN adversarial
pressure moves the DSC needle substantially.**

![DSC forest plot: pre-fix vs post-fix per variant](hpc_pulled/fixed_analysis/figures_fixed/results/fig_dsc_forest_fixed.png)
*Figure 4.1 — Per-variant DSC on the RAovSeg test set, pre-fix (grey)
vs post-fix (colour), with 95% CI whiskers. Real-only baseline in dashed
line. Every post-fix bar is higher than its pre-fix counterpart.*

### 4.2 The mechanistic footprint — ovary intensity distributions

Post-fix ovary-voxel intensity summary (from
`figures_fixed/mechanism/mech_ovary_intensity_table.csv`):

| Variant | Ovary mean | In-window (%) | Pre-fix in-window |
|---|---|---|---|
| Real D2 (pooled) | 0.521 | 10.6% | — |
| spade_fixed | 0.241 | 20.6% | 18.8% (essentially unchanged) |
| concat_fixed | 0.246 | **54.8%** | **16.2%** — jumped 3.4× |
| exp2_fixed | 0.344 | 9.9% | (all three identical pre-fix) |
| exp2_lam05_fixed | 0.322 | 19.7% | " |
| exp2_lam50_fixed | 0.340 | 9.1% | " |

**Concat's in-window fraction jumping from 16% to 55% is the dominant
mechanistic signature of the fix.** PatchGAN, once actually training,
tightens the intensity distribution around RAovSeg's enhancement
window [0.22, 0.30]. Spade's in-window was already high pre-fix (18.8%)
because its explicit label-conditioning modulation had captured most
of the localization signal even without adversarial pressure.

![Ovary voxel intensity histograms — real vs each post-fix variant](hpc_pulled/fixed_analysis/figures_fixed/mechanism/fig_mech_ovary_hist.png)
*Figure 4.2 — Ovary-voxel intensity distributions. Vertical band marks
RAovSeg's enhancement window [0.22, 0.30]. Concat_fixed pulls its
distribution tightly into the window (54.8% inside); real D2 (top) sits
mostly above (10.6% inside).*

![Real vs synth spatial overlay per variant](hpc_pulled/fixed_analysis/figures_fixed/mechanism/fig_mech_overlay.png)
*Figure 4.3 — Body-region overlay of real D2 vs each fixed variant. Shows
where in the anatomy the intensity mismatch lives.*

### 4.3 Visual evidence — what the images look like

Per-variant 20 good + 5 bad slice grids are in
`figures_fixed/samples/<variant>/`. The observation from human
inspection:

- **Pre-fix synth (buggy)** looked **smoother, softer, more
  plausibly medical.** Blurry-but-anatomically-familiar. Boring in a
  way that suggested "the model averaged everything and produced a
  reasonable-looking mean."
- **Post-fix synth (real PatchGAN)** looks **rougher, higher-frequency,
  less anatomically coherent at a glance.** Textures that fool a
  70×70 patch classifier but read as "off" when a human sees the
  whole slice.

Pre-fix images had the aesthetic of "an average real slice." Post-fix
images have the aesthetic of "a real slice that's been through
adversarial noise" — closer to the real distribution in a specific
sense (patch statistics) but visually artificial.

![Post-fix synth — 25 high-quality-score slices across all variants](hpc_pulled/fixed_analysis/figures_fixed/samples/good_grid.png)
*Figure 4.4a — Twenty-five post-fix synth slices ranked highest by
quality score (log(ov_vox) · (in-window + 0.5) − 3·window_distance).
Ovary contour in green. Note the rougher patch-level texture
characteristic of active PatchGAN training.*

![Post-fix synth — failure-mode slices](hpc_pulled/fixed_analysis/figures_fixed/samples/bad_grid.png)
*Figure 4.4b — Bottom-ranked slices per variant. Failure modes include
absent-ovary volumes and out-of-window intensity blobs.*

![Real vs synth per variant, matched anatomy](hpc_pulled/fixed_analysis/figures_fixed/samples/sample_grid_D2-000_vs_D2-900.png)
*Figure 4.5 — Real subject D2-000 (red contour) vs each post-fix variant
synthesising subject D2-900 (green contour), same slice indices.
Illustrates the "less realistic but higher DSC" pattern: post-fix synth
is patch-noisy where the real is smooth.*

### 4.4 What FID / LPIPS / hist_KL say — MEASURED (Aug 2026)

Pre-fix numbers (from `metrics/master_metrics.csv`):

| Variant | FID | hist_KL | LPIPS_mean |
|---|---|---|---|
| exp1a (no PatchGAN) | 188.2 | 8.15 | 0.824 |
| exp1b (no PatchGAN) | 200.1 | 6.89 | 0.745 |
| exp1c_concat (buggy PatchGAN = 1a + dead D) | 166.5 | 5.79 | 0.773 |
| exp1c_spade (buggy PatchGAN = 1b + dead D) | 188.1 | 7.20 | 0.699 |

Post-fix numbers (from `metrics/fixed/quality_*_fixed.json`, n_synth=256, n_real=200):

| Variant | FID | hist_KL | LPIPS_mean |
|---|---|---|---|
| exp1c_concat_fixed | **271.7** | 2.62 | 0.768 |
| exp1c_spade_fixed  | **274.1** | 0.96 | 0.725 |
| exp2_fixed         | 267.1 | 11.05 | 0.591 |
| exp2_lam05_fixed   | 349.7 | 4.86 | 0.640 |
| exp2_lam50_fixed   | 381.4 | 5.56 | 0.623 |

Pre → post deltas on the two matched Phase-1 variants:

| Metric | concat delta | spade delta | Direction |
|---|---|---|---|
| **FID** | 166.5 → 271.7 (**+63%**) | 188.1 → 274.1 (**+46%**) | WORSE — confirms prediction |
| LPIPS_mean | 0.773 → 0.768 (−0.01) | 0.699 → 0.725 (+0.03) | flat / slightly worse |
| **hist_KL** | 5.79 → 2.62 (**−55%**) | 7.20 → 0.96 (**−87%**) | BETTER — did not predict this |

**FID prediction confirmed** (+46% to +63% worse on both Phase 1
variants). But the three realism metrics do NOT agree with each other:

- **FID (Inception feature distance)** got dramatically WORSE.
- **LPIPS (per-image perceptual)** stayed essentially flat.
- **hist_KL (intensity histogram)** got dramatically BETTER.

That the three "realism" metrics point in three different directions
on the same synth is itself a strong result. It means "does this
synth look like real?" is not a single question — different metrics
answer different sub-questions and can disagree with each other.

**Mechanistic reading**: PatchGAN operates on 70×70 pixel patches
and shapes local intensity distributions. So intensity-domain
realism (`hist_KL`, comparing whole-image intensity histograms)
improved — the post-fix synth genuinely does have more real-like
intensity distributions. But Inception features (FID) capture
higher-order structure — edges, textures, part-relationships —
that PatchGAN's local-patch signal doesn't preserve. FID went up
because the generator sacrificed higher-order visual coherence for
patch-level intensity fidelity. LPIPS sits between the two and
moved little.

The utility-vs-realism story is therefore more nuanced than "all
realism drops while utility rises." The correct statement is:
**PatchGAN improves realism at the intensity-domain (hist_KL) and
utility for segmentation, at the cost of higher-order visual
realism (FID). Standard "does it look real" evaluation with FID
alone would call these generators worse, whereas by hist_KL and
DSC they are better.**

## 5. The utility-vs-realism divergence

### 5.1 What the pre/post evidence proves

The bug fix ran a controlled experiment we didn't design. Two versions
of the same generator (same seed, same data, same architecture, same
config) — one with PatchGAN active, one with PatchGAN silently dead —
produced two very different kinds of synth:

|  | Pre-fix (PatchGAN dead) | Post-fix (PatchGAN alive) |
|---|---|---|
| Visual realism | HIGHER (smooth, plausible) | LOWER (rough, artificial) |
| Downstream DSC | LOWER (0.02-0.18) | HIGHER (0.16-0.23) |
| Ovary in enhancement window | LOWER (16% concat, 18% spade) | HIGHER (55% concat, 21% spade) |
| Distribution match to real (predicted FID) | BETTER (166-200) | WORSE (predicted 250-350) |

**These are opposite directions on the "quality" axis, depending on how
you define quality.** If quality = realism, pre-fix wins. If quality =
utility for segmentation, post-fix wins.

![Utility-vs-realism divergence, real vs synth diagnostic](hpc_pulled/fixed_analysis/figures_fixed/diagnostic/synth_vs_real_fixed.png)
*Figure 5.1 — Diagnostic panel comparing real D2 against each post-fix
variant. Divergence between "matches real distribution" and "matches
enhancement-window band" is directly visible per variant.*

### 5.2 Why this happens — the mechanism

Two loss terms compete during post-fix training:

1. **`L_diff` (MSE)** — rewards being CLOSE TO AVERAGE. Wants smooth,
   mean-like predictions. Naturally produces "plausible" images
   because averages of real images look plausible.
2. **`λ·L_adv` (PatchGAN)** — rewards patch-level FOOLING of a
   discriminator. Discriminator classifies 70×70 patches as real/fake.
   To fool it, the generator learns high-frequency texture patterns
   that MATCH real T2FS at the patch level.

The two objectives can conflict:
- MSE wants the ovary region to be a soft, averaged intensity.
- PatchGAN wants the ovary region to have real-looking local texture
  variance (including intensities that fall in the ranges real ovary
  voxels occupy).

Under PatchGAN pressure, the generator sacrifices global visual
coherence for local texture matching. Because RAovSeg's downstream
pipeline uses a specific intensity window [0.22, 0.30] to enhance
ovary voxels — and PatchGAN's local optimization tends to produce
voxels in that window — the segmenter benefits, even as the whole
image starts looking artificial.

### 5.3 Why standard quality metrics fail here

FID, LPIPS, hist_KL are all "does synth look like real overall"
measures. They were designed on natural images (Cityscapes, ImageNet)
where "looks like real" and "is useful for downstream tasks" tend to
align. For medical augmentation with a specific downstream pipeline,
that alignment breaks:

- The downstream pipeline (RAovSeg) has its own inductive biases
  (enhancement window, 2-stage classifier+segmenter, slice-level
  ovary classifier).
- Synth quality that matches THOSE biases is what helps DSC.
- Synth quality that matches human perception (FID/LPIPS/visual
  inspection) may or may not overlap with what helps DSC.
- In our bug-fix experiment, the two DID NOT overlap. Post-fix synth
  was worse by generic-realism measures and better by downstream-DSC
  measures.

### 5.4 The correlation data supports this — MEASURED (Aug 2026, n=5)

Post-fix, all five metrics computed. Correlations with per-variant DSC:

| Metric | Pearson r | p_r | Spearman ρ | p_ρ | Reads as |
|---|---|---|---|---|---|
| **ovary_mean** | **−0.84** | 0.07 | −0.70 | 0.19 | lower ovary intensity → higher DSC |
| **FID** | **−0.83** | 0.08 | −0.60 | 0.29 | higher FID → higher DSC (further from real = better!) |
| **LPIPS_mean** | **+0.68** | 0.20 | +0.60 | 0.29 | higher perceptual distance → higher DSC |
| **hist_KL** | −0.50 | 0.39 | −0.70 | 0.19 | lower histogram divergence → higher DSC |
| **in_window_pct** | +0.42 | 0.48 | **+0.80** | 0.10 | more voxels in enhancement window → higher DSC |

Interpretation:

- **FID vs DSC r = −0.83.** Within the 5 post-fix variants, the ones
  with WORSE FID have BETTER DSC. This is the utility-vs-realism
  divergence as a numeric correlation — same sign of relationship as
  a t-test between pre-fix (low FID, low DSC) and post-fix (higher FID,
  higher DSC).
- **LPIPS vs DSC r = +0.68.** Same story: further from real perceptually
  = better DSC.
- **hist_KL vs DSC r = −0.50** (weaker). Post-fix, tighter intensity
  histograms slightly track higher DSC. Same direction as the
  task-specific metrics; makes sense because both hist_KL and in_window
  measure things in intensity space, and PatchGAN pushed the intensity
  distribution in a coherent direction.

The pattern that emerges: on n=5 post-fix variants, **FID and LPIPS
are negatively coupled with utility** (higher = better DSC),
**hist_KL and ovary_mean are positively coupled with utility**
(lower = better DSC). These are the same-magnitude correlations
(|r| ≈ 0.5–0.85) pointing in different directions on the same synth.
The field's default single-metric-summary evaluation would give
opposite rankings depending on which metric you pick.

### 5.5 The divergence splits by downstream metric family (Aug 2026)

Extended the correlation analysis to four downstream metrics
(DSC, HD95_mm, sensitivity, volume_error). FID's correlation with
each:

| FID vs | Pearson r | Direction |
|---|---|---|
| DSC | −0.84 | worse FID → higher DSC (divergence) |
| **HD95_mm** | **+0.87** | **worse FID → worse boundaries (NO divergence)** |
| sensitivity | −0.62 | worse FID → lower recall (weak divergence) |
| volume_error | −0.70 | worse FID → smaller volume error (divergence) |

**Nuance**: FID does not correlate with downstream utility in a
single direction. It correlates negatively with overlap/detection
metrics (DSC, sensitivity, volume_error) but positively with
boundary quality (HD95). PatchGAN training degrades feature-level
realism (higher FID) which hurts boundary quality but helps
detection because it also tightens intensity distributions toward
RAovSeg's enhancement window.

**Refined story**: the utility-vs-realism divergence is
metric-family-specific, not universal. FID predicts downstream
quality with opposite signs depending on which downstream metric
you value. For boundary accuracy it works as the field assumes.
For overlap/detection it works in the opposite direction. A
practitioner picking a generator by FID alone would land on
different choices depending on which downstream metric they
happened to care about — which is itself a strong argument for
never picking on FID alone.

### 5.6 Statistical honesty — the correlations do not pass n=5 significance

Bootstrap CIs (5,000 draws) on all 20 cells of the 5×4 matrix:
**every single 95% CI includes zero**. The point-estimate r values
are in the mechanistically-expected direction, but n = 5 does not
support statistical-significance claims.

Adding the 2 matched Phase-1 pre-fix variants (n = 7 total) flips
the FID vs DSC correlation from −0.84 to **+0.41**. Simpson's
paradox: the high-λ post-fix variants (exp2_lam05, exp2_lam50) are
worse on *both* FID and DSC because adversarial training becomes
unstable at high λ. So the within-post-fix "utility-vs-realism"
negative correlation is confounded by λ-instability, not a clean
utility/realism trade-off.

**What the transition itself shows.** The cleanest evidence for
the utility-vs-realism divergence is the pre→post transition for
the two matched Phase-1 variants:

- concat: FID 166 → 272 (+63%), DSC 0.053 → 0.202 (+280%)
- spade: FID 188 → 274 (+46%), DSC 0.178 → 0.226 (+27%)

Turning PatchGAN on ( = removing the `.detach()` bug ) traded some
FID-style realism for downstream detection utility. This is
directly observable in matched pairs and does not depend on
correlation statistics.

**Refined thesis**: PatchGAN training makes a *qualitative* trade
at the on/off level (some realism for some utility), but does
NOT support a *quantitative* claim that "worse FID → better DSC"
along the λ axis. Cranking λ up degrades both axes. Optimum λ
appears to be the minimum that keeps the discriminator training
(~0.01 in this study).

![Metric-vs-DSC scatter grid (n=5 post-fix variants)](hpc_pulled/fixed_analysis/figures_fixed/correlation/summary_grid.png)
*Figure 5.2 — Per-variant image-domain metrics against per-variant DSC.
`ovary_mean` (top-left, r = −0.85) and `in_window_pct` (top-right,
ρ = +0.80) are the two task-specific metrics that track DSC in the
mechanistically-expected direction.*

![In-window fraction vs DSC — the monotonic-saturating pattern](hpc_pulled/fixed_analysis/figures_fixed/correlation/scatter_in_window_pct.png)
*Figure 5.3 — Isolated scatter of in-window fraction against DSC.
Monotonic ordering (Spearman ρ = +0.80) with a visible plateau above
~20% — additional in-window match yields diminishing returns because
RAovSeg's slice-selection stage is already saturated.*

![Counterfactual Localisation Ratio per variant](hpc_pulled/fixed_analysis/figures_fixed/clr/fig_clr_counterfactual.png)
*Figure 5.4 — CLR by variant: fraction of ovary saliency mass that
disappears when the ovary label channel is zeroed at inference. Higher
= more label-conditional. Complements the intensity-window story with
a per-organ conditionality signal.*

### 5.5 The DSC standard-deviation clue

Post-fix `exp2_lam50_fixed` has std 0.147 for n=3 seeds. That's HUGE
compared to spade's 0.012. High variance across seeds is a classic
symptom of **unstable adversarial training** — different seeds
converge to different local minima of the adversarial objective.

Interpretation: at λ=0.5, PatchGAN is aggressive enough that the
training dynamics become chaotic. Some seeds land in
"good-for-segmentation" basins, others land in "adversarially-noisy"
basins that segment poorly. This is another symptom of the
utility-realism tension — the adversarial pressure is doing SOMETHING
strongly, but WHAT it does depends on the seed.

## 6. Why this matters for the field

### 6.1 The medical augmentation literature has this backwards

Standard practice: evaluate synth generators with FID, LPIPS, IS
(Inception Score). Papers ranking generators by these metrics assume
"more realistic = more useful for downstream." Our data says that
assumption is wrong for at least one clinically-plausible pipeline
(RAovSeg).

If a paper reports:
- "Our new generator achieves FID = X (better than SOTA)"

Without also reporting:
- "Downstream DSC on task T with segmenter S: Y"

The reader has no way to know if the generator is actually usable.
FID improvement CAN mean less useful synth if the downstream pipeline
has inductive biases that don't align with Inception features.

### 6.2 The pre-fix "concat is architecturally broken" claim was wrong

Pre-fix (with dead PatchGAN), concat had CLR ~0.03 (no per-organ
localization) and DSC ~0.05. The pre-fix story was: "concat
architecturally uses the label globally, so it can't produce
localized ovary tissue, so it can't help segmentation."

Post-fix, concat's DSC jumped to 0.202 — behind SPADE's 0.226 but
NOT catastrophically. The pre-fix architectural claim was actually
a bug artifact. Concat with real PatchGAN gets useful ovary voxels;
it just does so via patch-level texture matching rather than SPADE's
per-pixel modulation. **The pathway to useful synth is not unique.**

### 6.3 The Phase 2 "cross-domain collapse" claim was mostly wrong

Pre-fix Phase 2 got DSC 0.020, framed as "cross-domain style
transfer catastrophically fails." Post-fix Phase 2 gets DSC
0.16-0.19 — still below Phase 1 but the 8-9× "collapse" was mostly
the bug, not the paradigm.

Cross-domain synth is USABLE (DSC 0.17 vs baseline 0.29 = −41%
gap). It's not the panacea the initial framing hoped for, but it's
not the total failure the pre-fix numbers implied.

## 7. Recommendations

### 7.1 For our own dissertation

- **Substitute all pre-fix DSC numbers with post-fix numbers** in
  Chapters 4, 5, 6.
- **Retract the "concat architecturally locked out"** and
  "Phase 2 catastrophically collapses" claims explicitly. Frame the
  retraction as scientific integrity (bug found, results retested,
  interpretation updated).
- **Elevate the utility-vs-realism divergence as a headline claim**
  in the discussion. This is the paper's most novel contribution.
- **When FID/LPIPS results come back**, add a specific figure showing
  the divergence: X-axis = FID (lower is more realistic), Y-axis =
  DSC (higher is more useful). If the trend line is flat or negative,
  that IS the paper's headline.

### 7.2 For future medical augmentation work

- **Always report downstream task metrics**, not just FID/LPIPS.
- **Design task-specific quality metrics** that encode the downstream
  pipeline's inductive biases. For RAovSeg, that's `in_window_pct`;
  for other pipelines it's whatever the pipeline's preprocessing
  emphasizes.
- **Prefer generators that WERE trained against the target segmenter**
  (task-aware training) over ones that WEREN'T (task-agnostic
  training). Even a broken adversarial signal that pushes toward
  task-relevant intensity ranges beats a smooth, task-blind generator.

### 7.3 For future PatchGAN tuning on this pipeline

- λ_peak = 0.01 was already low; post-fix λ_peak = 0.5 shows signs
  of instability (high seed variance). Try:
  - λ_peak = 0.001-0.005 for softer adversarial push
  - Longer warmup (10k → 50k) so MSE dominates early
- Adversarial + MSE mixing is task-specific; there's no universal
  optimum. A Tier 1-style sweep of `(λ_peak, warmup_end, ramp_end)`
  would find our particular pipeline's sweet spot.

## 8. Meta-lesson from the bug

The bug went undetected for weeks because the observations were
plausibly consistent with a bad DDPM. "Concat CLR is low"
explained "concat DSC is low." "MSE dominates PatchGAN at low λ"
explained "Phase 2 collapses." Neither mechanistic story was
actually verified — they were plausible-sounding narratives fit to
the numbers.

A direct gradient measurement — `torch.autograd.grad(λ · L_adv,
model.parameters())` — would have caught it in 30 minutes. It would
have printed literal zero and pointed straight at the graph severance.

**Meta-lesson**: when a mechanistic story is invoked to explain a
result, MEASURE THE INVARIANT the mechanism relies on. Don't accept
plausible narratives; verify them.

This isn't just a debugging tip. It generalizes to the whole
utility-vs-realism argument in §5. The pre-fix "smooth realistic
synth" was ACCEPTED as good because it looked good. Nobody measured
whether the segmenter actually preferred it. When we finally did
measure — via the retrained pipeline giving 4-9× better DSC — the
realistic-looking synth turned out to be the WORSE synth for the job.

**Realism is a plausibility heuristic, not a utility measurement.**
For medical augmentation, we should measure utility directly. The
bug fix's before/after is the empirical demonstration of why.

---

## Appendix A — Where the evidence lives

| Evidence | File |
|---|---|
| Bug technical detail | [LAMBDA_ABLATION_COLLAPSE.md](LAMBDA_ABLATION_COLLAPSE.md) |
| Fix diff | [src/Generator/train.py](src/Generator/train.py) line 464 |
| Pre-fix DSC per variant | [docs_archive/RAOVSEG_AUGMENTATION_EXPERIMENT.md](docs_archive/RAOVSEG_AUGMENTATION_EXPERIMENT.md) |
| Post-fix DSC per variant | `runs/raov_aug_<variant>_fixed_seed{0,1,2}/metrics_ov.json` |
| Mechanism (ovary intensity) | `figures_fixed/mechanism/mech_ovary_intensity_table.csv` |
| 20 good + 5 bad per variant | `figures_fixed/samples/<variant>/` |
| Correlation table | `figures_fixed/correlation/correlation_table.csv` |
| Full correlation analysis | [METRIC_DSC_CORRELATION.md](METRIC_DSC_CORRELATION.md) §11 |
| FID / LPIPS / hist_KL (pre-fix) | `metrics/master_metrics.csv` |
| FID / LPIPS / hist_KL (post-fix) | `metrics/fixed/quality_*_fixed.json` (pending) |

## Appendix B — Suggested dissertation revisions

Sections that need editing based on this document:

- **Chapter 4 §4.3.4, §4.3.6** — "concat locked out" claim → retract, replace with post-fix number 0.202
- **Chapter 4 §4.5-4.6** — "Phase 2 catastrophic collapse" → soften, cite post-fix numbers 0.16-0.19
- **Chapter 4 §4.2.4** — "no metric predicts DSC" → replace with post-fix correlation (ovary_mean r=-0.85, in_window ρ=+0.80)
- **Chapter 5 §5.1** — Claim 1 (bad synth is worse than no synth) → soften from -93% collapse to -34% gap
- **Chapter 5 §5.2** — Claim 2 (concat architecturally locked out) → retract, replace with utility-vs-realism framing
- **Chapter 5 §5.4** — Claim 4 (preprocessing pipeline alignment matters more) → strengthen with the new correlation evidence
- **Chapter 5 (NEW §5.7 or §5.8)** — Utility-vs-realism divergence as a headline claim. Draft in this document §5.

## Appendix C — One-paragraph summary for the abstract

Draft:

> *"An initially reported catastrophic collapse of PatchGAN-augmented
> generators was discovered to result from a code bug that severed
> the discriminator's gradient path to the generator. After fixing
> the bug and retraining, all 5 affected generator configurations
> improved their downstream ovary segmentation DSC by 27-800%. However,
> the retrained synth images appear less visually realistic to human
> inspection and (based on preliminary FID data) will likely have
> worse standard quality metrics. This bug-fix natural experiment
> provides direct evidence for a task-utility vs perceptual-realism
> divergence: synthetic medical images optimized for a specific
> downstream pipeline's inductive biases can outperform more visually
> realistic alternatives, and standard quality metrics like FID may
> be actively misleading for the medical augmentation use case."*
