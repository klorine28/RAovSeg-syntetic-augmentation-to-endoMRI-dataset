# 5 — Discussion

> **Target: 1,300 words.** Four headline claims (§5.1–5.4),
> meta-lessons (§5.5), and interpretive limitations (§5.6). Project-level
> limitations and future work move to Chapter 6.

> ⚠️ **POST-FIX UPDATE (2026-08-11)**: The four headline claims in §5.1–5.4
> were framed against pre-fix DSC and CLR values, all of which were
> contaminated by the PatchGAN gradient-severance bug (see §3.10.2 and
> [LAMBDA_ABLATION_COLLAPSE.md](../LAMBDA_ABLATION_COLLAPSE.md)). §5.9
> at the end of this chapter reframes each claim with the post-fix numbers.
> Text in §5.1–5.6 is preserved for record; where a claim there is
> reframed, §5.9 supersedes.

---

## 5.1 Claim 1 — bad synth is worse than no synth [target: 250 words]

Phase 2's exp2 downstream DSC of **0.020 ± 0.010** (n = 3) —  93%
below the real-only baseline of 0.290 — demonstrates that a mediocre
generator does not merely fail to help; it **actively corrupts the
downstream training signal**. This is the sharpest empirical lesson
from the two-phase study.

Three properties strengthen the claim:
- Standard deviation across seeds is ~0.010 (much tighter than Phase
  1's ~0.054), so the failure is stable, not variance-driven.
- All three seeds land in the same failure mode: predict near-zero
  ovary on essentially every test subject.
- The mechanism is understood (Chapter 4, §4.5.3): the DDPM MSE loss
  on D1 T2 dominated the adversarial signal from the unconditional
  D2-trained discriminator at λ = 0.01. The generator plateaued at a
  "gray blob" that satisfies neither reconstruction nor style-transfer
  objectives.

The implication for the field is direct: at n < 50 real subjects,
augmentation quality is not optional. A generator whose outputs do not
faithfully match the target distribution can degrade a working
baseline by 90%. This warning applies to any medical vision task at
similar data scale using generative augmentation. It also refines the
common assumption that "more data is always better" — mediocre
synthetic data violates that assumption in a specific and quantifiable
way, at least in the ovary segmentation regime this work occupies.

## 5.2 Claim 2 — concat conditioning is architecturally locked out [target: 275 words]

Concat-conditioned generators (Exp 1a, 1c_concat) achieve
Counterfactual Localisation Ratio (CLR) values of 0.013–0.069 across
all target channels. In interpretive terms: removing the uterus label
channel and regenerating with the same initial noise changes the
image nearly uniformly across the whole frame, rather than
preferentially inside the uterus region. The label is used globally,
not per-organ.

The downstream consequences are severe and consistent across every
preprocessing intervention:

| Fix level | Concat DSC | Δ vs baseline (0.290) |
|---|---|---|
| v1 (no fixes) | 0.150 ± 0.006 | −48% |
| v2 (framing + hist match + body silhouette) | 0.044 ± 0.039 | −85% |
| v3 (v2 + Path B label-aware rescale) | 0.053 ± 0.056 | −82% |

The mechanism is clear from Chapter 4's diagnostic (§4.3.4).
Label-aware preprocessing fixes (v2 histogram matching, v3 Path B
ovary rescale) rely on the generator having localised, ovary-textured
content in the correct spatial location. Concat's CLR ≈ 0.03 means it
does not. Rank-based histogram matching then places bright pixels
wherever the generator happened to make them bright — random
locations — and the segmenter trains to predict ovary at random
locations. Label-aware Path B rescale forces bright pixels *inside*
the ovary mask, but disconnected from surrounding synth tissue, so
the segmenter cannot learn from an intensity-forced blob with no
textured context.

Concat is not a rescuable augmentation source at Phase 1 preprocessing
sophistication. This is an architectural limitation, not a
preprocessing failure. For downstream label-aware tasks (segmentation,
detection), per-organ localisation at the generator matters more than
raw image realism.

## 5.3 Claim 3 — SPADE conditioning approaches but does not close the gap [target: 275 words]

SPADE-conditioned generators achieve CLR values of 0.30–0.53 across
target channels — genuine per-organ localisation. This gives the
preprocessing fixes something to align with, and the downstream
trajectory reflects that:

| Fix level | SPADE DSC | Note |
|---|---|---|
| v1 (no fixes) | 0.138 ± 0.049 | Baseline augmentation attempt |
| v2 (3 preprocessing fixes) | 0.169 ± 0.037 | +22% vs v1 |
| v3 (v2 + Path B t = 0.26, n = 3) | 0.218 ± 0.057 | +58% vs v1 |
| **v3 revised at n = 8** | **0.178 ± 0.054** | −38% vs baseline 0.290 |

Every preprocessing fix moved SPADE upward, up to a ceiling. The
sequence tells us three things:

- Preprocessing pipeline alignment is necessary (v2 → v3 mattered).
- It is not sufficient (0.178 remains 38% below baseline).
- The gap (0.11) is 2× the cross-seed standard deviation (0.054) —
  the underperformance is statistically robust.

The n = 8 variance study corrects an earlier optimistic reading. At
n = 3, v3 SPADE appeared to be approaching baseline (0.218 → −25%
gap). Five additional seeds averaged 0.154, dragging the mean to
0.178. The original narrative — "variance is masking a real benefit"
— did not survive the added seeds.

Options B (target intensity sweep) and C (skip enhancement for synth)
both landed below v3, confirming that further preprocessing tuning
does not move the needle. Phase 1 is exhausted at 0.178 for SPADE.

The interpretive implication: even with a generator that achieves
per-organ localisation and careful preprocessing alignment, DDPM
augmentation at n = 30 real subjects does not match the real-only
baseline. Per-organ localisation is necessary but not sufficient at
this data scale.

## 5.4 Claim 4 — preprocessing pipeline alignment matters more than raw synth quality [target: 300 words]

FID and hist_KL do not predict downstream success. The 2×2 Phase 1
quality map (Chapter 4, §4.2.4) shows no single winner across
quality metrics: 1c_concat wins FID (166) and hist_KL (5.79),
1c_spade wins LPIPS (0.699), the two SPADE variants win CLR.
Downstream DSC does not correspond to any of these standard metrics —
1c_concat has the worst downstream DSC (0.053 at v3), 1c_spade has
the best (0.178 at v3, n = 8).

What does predict downstream utility:

1. **Field-of-view match** — synth NIfTI must be saved at the source
   real subject's spacing/origin/direction so downstream preprocessing
   produces synth with the same body framing (~60% of frame) as real.
2. **Body silhouette cleanup** — outside-body hallucinations survive
   downstream percentile-clip + minmax and get amplified into
   structured noise; kill them at generation time.
3. **Intensity distribution match** — rank-based histogram matching
   aligns the post-clip distribution but does not by itself guarantee
   the ovary lands in the enhancement window.
4. **Label-aware ovary intensity targeting** (Path B) — the single
   most impactful fix for SPADE (0.169 → 0.218 at n = 3).

The meta-lesson is that synthetic generators for downstream
augmentation must be designed with awareness of the downstream
consumer's preprocessing assumptions. RAovSeg's ovary enhancement
rule at [0.22, 0.30] is a hidden pipeline assumption that turns out
to matter more than raw image realism. If we had reported only FID
and moved directly to downstream evaluation, we would have picked
1c_concat (best FID) and produced −85% DSC. The CLR + per-organ
localisation reasoning pointed us to SPADE; the downstream pipeline
analysis pointed us to the enhancement window.

## 5.5 Meta-lessons for the field [target: 100 words]

Three cross-cutting lessons:

- **n = 3 seeds are insufficient for downstream augmentation claims at
  n < 50 real.** The v3 SPADE mean went from 0.218 (n = 3) to 0.178
  (n = 8) — a 22% drop. Per-subject variance dominates cross-seed
  variance by ~4×, so aggregate means without per-subject reporting
  hide universal-failure patterns like D2-005 and D2-023.
- **FID does not predict downstream utility** for label-aware tasks.
  Task-relevant metrics like CLR are essential.
- **Cross-domain DDPM + adversarial translation at n < 50 is
  architecturally insufficient**, at least under the standard
  λ schedule.

## 5.6 Limitations of interpretation [target: 100 words]

Three limitations bear on interpretation of the results above:

- **Small sacred test set (n = 8)** produces wide DSC confidence
  intervals. Per-subject standard deviation across the 8 subjects
  (~0.24) exceeds cross-seed standard deviation (~0.054), so
  differences smaller than ~0.05 are within noise.
- **D2-005 and D2-023 as universal failures.** Whether they also fail
  under the real-only baseline (dataset property) or only under
  augmented training (distribution-shift artefact) is unresolved.
- **Single downstream architecture (RAovSeg).** Whether the
  conclusions transfer to nnU-Net or TotalSegmentator is unknown.

Project-level limitations and future work are addressed in Chapter 6.

---

## 5.7 Post-fix retrain — updated claims (Aug 2026) [target: 400 words]

§§5.1-5.6 were written against measurements from a training pipeline
that contained a PatchGAN gradient-severance bug. All 1c and Phase 2
training runs had `|∇(λ·L_adv)| = 0` — the discriminator trained on its
own path but never influenced the generator. Full bug documentation:
[`LAMBDA_ABLATION_COLLAPSE.md`](../LAMBDA_ABLATION_COLLAPSE.md).
The fix, retrains, and updated results are documented in Chapter 4
§4.11. This section revises the four headline claims accordingly.

### 5.7.1 Claim 1 revised — "bad synth is worse than no synth"

**Pre-fix version**: Phase 2 collapsed to DSC 0.020, −93% below the
0.290 real-only baseline, attributed to a generator that plateaued at
"gray blob" body silhouette. The claim was that mediocre generators
actively corrupt segmentation training.

**Post-fix version**: fixed Phase 2 gets DSC 0.16-0.19, −34% to −45%
below baseline. This is still a meaningful gap, but the 8-9× less
severe collapse means the "actively corrupts" framing is too strong.
The revised claim: **cross-domain DDPM+adversarial synthesis at n<50
real subjects does not close the gap to real-only baseline, but it
does not catastrophically corrupt training either.** The gap is a
data-scale ceiling, not a synth-poisoning mechanism.

### 5.7.2 Claim 2 revised — "concat conditioning is architecturally locked out"

**Pre-fix version**: concat achieved DSC 0.053, ~4× lower than SPADE
(0.178), attributed to concat's low CLR (0.03) meaning the label was
used globally rather than per-organ.

**Post-fix version**: concat with real PatchGAN gets DSC 0.202 vs
SPADE's 0.226 — a small gap (0.024), not architectural lockout. In-window
fraction for concat jumped from 16% to 55% — direct mechanistic evidence
that PatchGAN was doing meaningful work by tightening the intensity
distribution. **Revised claim: concat conditioning is competitive with
SPADE once PatchGAN is training. Both are usable augmentation sources;
SPADE holds a small edge (~+0.02 DSC).**

### 5.7.3 Claim 3 revised — "SPADE approaches but does not close the gap"

**Pre-fix version**: SPADE at n=8 hit 0.178 ± 0.054, −39% vs baseline.

**Post-fix version**: SPADE at n=3 hits 0.226 ± 0.012, −22% vs
baseline. The gap remains real (0.064 = 1.2× the observed variance)
but narrower than pre-fix reported. **Revised claim stands but with
the corrected magnitude — SPADE augmentation lands within striking
distance of baseline but does not cross it, consistent with a
data-scale ceiling at n=30 real subjects.**

### 5.7.4 Claim 4 revised — "preprocessing pipeline alignment matters more than raw synth quality"

**Pre-fix version**: standard metrics (FID, hist_KL, LPIPS) did not
predict downstream DSC; task-specific metrics (CLR, in_window %) did.

**Post-fix version**: with n=5 fixed variants, task-specific
correlations survive (ovary_mean vs DSC: r = -0.85, p = 0.07;
in_window vs DSC: ρ = +0.80, p = 0.10). Generic metrics were not
recomputed for the fixed variants, so their post-fix correlation
strength is unknown — but the pre-fix flat pattern suggests they'd
remain uncorrelated. **Revised claim: task-specific metrics predict
DSC in the expected direction with weak-to-suggestive significance at
n=5; the relationship is monotonic but non-linear (saturates above
in-window ~20%).** Cannot claim generic metrics fail without
recomputing them — that comparison is left for future work.

### 5.7.5 Meta-lesson from the bug fix

The bug went undetected for weeks because the observations were plausibly
consistent with a bad DDPM (concat DSC 0.053 "explained" by low CLR;
Phase 2 DSC 0.020 "explained" by MSE dominating adversarial). A direct
gradient measurement (`torch.autograd.grad(lam * L_adv, params)`) would
have caught it in 30 minutes and prevented weeks of misinterpreted
results. **Meta-lesson: when a mechanistic story is invoked to explain a
result, measure the invariant the mechanism relies on. Don't accept
plausible narratives; verify them.**

---

## 5.8 Correlation analysis and mechanistic interpretation [target: 500 words]

Post-fix, five variants (concat, spade, exp2, exp2_lam05, exp2_lam50)
gave a small but consistent-enough sample to correlate per-variant
image-domain metrics against per-variant DSC. Two metrics were
computed on the fixed synth (`ovary_mean`, `in_window_pct`); the
generic distributional metrics (FID, LPIPS, hist_KL) were not
recomputed and their pre-fix values are used only as anchor points.
Full numerics and scatter plots are in §4.11.3; this section
discusses what those correlations *mean* and what they suggest for
future evaluation.

### 5.8.1 All five metrics correlate — but in three different directions

Post-fix, all five image-domain metrics were measured. On the same
5 variants, their correlations with per-variant DSC are:

- `ovary_mean` (task-specific, intensity): r = −0.84 (p = 0.07)
- **FID** (Inception feature distance): **r = −0.83** (p = 0.08)
- **LPIPS_mean** (perceptual): **r = +0.68** (p = 0.20)
- hist_KL (intensity histogram): r = −0.50 (p = 0.39)
- `in_window_pct` (task-specific, threshold): ρ = +0.80 (p = 0.10)

Two features of this table are the point:

1. **Standard "realism" metrics (FID and LPIPS) correlate with DSC
   in the *opposite* direction from what the field's default
   evaluation assumes.** FID r = −0.83 means: among post-fix
   variants, the ones that look *further* from real by Inception
   features have *higher* downstream DSC. LPIPS says the same
   (r = +0.68: further perceptually → higher DSC). If you ranked
   these five generators by FID alone you would get an ordering
   nearly opposite to their downstream ranking.
2. **The three "distributional" metrics disagree with each other.**
   FID gets much worse post-fix (+46% to +63% on the two matched
   Phase-1 variants); hist_KL gets much better (−55% to −87%);
   LPIPS barely moves. "Does this synth look like real?" is not
   a single question — different metrics answer different
   sub-questions and disagree.

The task-specific metrics (ovary_mean, in_window_pct) point in the
same direction as hist_KL: lower intensity divergence, higher
in-window match, better DSC. These three all live in the
intensity-domain sub-question of realism.

FID and LPIPS live in the feature-domain sub-question. They agree
with each other and disagree with the intensity-domain group.
Neither reaches conventional significance thresholds (p ≈ 0.07–0.39,
n = 5), so these are *suggestive* rather than *established*
relationships — but the sign consistency across five metrics is
itself informative.

### 5.8.2 Why the correlation signs are what they are

The `ovary_mean` correlation flips sign against the real distribution,
and this is not an artefact — it is the central finding. Real-D2
ovary intensities pool at mean 0.521, well above RAovSeg's
[0.22, 0.30] enhancement window. Real-D2 in-window fraction is 10.6%.
So the real-D2 distribution is *itself* a poor-alignment case from
the enhancement-window perspective; the segmenter compensates via
its downstream stages (ResClass thresholding, AttUSeg's attention
gates). Any generator whose ovary intensity distribution sits
*closer to the window than real does* is, from the preprocessing
pipeline's perspective, a gift: more ovary voxels survive to
segmentation without amplitude loss.

Two consequences worth stating explicitly:

1. **"Realism as ground truth" is wrong here.** The real
   distribution is not the optimum for this segmenter. Chasing FID
   toward real would move away from the enhancement-window sweet spot.
2. **The monotonic-with-saturation shape** of `in_window_pct` (ρ =
   +0.80, saturating around 20%) is a threshold effect from
   RAovSeg's discrete slice-selection stage. Once ResClass has
   enough in-window signal to pick the right slice, it picks it;
   extra in-window voxels beyond that point yield no DSC gain
   because AttUSeg is then bounded by mask geometry, not intensity.

### 5.8.3 Possible confounds and their bounds

Three confounds could produce these correlations without the
underlying mechanism being real:

- **n = 5 is small.** With five points, one outlier can drive an r
  of −0.85. concat_fixed is an outlier on `in_window_pct` (54.8%,
  next highest 20.6%). The Spearman/Pearson split (§4.11.3) tests
  robustness — the Spearman-rank story survives, so the monotone
  ordering is not a pure outlier artefact.
- **Correlated test sets.** All five variants were evaluated on the
  same eight D2 test subjects. Variance in the target DSCs is not
  independent across variants (subject-D2-023 fails universally,
  subject-D2-016 succeeds universally). This inflates apparent
  correlation strength; the reported r values should be treated as
  upper bounds.
- **Shared architectural ancestry.** Four of the five variants share
  the concat-Phase-1 backbone with only λ or conditioning
  differences. The correlations may partly reflect that shared
  lineage rather than an independent metric–DSC relationship. A
  distinct architecture (a transformer-based generator, a different
  UNet depth) would test this.

None of these confounds invalidates the mechanism, which is
independently supported by the intensity-distribution histograms
in §4.11.2 and by the direct DSC gain the fix produced. They do
cap the statistical strength claimable on n = 5.

### 5.8.3b The divergence is metric-family-specific (extended 5×4 evidence)

The DSC-only picture in §5.8.1 was extended to four downstream
metrics (DSC, HD95_mm, sensitivity, volume_error) using the same
n = 5 variants. The full Pearson r matrix is in [§4.11.3.1b](../dissertation_docs/04_experiments_and_results.md#41131b-extended-downstream-metrics--5-image--4-downstream-aug-2026). Its most important row is FID:

| FID vs | Pearson r | Interpretation |
|---|---|---|
| DSC | **−0.84** | worse FID → higher DSC (divergence holds) |
| HD95_mm | **+0.87** | worse FID → worse boundaries (divergence FAILS) |
| sensitivity | −0.62 | worse FID → lower recall (weak divergence) |
| volume_error | −0.70 | worse FID → smaller volume error (divergence holds) |

FID does not correlate with all downstream measures in the same
direction. It correlates negatively with overlap/detection metrics
(DSC, sensitivity, volume_error) — the utility-vs-realism divergence
— but *positively* with boundary quality (HD95). Feature-level
realism helps where boundary accuracy matters and hurts where
enhancement-window intensity matching matters.

**Mechanistic reading.** FID is computed from Inception features,
which encode higher-order structure — edges, textures,
part-relationships. Boundary quality (HD95) depends on the same
higher-order structure, so the two track together. Overlap metrics
(DSC, sensitivity) depend on whether ovary voxels survive
enhancement-window preprocessing — an intensity-domain question
that PatchGAN improves (higher `in_window_pct`) while degrading
feature-level structure (higher FID). The two mechanisms are
independent, so one image metric (FID) can predict two downstream
metric families in opposite directions.

**Refined thesis.** The utility-vs-realism divergence in this
dissertation is not "FID is uninformative for downstream
utility" — it is "FID predicts downstream utility with opposite
signs depending on which downstream metric you care about." A
generator picked on FID would beat the field on HD95 but lose to
it on DSC. On this dataset with this segmenter, the two axes of
quality that FID conflates are decoupled by PatchGAN training.

### 5.8.3c Statistical robustness — what the correlations can and cannot support

Two follow-up analyses tested how far the correlation findings can
be pushed statistically.

**Bootstrap CIs at n = 5.** 5,000 resampling draws of each of the
20 correlation cells in §4.11.3.1b. Every single cell's 95% CI
includes zero. Even the strongest correlations — FID vs DSC
(r = −0.84) and ovary_mean vs DSC (r = −0.85) — cannot be
statistically distinguished from zero at n = 5. The direction of
each correlation is mechanistically expected and consistent across
image metrics, but the sample size does not support
statistical-significance claims at conventional thresholds.

**Combined pre-fix + post-fix (n = 7).** Adding the two matched
Phase-1 pre-fix variants (exp1c_concat_pre and exp1c_spade_pre,
both with buggy dead PatchGAN) to the 5 post-fix variants gives
n = 7. The FID vs DSC correlation *flips sign* from −0.84 within
post-fix to +0.41 across the combined sample (§4.11.3.1c). This
is a Simpson's-paradox effect and it forces a more careful
statement of the utility-vs-realism finding.

**What the pre→post transition actually shows.** Both matched
Phase-1 variants moved in the same direction after the fix:
concat FID 166 → 272 (+63%) with DSC 0.053 → 0.202 (+280%); spade
FID 188 → 274 (+46%) with DSC 0.178 → 0.226 (+27%). Turning
PatchGAN on trades some realism for downstream utility. This is
the utility-vs-realism divergence at the "PatchGAN off → on"
transition, and it is directly observable in matched pairs.

**What the within-post-fix negative correlation actually shows.**
The strong within-post-fix r = −0.84 (FID vs DSC) is driven by
the two high-λ variants (exp2_lam05, exp2_lam50) being worse on
*both* FID and DSC. This is training instability at high λ, not
a genuine "worse realism = better utility" trade. Cranking λ up
does not improve downstream utility; it degrades both axes.

**Refined position for the dissertation.** The utility-vs-realism
divergence is a *qualitative on/off effect* of PatchGAN training,
not a *quantitative gradient* along the λ axis. Turning PatchGAN
on improves downstream utility while degrading FID-style realism.
Cranking PatchGAN up hurts both. The optimum in this study
appears to be the minimum λ that keeps the discriminator training
(λ_peak ≈ 0.01). Larger studies (n ≥ 15 variants) would be
needed to establish whether the within-fix correlations are real
signal or noise at these sample sizes.

### 5.8.4 Practical takeaway for synth-augmentation evaluation

For anyone evaluating a synthetic augmentation pipeline downstream
of a target-tuned preprocessor, this dissertation's evidence
suggests four concrete guidelines:

- **Report multiple metric families, not one.** A single-number FID
  or LPIPS is insufficient because the three distributional metrics
  we measured (FID, LPIPS, hist_KL) disagreed with each other on
  the same synth. Reporting only FID would have led to the wrong
  ranking of our five generators.
- **Include at least one task-specific metric.** The metric should
  encode the invariant the downstream consumer's preprocessing
  exploits (here: `in_window_pct`; elsewhere: whatever the
  preprocessor amplifies or suppresses).
- **Characterise the preprocessor's assumptions before choosing
  metrics.** RAovSeg treats intensity in [0.22, 0.30] as amplifiable
  signal. A different segmenter with a different preprocessor would
  care about different statistics; the right task-specific metric
  changes with it.
- **Do not assume "closer to real" is the optimisation direction.**
  When the downstream pipeline is non-linear and target-tuned, the
  real distribution may not be the optimum for that pipeline.
  Generators that deliberately deviate from real by Inception-feature
  distance (higher FID) toward the pipeline's operating window can
  outperform generators that optimise for that flavour of realism.
  This is the operational content of the utility-vs-realism divergence
  documented in §4.11.7.

---

## 5.9 Post-fix reframing of §5.1–5.4 (2026-08-11)

The `.detach()` bug (§3.10.2) means the four headline claims in
§5.1–5.4 were formulated against measurements that included the bug
as a confound. With the corrected numbers from §4.7, each claim
needs revision. Text of §5.1–5.6 above is preserved for reproducibility;
this section supersedes.

### 5.9.1 Claim 1 revised — "Bad synth is worse than no synth" → "Bad synth is worse than no synth, but Phase 2's synth wasn't as bad as we thought"

Phase 2 exp2 pre-fix: DSC = 0.020 (−93% vs baseline). Post-fix: DSC
= 0.188 (−35% vs baseline). Still a worse-than-baseline result but
the "catastrophic corruption" framing was measuring the bug, not the
cross-domain paradigm. The mechanistic claim survives (synth *can*
degrade the training signal), the magnitude does not.

### 5.9.2 Claim 2 revised — "SPADE localises more than concat" → "SPADE localises more than concat by ~6× with real PatchGAN, ~4× without"

With PatchGAN correctly wired, E_ovL: 1c_SPADE_FIXED = 1741× vs 1c_concat_FIXED
= 282× (6.2× ratio). Without PatchGAN: 1b SPADE = 968× vs 1a concat = 83×
(11.7× ratio). PatchGAN narrows the SPADE-vs-concat gap because it
amplifies concat more than it amplifies SPADE — likely because concat
has more room to gain from an adversarial signal.

### 5.9.3 Claim 3 revised — "λ_peak doesn't matter in Phase 2" → "λ_peak monotonically hurts Phase 2 above 0.01"

Pre-fix: three λ variants collapsed to identical outputs. Post-fix:
E_em decreases monotonically with λ (682 → 583 → 134 as λ 0.01 →
0.05 → 0.5). DSC ordering is within-noise at n=3 (0.188, 0.173,
0.158) but consistent with the specificity trend. The takeaway is
now "cross-domain adversarial pressure destroys per-channel
localisation above λ ≈ 0.01", not "λ has no effect".

### 5.9.4 Claim 4 revised — "Metrics don't predict utility" → "Metrics don't predict utility, and the fix confirms the same after correcting for the bug"

Pre-fix: FID / LPIPS / hist_KL did not predict downstream DSC in the
2×2. Post-fix: 1c_concat_FIXED gained +0.15 DSC over pre-fix while
its FID/LPIPS presumably barely moved (metrics not recomputed for
fixed variants yet — see §4.7.4 open item). The realism metrics
still say concat is worse than SPADE; the DSC now says concat is
approaching SPADE. The utility-vs-realism divergence is stronger
after the fix, not weaker.

### 5.9.5 New claim from the fix — "The PatchGAN mechanism is per-channel localisation, not texture"

Pre-fix, the discussion attributed PatchGAN's role to texture
realism. Post-fix, PatchGAN with a live gradient primarily sharpens
per-channel enrichment (all E_* values jump for both concat_FIXED
and SPADE_FIXED). Concat's in-window fraction jumps from 16% to 55%
— the mechanistic signature is intensity concentration inside the
segmenter's operating band, not texture per se.
