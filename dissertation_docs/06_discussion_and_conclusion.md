# 06 — Discussion and conclusion

> The four headline claims, meta-lessons for the field, limitations,
> future work, and conclusion. Source: `../docs_archive/PAPER_OUTLINE.md`,
> `../docs_archive/RAOVSEG_AUGMENTATION_EXPERIMENT.md` §8h.3, memory files
> `project_phase2_result.md`, `project_variance_findings.md`.

---

## 6.1 The four headline claims

### 6.1.1 Claim 1 — bad synth is worse than no synth

Phase 2's exp2 downstream DSC of **0.020 ± 0.010** (n=3) — 93% below the
real-only baseline of 0.290 — demonstrates that a mediocre generator
does not merely fail to help; it **actively corrupts the downstream
training signal**. This is the sharpest empirical lesson from the whole
two-phase study.

**Why this claim is stronger than Phase 1's −39%**:
- Std across seeds is ~0.010 (much tighter than Phase 1's ~0.054) →
  the failure is stable, not variance-driven.
- All 3 seeds land in the same failure mode: predict near-zero ovary on
  essentially every test subject.
- The mechanism is understood (§5.5.3): the DDPM MSE loss on D1 T2
  dominated the adversarial signal from the unconditional D2-trained
  discriminator at λ_peak = 0.01. Generator plateaued at "gray blob."

**Practical implication for the field**: at n < 50 real subjects,
augmentation quality is not optional. A generator whose outputs do not
faithfully match the target distribution can degrade a working
baseline by 90%. This warning applies to any medical vision task at
similar data scale using generative augmentation.

**How Phase 1 relates to Claim 1**: Phase 1 already showed
degradation (−38% at best) but with wider seed variance and a plausible
"we're closing the gap" trajectory across v1 → v2 → v3. Phase 2 shows
what the failure looks like when the generator does not even acquire
the target style at all — sharper, stabler, and unambiguously worse
than no augmentation.

### 6.1.2 Claim 2 — concat conditioning is architecturally locked out

Concat-conditioned generators have Counterfactual Localisation Ratio
(CLR) values of 0.013–0.069 across all target channels
(§5.2.2, §5.2.3). This means removing the uterus label channel and
regenerating with the same noise changes the image nearly-uniformly
across the whole frame, rather than preferentially inside the uterus
region.

The downstream consequences are severe and consistent across every
preprocessing intervention:

| Fix level | Concat DSC | Interpretation |
|---|---|---|
| v1 (no fixes) | 0.150 ± 0.006 | −48% vs baseline |
| v2 (framing + hist match + body silhouette) | 0.044 ± 0.039 | −85% |
| v3 (v2 + Path B label-aware rescale) | 0.053 ± 0.056 | −82% |

**The mechanism**: label-aware preprocessing fixes (v2 histogram
matching, v3 Path B ovary rescale) rely on the generator having
localised, ovary-textured content in the correct spatial location.
Concat's CLR ≈ 0.03 means it does not have this. Rank-based histogram
matching (v2) then places bright pixels wherever the generator
happened to make them bright — random locations — and the segmenter
trains to predict ovary at random locations. Label-aware Path B rescale
(v3) then forces bright pixels *inside* the ovary label mask,
disconnected from surrounding synth tissue, and the segmenter can't
learn from a bright blob with no textured context.

**Concat is not a rescuable augmentation source** at Phase 1
preprocessing sophistication. This is an architectural limitation, not
a preprocessing failure, and it will not be fixed by more preprocessing
tuning.

**Implication for practice**: for downstream label-aware tasks
(segmentation, detection), per-organ localisation at the generator
matters more than raw image realism. Global conditioning mechanisms
(concat, class embedding, global cross-attention on a summary token) are
architecturally unlikely to produce useful segmentation-augmentation
synth even if their samples look realistic.

### 6.1.3 Claim 3 — SPADE conditioning approaches but does not close the gap

SPADE-conditioned generators (1b, 1c_spade) achieve CLR values of
0.30–0.53 across target channels — real per-organ localisation. This
gives the preprocessing fixes something to align with, and the downstream
trajectory reflects that:

| Fix level | SPADE DSC | Note |
|---|---|---|
| v1 (no fixes) | 0.138 ± 0.049 | Baseline augmentation attempt |
| v2 (3 preprocessing fixes) | 0.169 ± 0.037 | +22% vs v1 |
| v3 (v2 + Path B t=0.26, n=3) | 0.218 ± 0.057 | +58% vs v1 |
| **v3 revised at n=8** | **0.178 ± 0.054** | −38% vs baseline 0.290 |

**Every preprocessing fix moved SPADE upward**, up to a ceiling. The
sequence tells us:

- Preprocessing pipeline alignment is necessary (v2 → v3 mattered).
- It is not sufficient (0.178 is still 38% below baseline).
- The gap (0.11) is 2× the cross-seed std (0.054) — the underperformance
  is statistically robust.

**The variance-study correction**: n=3 had suggested SPADE was
approaching baseline (0.218 → −25%). n=8 revealed the mean was a
serendipitously high draw; seeds 3–7 averaged 0.154. The paper's
final SPADE number is 0.178, not 0.218.

**Options B and C both landed lower than v3 at t = 0.26**
(0.165–0.189), confirming that further preprocessing tuning does not
move the needle. Phase 1 is exhausted.

**Implication for practice**: even with a generator that achieves
per-organ localisation and careful preprocessing alignment,
DDPM-synthesised augmentation at n = 30 real subjects does not match
real-only baseline. The residual gap is attributable to two factors we
did not overcome in Phase 1: (i) generator quality at this data scale
(FID ~188, hist_KL ~7.2), and (ii) per-subject tractability of specific
test subjects (D2-005, D2-023 remain universal failures).

### 6.1.4 Claim 4 — preprocessing pipeline alignment matters more than raw synth quality

FID and hist_KL do not predict downstream success. What predicts
success (or, in our case, ceiling-level near-success):

1. **Field-of-view match** — synth NIfTI must be saved at the source
   real subject's spacing/origin/direction so the downstream
   preprocessing produces synth with the same body framing (~60% of
   frame) as real. (v2 Fix 3, `--no-resample-to-source` disables.)

2. **Body silhouette cleanup** — outside-body hallucinations survive the
   downstream percentile-clip + minmax and get amplified into structured
   noise. Kill them at generation time by masking with the
   `outside_body` channel. (v2 Fix 1.)

3. **Intensity distribution match** — rank-based histogram matching
   (v2 Fix 2) aligns the synth's post-clip distribution to real, but by
   itself does not guarantee the ovary lands in the enhancement window.

4. **Label-aware ovary intensity targeting** (v3 Path B) — the ovary
   label mask is used to explicitly force ovary pixels to intensity
   t = 0.26 (middle of RAovSeg's [0.22, 0.30] enhancement window). This
   is the single most impactful fix for SPADE (0.169 → 0.218 at n=3).

**Meta-lesson for the field**: synthetic generators for downstream
augmentation must be designed with awareness of the downstream
consumer's preprocessing assumptions. In our case, RAovSeg's ovary
enhancement rule at [0.22, 0.30] is a hidden pipeline assumption that
turns out to matter more than raw image realism. FID does not measure
whether the synth's ovary lands in the enhancement window; hist_KL
measures the aggregate distribution but not the per-organ intensity;
LPIPS measures perceptual similarity but not the segmenter-relevant
intensity band.

If we had reported only FID and moved directly to downstream evaluation,
we would have picked 1c_concat (best FID) and produced −85% DSC. The
CLR + per-organ localisation reasoning is what pointed us to SPADE, and
the downstream pipeline analysis is what pointed us to the enhancement
window.

**Implication for practice**: when publishing a synthetic-augmentation
result, characterise the downstream consumer's preprocessing chain
explicitly, identify its hidden assumptions (intensity windows,
morphological priors, field-of-view expectations), and evaluate whether
the synth satisfies them per-region. Aggregate distributional metrics
(FID, hist_KL) do not do this work.

## 6.2 Wider meta-lessons

### 6.2.1 The variance-study lesson

n=3 seeds are insufficient for downstream augmentation claims at this
data scale. Our v3 SPADE went from 0.218 (n=3) to 0.178 (n=8) — a 22%
drop in the reported mean. The "variance-masks-a-real-benefit" reading
was contradicted by the extra 5 seeds.

**Recommendation for the field**: n ≥ 5, ideally n ≥ 8, for downstream
augmentation DSC reporting. Per-subject variance dominates cross-seed
variance by ~4×, and reporting only the aggregate mean loses the
per-subject failure story (D2-005 and D2-023 as universal failures).

### 6.2.2 The FID-does-not-predict-downstream lesson

The 2×2 ablation produces a clean architectural map (§5.2.4) with no
single winner across quality metrics. 1c_concat wins FID and hist_KL;
1c_spade wins LPIPS. **Downstream DSC does not correspond to any of
these.** 1c_concat has the worst downstream DSC (v3: 0.053); 1c_spade
has the best (v3 n=8: 0.178).

The correspondence that *does* hold is between CLR and downstream
utility: high CLR → some downstream benefit is possible; low CLR →
architecturally locked out.

**Recommendation for the field**: adopt task-relevant, downstream-aware
metrics like CLR when evaluating generators intended for augmentation
of label-aware tasks. FID captures distributional realism but not
label-image consistency; hist_KL captures intensity distribution but
not spatial correspondence.

### 6.2.3 The cross-domain "same failure at 5×" lesson

exp2 (λ_peak = 0.01) produced "gray blob" generator with DSC 0.020.
exp2_lam05 (λ_peak = 0.05) tests whether 5× stronger adversarial
signal fixes it. If [as expected] exp2_lam05 lands in [0.02, 0.15], the
lesson is that **DDPM + adversarial cross-domain translation with
unconditional D is architecturally insufficient at n < 50 real** —
λ tuning is not the missing piece.

The natural next step (§6.4 future work) is paired-cohort image-to-image
translation (Pix2Pix) if D2's T2/T2FS pairs can be obtained, or
CycleGAN-style cycle-consistency if only unpaired data is available.
Neither is DDPM + adv, and neither was tested in this dissertation.

## 6.3 Limitations

### 6.3.1 Small test set (n = 8 D2 sacred subjects)

DSC confidence intervals are wide because the test set is small. Per-
subject variance across the 8 subjects is ~0.24 within a single seed —
much larger than cross-seed variance. Two subjects (D2-005, D2-023) are
universal-failure cases that alone cap achievable mean DSC at ~0.22.

**Consequence**: differences in mean DSC between augmentation variants
smaller than ~0.05 are within per-subject noise. The paper's headline
claims (−38% Phase 1, −93% Phase 2) are safely outside this noise; more
nuanced within-Phase-1 comparisons should be treated with caution.

### 6.3.2 D2-005 and D2-023 as universal failures — dataset property or artefact?

An open question we did not resolve: do D2-005 and D2-023 also fail
under the real-only baseline (making them a dataset property — perhaps
non-standard anatomy or imaging quality that RAovSeg cannot handle at
n=30 training subjects), or do they fail *only* under augmented training
(distribution-shift artefact of the synth data pushing the model away
from their appearance)?

If dataset property: they were always going to be zero, and the paper's
mean-DSC comparisons should ideally exclude them or report both
"raw" and "excluding-known-failures" numbers.

If artefact: the augmentation is corrupting the segmenter's ability to
handle rare-appearance subjects. This would sharpen Claim 1 (bad synth
poisons rare-appearance handling first).

Resolving this requires a per-subject DSC breakdown from the real-only
baseline runs at n ≥ 5 seeds — a small follow-up that would add real
paper value. Deferred to future work.

### 6.3.3 2D axial slice synthesis

The generator is 2D. 3D coherence is patched at inference via ISCS
(§4.7) — a partial fix that gives slice-to-slice consistency but does
not guarantee 3D anatomical plausibility (e.g. an organ that changes
shape non-smoothly across slices). RAovSeg is per-slice, so 2D
generation is native to the downstream task; but future work might
consider 3D DDPM (Med-DDPM style) for tasks with 3D consumers.

### 6.3.4 Single downstream architecture (RAovSeg)

We tested only RAovSeg (ResClass + AttUSeg + morphological postprocess)
as the downstream consumer. Whether nnU-Net, TotalSegmentator, or a
different 2-stage pipeline responds differently to the same synth is
unknown.

**Reason to suspect other segmenters may respond differently**:
RAovSeg's [0.22, 0.30] enhancement rule is idiosyncratic. A segmenter
that does not apply this rule would not experience the "synth ovary
doesn't fire the enhancement" failure mode. It might still experience
the framing or histogram issues, but the story would look different.

**Reason to suspect the conclusions still transfer**: the CLR ↔
downstream-utility link is architecture-agnostic. A generator that
does not localise per-organ is unlikely to help *any* label-aware
downstream task.

### 6.3.5 Compute budget

~10 GPU-days total on Sheffield Stanage A100s (~40 hours generator
training × 4 Phase 1 variants + ~40 hours Phase 2 + ~30 hours downstream
runs across all versions and seeds). This constrains the ablation
depth: we did not sweep DDPM training steps beyond 80k/100k, β-schedule
choice, self-attention depth, or discriminator base_channels. A larger
compute budget would allow broader ablation but is unlikely to change
the negative-result direction.

### 6.3.6 No inter-rater re-analysis

The 0.48 ± 0.24 inter-rater ceiling from Liang et al. (2025) was
adopted as-is. We did not run our own inter-rater study on UT-EndoMRI.
The 0.290 baseline is well below 0.48, but whether specific test
subjects (D2-005, D2-023) are near-inter-rater-agreement failures or
straight-up-hard cases is unresolved.

## 6.4 Future work

Ordered by likely value, with rough effort estimates.

### 6.4.1 Paired-cohort image-to-image translation on D2's T2 / T2FS

**High priority.** If D2 subjects have paired T2 and T2FS acquisitions
in the raw dataset (this needs checking; the current dissertation
treats D2 as T2FS-only and D1 as T2-only), a Pix2Pix or ControlNet
approach with per-subject paired supervision would sidestep the entire
"cross-domain style transfer via adversarial signal alone" failure mode
of Phase 2. Paired supervision is 100× stronger than
CycleGAN/unpaired adversarial signal.

**Effort**: 2–3 weeks. Requires (i) checking UT-EndoMRI for T2/T2FS
paired acquisitions, (ii) implementing paired supervision loss, (iii)
downstream evaluation.

### 6.4.2 Semi-supervised RAovSeg pretraining

**Medium priority.** Instead of using synth for augmentation, use it
for pretraining. Train an AttUSeg on synth + pseudo-labels, then
fine-tune on real. This is a different paradigm from augmentation and
may be less sensitive to synth quality — the fine-tuning step corrects
distribution shift.

**Effort**: 1–2 weeks. Straightforward modification to the RAovSeg
training scripts.

### 6.4.3 Larger real-data collection

**High value if feasible; long timeline.** The n = 30 constraint is
fundamental. Multi-institutional collection at n = 100–200 subjects
would move the segmentation baseline substantially closer to inter-rater
0.48 and likely change the augmentation calculus entirely (positive
augmentation results are common at n > 100).

**Effort**: 6–24 months. Requires IRB, site partnerships, annotation
budget.

### 6.4.4 Alternative downstream segmenters

**Medium priority.** Reproduce nnU-Net or TotalSegmentator baselines on
D2 and re-run the Phase 1 v3 SPADE augmentation experiment. This tests
whether the conclusions are RAovSeg-specific or general.

**Effort**: 1–2 weeks per segmenter, mostly the baseline reproduction.

### 6.4.5 Per-subject baseline analysis (small, high-value)

**Small effort, resolves §6.3.2 open question.** Run 5 seeds of the
real-only RAovSeg baseline and record per-subject DSC. Compare
D2-005, D2-023 real-only performance against their augmented
performance. If real-only also produces DSC 0 on those subjects,
they are dataset-property failures, not synth-augmentation artefacts.

**Effort**: 1 day of HPC time + analysis.

### 6.4.6 Higher-resolution FID reporting

**Cosmetic.** Current N = 256 gives FID noise floor of ~±30. Bumping to
N = 1024+ per variant would give paper-quality FID with tighter CI.
Does not affect the story.

**Effort**: ~25 min per variant on A100.

## 6.5 Conclusion

This dissertation empirically evaluates whether conditional-DDPM synthetic
augmentation can improve ovary segmentation in a data-starved pelvic MRI
regime (RAovSeg, n = 30 D2 training subjects, baseline DSC 0.290).
Across a systematic 2×2 generator architectural ablation (concat vs SPADE
conditioning × ±PatchGAN adversarial regularisation) plus a cross-domain
Phase 2 extension (D1 T2 generator + D2 T2FS discriminator), **no
configuration produces synth that improves the downstream DSC beyond the
real-only baseline**.

The trajectory across Phase 1 preprocessing fixes (v1 → v2 → v3) shows
that SPADE conditioning approaches but does not close the gap (best
n = 8 result: 0.178 ± 0.054, −38% vs baseline). Concat conditioning is
architecturally locked out (CLR ≈ 0.03) and is not rescuable by
preprocessing tuning. The Phase 2 cross-domain extension produces a
catastrophic collapse (DSC 0.020 ± 0.010, −93% vs baseline) driven by
DDPM MSE loss dominating adversarial signal for style transfer.

The negative results support four claims of independent interest to the
field:

1. **Bad synth is worse than no synth.** At n < 50 real subjects,
   mediocre generators do not fail neutrally; they poison downstream
   training.

2. **Concat conditioning is architecturally locked out.** Global
   conditioning mechanisms cannot produce useful segmentation-
   augmentation synth even if the samples look realistic.

3. **SPADE conditioning approaches but does not close the gap.** Per-
   organ localisation is necessary but not sufficient at this data
   scale.

4. **Preprocessing pipeline alignment matters more than raw synth
   quality.** FID does not predict downstream utility; the downstream
   consumer's hidden pipeline assumptions (e.g. RAovSeg's ovary
   enhancement window at [0.22, 0.30]) drive success or failure.

The dissertation's practical contributions include the CLR
interpretability metric (which predicts downstream utility better than
FID/hist_KL/LPIPS), the label-aware ovary intensity rescaling
technique (Path B, the single most impactful preprocessing fix), and a
variance-study protocol (n ≥ 5 seeds) that the field's standard n = 3
reporting would have obscured.

**The path to a positive result is not more sampling or more
preprocessing tuning** but a re-thinking of the generator's alignment
with the downstream consumer's preprocessing assumptions. Paired-cohort
image-to-image translation (Pix2Pix on D2's T2/T2FS pairs, if they
exist) is the natural next step; larger real-data collection would
change the augmentation calculus entirely; alternative downstream
segmenters would test whether the conclusions transfer. Any of these
lines would build on the empirical scaffold this dissertation
provides.

*[Placeholder: update Conclusion when exp2_lam05 results land. If DSC
∈ [0.20, ∞), revise Phase 2 framing per §5.6.2 matrix.]*
