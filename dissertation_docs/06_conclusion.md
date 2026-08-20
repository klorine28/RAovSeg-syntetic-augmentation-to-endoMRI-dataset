# 6 — Conclusion

> **Target: 700 words.** Research summary, contributions, project-level
> limitations, future work, and closing statement.

---

## 6.1 Summary [target: 250 words]

This dissertation evaluated whether conditional-DDPM synthetic
augmentation can improve ovary segmentation in a data-starved pelvic
MRI regime (RAovSeg, 30 D2 training subjects, baseline DSC 0.290).
Across a systematic 2×2 generator architectural ablation (concat vs
SPADE conditioning crossed with the presence or absence of a PatchGAN
adversarial regulariser) and a cross-domain Phase 2 extension
(D1 T2 generator with D2 T2FS discriminator), **no configuration
produced synth that improved downstream DSC beyond the real-only
baseline**.

Phase 1's preprocessing-fix trajectory (v1 → v2 → v3, plus target
intensity sweeps and enhancement-toggle diagnostics) showed that
SPADE conditioning approaches but does not close the gap, plateauing
at DSC 0.178 ± 0.054 (n = 8), 38% below baseline. Concat conditioning
is architecturally locked out (CLR ≈ 0.03) and is not rescuable by
preprocessing tuning. The Phase 2 cross-domain extension produced a
catastrophic collapse (DSC 0.020 ± 0.010, −93% vs baseline) driven by
DDPM MSE loss dominating the adversarial signal in T2 → T2FS style
transfer.

The negative results support four claims of independent interest to
the field: (1) bad synth is worse than no synth at n < 50 real
subjects; (2) concat conditioning is architecturally locked out for
label-aware augmentation; (3) SPADE conditioning approaches but does
not close the gap at this data scale; (4) preprocessing pipeline
alignment matters more than raw synth quality. These claims and
their evidence are the discussion in Chapter 5.

## 6.2 Contributions [target: 100 words]

The dissertation contributes:

- A clean 2×2 conditional-DDPM ablation on pelvic T2FS MRI, fully
  characterised on FID, hist_KL, LPIPS-NN, and two novel per-channel
  interpretability metrics (CLR and OSI).
- The **Counterfactual Localisation Ratio (CLR)**, which predicts
  downstream utility for label-aware tasks where standard
  distributional metrics do not.
- The **label-aware ovary intensity rescaling technique** (Path B),
  the single most impactful preprocessing fix uncovered by this work.
- A variance-study protocol (n ≥ 5 seeds) that surfaces per-subject
  failure modes obscured by standard n = 3 reporting.

## 6.3 Limitations [target: 150 words]

Three project-level limitations shape the interpretation of the
results:

- **Compute budget** (~10 GPU-days on Sheffield Stanage A100s)
  constrained ablation depth. Beyond the 2×2 architectural spine and
  cross-domain Phase 2, sweeps over the DDPM step schedule, β
  schedule, discriminator width, and self-attention depth were not
  performed. A larger budget would broaden the ablation but is
  unlikely to reverse the negative direction.
- **Single downstream architecture (RAovSeg).** The idiosyncratic
  intensity-enhancement rule at [0.22, 0.30] is central to this
  dissertation's failure-mode story. Whether nnU-Net,
  TotalSegmentator, or a segmenter with different preprocessing
  produces the same conclusions is not established here.
- **No inter-rater re-analysis.** The 0.48 ± 0.24 inter-rater ceiling
  was adopted from Liang et al. (2025) unchanged. Whether the
  universal-failure subjects (D2-005, D2-023) sit near inter-rater
  disagreement or are simply hard cases is unresolved.

## 6.4 Future work [target: 150 words]

Four directions are ordered by likely value:

1. **Paired-cohort image-to-image translation** on D2's T2 / T2FS
   pairs, if per-subject pairs exist in UT-EndoMRI. Paired supervision
   (Pix2Pix style) is 100× stronger than the unpaired adversarial
   signal that Phase 2 attempted, and directly addresses the
   T2 → T2FS style-transfer failure mode.
2. **Semi-supervised RAovSeg pretraining**. Use synth for pretraining
   rather than augmentation; the fine-tuning step on real corrects
   distribution shift.
3. **Larger real-data collection** (n = 100–200 across multiple
   institutions). At n > 100 the augmentation calculus changes
   substantially; positive DDPM augmentation results become common.
4. **Alternative downstream segmenters** (nnU-Net,
   TotalSegmentator). Reproducing Phase 1 v3 SPADE with a different
   segmenter would test whether the conclusions here are
   RAovSeg-specific or general.

## 6.5 Closing statement [target: 50 words]

The path to a positive synthetic-augmentation result at n < 50 real
subjects is not more sampling or more preprocessing tuning but a
re-thinking of the generator's alignment with the downstream
consumer's preprocessing assumptions. Paired supervision and larger
real-data collection remain the natural next steps.

---

## 6.6 Post-fix corrections (Aug 2026)

§§6.1–6.2 above were written against measurements taken with a
PatchGAN gradient-severance bug in the training pipeline (documented
in [`LAMBDA_ABLATION_COLLAPSE.md`](../LAMBDA_ABLATION_COLLAPSE.md);
consequences in Chapter 4 §4.11 and Chapter 5 §5.7). The bug caused
`|∇(λ·L_adv)| = 0` on every 1c and Phase 2 training step, so those
generators never received an adversarial gradient. After fixing and
retraining all 5 affected generators, the summary and contributions
change as follows.

### 6.6.1 Corrected summary

Replace §6.1 paragraph 2 ("Phase 1's preprocessing-fix trajectory…")
and paragraph 3 ("The negative results support…") with:

> Phase 1 SPADE conditioning plateaus at DSC 0.226 ± 0.012 (n = 3
> post-fix), 22% below baseline — narrower than the pre-fix estimate
> of 39% but still a real gap. Concat conditioning, previously
> reported as architecturally locked out at DSC 0.053, reaches
> DSC 0.202 ± 0.025 once PatchGAN is actually training — small gap
> (~0.024) to SPADE and no longer a lockout claim. Phase 2
> cross-domain synthesis produces DSC 0.16–0.19 (not 0.020), which
> is a substantial gap to baseline but not the catastrophic collapse
> previously reported.
>
> The negative results support three revised claims of independent
> interest to the field: (1) no configuration in this ablation
> beats the real-only baseline at n = 30 real training subjects,
> consistent with a data-scale ceiling; (2) task-specific
> preprocessing-aware metrics predict downstream DSC with the
> expected sign (`ovary_mean` r = −0.85, `in_window_pct` ρ = +0.80,
> n = 5) whereas distributional metrics (FID, LPIPS, hist_KL) do
> not; (3) under adversarial regularisation, visual realism and
> downstream utility can diverge — the "utility-vs-realism
> divergence" documented in §4.11.7 and §5.8.

### 6.6.2 Corrected contributions

Replace §6.2's four bullets with:

- A clean 2×2 conditional-DDPM ablation on pelvic T2FS MRI, with
  the PatchGAN adversarial pathway both correctly implemented and
  verified via direct gradient measurement.
- The **Counterfactual Localisation Ratio (CLR)** and the
  **in-window fraction** metric, two task-specific interpretability
  measures that correlate with downstream DSC where standard
  distributional metrics do not.
- The **label-aware ovary intensity rescaling technique** (Path B),
  a generator-independent preprocessing fix that contributes ~+0.05
  DSC on every configuration in this study.
- An empirical demonstration of the **utility-vs-realism divergence**
  (§4.11.7): the same generator produces visually rougher but
  downstream-more-useful output when PatchGAN is training, on a
  segmenter with non-linear target-tuned preprocessing.
- A **variance-study protocol** (n ≥ 5 seeds) that surfaces
  per-subject failure modes obscured by standard n = 3 reporting.
- A **debugging methodology** (§5.7.5 meta-lesson): when a
  mechanistic story is invoked to explain a result, measure the
  invariant the mechanism relies on directly; do not accept
  plausible narratives without verification.

### 6.6.3 Corrected closing statement

The pre-fix closing statement (§6.5) stands but with a corrected
scope. The path to a positive synthetic-augmentation result at
n < 50 real subjects is not more sampling or more preprocessing
tuning but (a) alignment of the generator's output distribution
with the downstream consumer's preprocessing assumptions, and
(b) evaluation with task-specific rather than realism-centric
metrics. Paired supervision and larger real-data collection remain
the natural next steps for beating baseline; the mechanistic and
metric findings of this dissertation are applicable regardless of
whether that beat happens.
