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
