# 1 — Introduction

> **Target: 1,000 words.** Motivation, problem, thesis, aims and
> objectives, contributions, and report structure. Modelled after the
> distinction-level Introduction chapter pattern in aca22mmm and
> acu23ns.

---

## 1.1 Background and motivation [target: 300 words]

Endometriosis is a chronic gynaecological condition in which
endometrial-like tissue grows outside the uterus, affecting an
estimated 10% of reproductive-age women worldwide (Zondervan et al.,
2020). Definitive diagnosis remains invasive (laparoscopy), and
non-invasive workup relies heavily on pelvic magnetic resonance imaging
(MRI). Within the imaging workup, **automated ovary segmentation** is
the natural quantitative anchor: ovary volume, morphology, and
endometrioma involvement inform surgical planning, longitudinal
monitoring, and phenotyping for research cohorts.

The clinical value of automation is substantial. Manual pelvic-organ
contouring on 3D MRI is slow and expert-limited, and inter-rater
agreement between expert radiologists on the ovary target caps at
Dice similarity coefficient (DSC) 0.48 ± 0.24 on UT-EndoMRI (Liang et
al., 2025). Any automated method can in principle approach this
ceiling but not exceed it. State-of-the-art methods sit well below:
Liang et al.'s RAovSeg pipeline, which this dissertation adopts as its
downstream anchor, achieves DSC 0.290 on the 8-subject held-out D2
test set. The gap between 0.290 and 0.48 defines a real headroom for
methodological improvement.

The natural approach to closing this gap in a data-starved clinical
regime — 30 D2 training subjects — is **synthetic data augmentation**.
If a generator can produce plausible labelled pelvic T2FS MRI slices,
mixing its outputs into the RAovSeg training pool should in principle
increase effective training-set size and improve downstream DSC. This
recipe is well-explored in medical imaging: Med-DDPM (Dorjsembe et al.,
2024), RoentGen (Chambon et al., 2022), and cross-domain Pix2Pix
approaches (Zhu et al., 2017) all report positive downstream results
in adjacent tasks. The empirical question this dissertation answers
is whether the recipe generalises to the pelvic MRI at n = 30 regime,
which sits below the ~100-subject threshold at which most published
positive augmentation results cluster.

This project draws on data science techniques central to the MSc
Data Science programme at Sheffield IJC — generative modelling,
ablation-driven experimental design, statistical variance analysis,
and downstream evaluation methodology.

## 1.2 Problem and thesis statement [target: 200 words]

The empirical question is straightforward: *can conditional-DDPM
synthetic augmentation improve RAovSeg ovary DSC beyond the real-only
baseline of 0.290 at n = 30 training subjects?* The answer, arrived at
via a two-phase experimental study, is **no**.

The dissertation's thesis:

> Naive synthetic augmentation via conditional DDPM does not improve
> ovary segmentation in the low-data pelvic MRI regime, and can
> actively harm downstream performance when the generator fails to
> acquire the target domain's style. A systematic architectural
> ablation (concat vs SPADE conditioning crossed with the presence
> or absence of PatchGAN adversarial regularisation) and a
> cross-domain extension (D1 T2 generator with D2 T2FS
> discriminator) both fail to match the real-only baseline. The path
> to a positive result is not more sampling or more preprocessing
> tuning but a re-thinking of the generator's alignment with the
> downstream consumer's preprocessing assumptions.

Reaching this conclusion required a methodical experimental sequence
that produced a set of contributions of independent interest,
detailed in §1.4.

## 1.3 Aims and objectives [target: 200 words]

The overall aim is to determine whether conditional-DDPM synthetic
augmentation can improve RAovSeg ovary DSC beyond the real-only
baseline of 0.290, and to characterise the mechanisms that drive
success or failure.

The specific objectives are:

1. **Reproduce the RAovSeg real-only baseline** (DSC 0.290) as a
   verified anchor for downstream comparisons.
2. **Design and train four conditional DDPM variants** in a clean 2×2
   ablation (concat vs SPADE conditioning × ±PatchGAN adversarial
   regularisation) on the D2 T2FS training pool.
3. **Characterise generator quality** using standard distributional
   metrics (FID, hist_KL, LPIPS-NN) and two novel per-channel
   interpretability metrics (CLR and OSI).
4. **Evaluate downstream utility** by mixing synth into RAovSeg's
   training pool and reporting ovary DSC on the sacred 8-subject test
   set, across preprocessing-fix versions v1 → v2 → v3.
5. **Perform a variance study** at n = 8 seeds to test whether the
   Phase 1 augmentation trajectory is statistically robust.
6. **Extend to cross-domain Phase 2** (D1 T2 generator, D2 T2FS
   discriminator) to test whether cross-cohort data diversity closes
   the gap.
7. **Formulate methodological recommendations** for the medical
   augmentation literature at n < 50 real subjects.

## 1.4 Contributions [target: 150 words]

The dissertation contributes:

1. **A clean 2×2 conditional-DDPM ablation** on pelvic T2FS MRI, fully
   characterised on FID, hist_KL, LPIPS-NN, and two novel per-channel
   interpretability metrics.
2. **An empirical negative result on downstream augmentation** —
   every tested configuration (all four Phase 1 variants plus
   cross-domain Phase 2) reduces RAovSeg's DSC below the real-only
   baseline.
3. **The Counterfactual Localisation Ratio (CLR)** as an
   interpretability metric that predicts downstream utility for
   label-aware augmentation tasks where standard distributional
   metrics do not.
4. **The label-aware ovary intensity rescaling technique (Path B)** —
   the single most impactful preprocessing fix uncovered by this
   work.
5. **A variance-study protocol (n ≥ 5 seeds)** that surfaces
   per-subject failure modes obscured by standard n = 3 reporting in
   the medical augmentation literature.

## 1.5 Overview of the report [target: 100 words]

Chapter 2 reviews the literature, from clinical context through
generative modelling to the specific concat/SPADE/PatchGAN techniques
this dissertation ablates. Chapter 3 details the methodology — the
UT-EndoMRI dataset, the RAovSeg downstream pipeline, the DDPM
generator design, training strategy, and preprocessing alignment.
Chapter 4 reports the experimental results, moving from Phase 1
generator quality through the v1 → v2 → v3 downstream trajectory to
the n = 8 variance study and the Phase 2 cross-domain collapse.
Chapter 5 interprets the four headline claims. Chapter 6 concludes
with contributions, limitations, and future work.
