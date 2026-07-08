# 01 — Introduction

> Dissertation-ready master. Consolidates the motivation, problem framing,
> thesis statement, and contributions. Source material: `../docs_archive/PAPER_OUTLINE.md`,
> `../docs_archive/architecture_dataflow_v2.md`, `../docs_archive/synthetic_mri_generator_design.md`,
> `../docs_archive/RAOVSEG_AUGMENTATION_EXPERIMENT.md`, memory files.

---

## 1.1 Clinical motivation

Endometriosis is a chronic gynaecological condition in which endometrial-
like tissue grows outside the uterus. Definitive diagnosis is invasive
(laparoscopy), and non-invasive workup relies heavily on pelvic MRI. The
imaging workup revolves around a small set of pelvic organs — the uterus,
the ovaries (and any endometriomas they harbour), and their relationship to
neighbouring structures. Among these targets, **automated ovary
segmentation** is the natural quantitative anchor because ovary volume,
morphology, and endometrioma involvement all feed downstream clinical
decision-making (surgical planning, longitudinal monitoring, phenotyping
for research cohorts).

The clinical value of automation is high:
1. Manual pelvic-organ contouring on 3D MRI is slow and expert-limited.
2. Inter-rater agreement on ovary segmentation caps at DSC ≈ **0.48 ± 0.24**
   (UT-EndoMRI human-vs-human DSC on the ovary target). This is the
   effective *ceiling* any automated method can aspire to.
3. State-of-the-art automated methods for endometriosis-focused pelvic MRI
   sit well below that ceiling. Liang et al. (2025, *Scientific Data*)
   published RAovSeg — a two-stage ResNet-slice-classifier + Attention U-Net
   pipeline with a fixed intensity-enhancement preprocessing rule — and
   reported a full-pipeline DSC of **0.290** on their D2 test set of 8
   subjects. This is our real-only baseline throughout the dissertation.

The gap between 0.290 and 0.48 is substantial (~66% headroom before the
inter-rater ceiling), which motivates methods that could push automated
DSC upward without requiring the further collection of hundreds of
expert-annotated subjects.

## 1.2 The data-scarcity problem

The dataset RAovSeg was developed against is UT-EndoMRI (Liang et al.
2025), which consists of two site-institutional cohorts:

| Cohort | Site | Sequence | Subjects |
|---|---|---|---|
| D1_MHS | Memorial Hermann | T2 (bright fat, non-fat-suppressed) | 51 |
| D2_TCPW | TCPW | T2FS (dark fat, fat-suppressed) | ~73 |

RAovSeg is trained/evaluated on D2_TCPW T2FS. Of the ~73 D2 subjects,
strict inclusion criteria (present T2FS + present ovary label + absent
cyst label + absent endometrioma label) yield **30 train-val subjects
and 8 held-out test subjects** — the classical low-data medical imaging
regime.

At n = 30 training subjects the segmenter is chronically data-starved:

- The 0.290 DSC is not simply "close to inter-rater 0.48"; it is far below,
  and the residual gap is characteristic of small-cohort medical vision
  tasks where diverse anatomical variation is underrepresented at training.
- Per-subject variance dominates: on the RAovSeg augmentation runs, the
  DSC standard deviation *within* a seed (across the 8 test subjects) is
  ~0.24 while the std *across* training seeds is ~0.054 — the segmenter's
  performance is fundamentally subject-tractability limited, not
  optimisation-limited.
- Two specific test subjects (**D2-005**, **D2-023**) are universal-failure
  cases in the augmented pipeline: DSC = 0 across all 8 seeds of the
  variance study. Whether they are also failures under the real-only
  baseline is an open question that would sharpen the paper's per-subject
  analysis; regardless, their existence caps the achievable mean DSC.

## 1.3 The natural but risky answer: synthetic augmentation

Faced with data scarcity, the standard research reflex is: *synthesise
more training data*. In our case, this means training a generator to
produce plausible pelvic T2FS MRI slices conditioned on anatomy labels,
then mixing generator outputs into the RAovSeg training pool.

The design idea is straightforward:
1. Train a **conditional denoising diffusion probabilistic model (DDPM)**
   that ingests a multi-channel anatomy label and produces the matching
   T2FS image slice.
2. Use it to generate synthetic (image, label) pairs.
3. Add them to RAovSeg's 30-subject training pool.
4. Measure DSC on the sacred 8-subject test set.

This is a well-explored recipe in medical imaging: Med-DDPM, MONAI
Generative examples, RoentGen, cross-domain Pix2Pix. Positive results are
common in large-domain tasks (CT, chest X-ray) with training sets in the
thousands. **Pelvic MRI at n=30 is an underexplored regime.** Whether
the diffusion + adversarial paradigm actually delivers a downstream
benefit at this data scale — or whether it fails, and if so how — is the
empirical question this dissertation answers.

The answer, from an extensive two-phase study, is that it **does not
help** and, in the cross-domain variant, actively **poisons the
downstream training signal**. The methodological path taken to reach that
conclusion — an architectural ablation, an iterative preprocessing
alignment story, a variance re-analysis, and a cross-domain extension —
produces a set of transferable lessons the field can build on.

## 1.4 Thesis statement

> **Naive synthetic augmentation via conditional DDPM does not improve
> ovary segmentation in the low-data pelvic MRI regime, and can actively
> harm downstream performance when the generator fails to acquire the
> target domain's style. Systematic architectural ablation
> (concat vs SPADE × ±PatchGAN) and cross-domain extension both fail to
> match the real-only baseline. The path to a positive result is not
> more sampling or more preprocessing tuning but a re-thinking of the
> generator's alignment with the downstream consumer's preprocessing
> assumptions.**

## 1.5 Contributions

The dissertation contributes:

1. **A clean 2×2 generator ablation** — concat vs SPADE conditioning
   crossed with ±PatchGAN adversarial regularisation — trained under
   matched conditions on the same 32 D2 subjects. Each cell is fully
   characterised on FID, hist_KL, LPIPS-NN, and two novel per-channel
   interpretability metrics (Counterfactual Localisation Ratio and Organ
   Specificity Index). The ablation produces a clean **architectural map**
   (localisation × realism) with no single winner across metrics.

2. **An empirical negative result on downstream augmentation** — every
   tested configuration (all four Phase 1 variants + cross-domain Phase
   2) reduces RAovSeg's DSC below the real-only baseline of 0.290. The
   best Phase 1 configuration (v3 SPADE with three preprocessing
   corrections + label-aware ovary intensity rescaling) plateaus at DSC
   0.178 ± 0.054 at n=8 seeds — 38% below baseline. Cross-domain Phase 2
   collapses to DSC 0.020 (−93%).

3. **A diagnostic decomposition of why the augmentation fails** — a
   sequence of preprocessing-alignment fixes (body silhouette masking,
   histogram matching, resampling to source frame, label-aware ovary
   rescaling) each partially closes the synth-vs-real distribution gap,
   but the sum falls short of restoring baseline. The residual gap traces
   back to the interaction between synth quality (per-organ localisation
   in particular) and RAovSeg's intensity-enhancement rule at [0.22, 0.30]
   — a hidden pipeline assumption that turns out to matter more than raw
   image realism.

4. **The "bad synth is worse than no synth" claim** — Phase 2's −93% DSC
   is stable across seeds (std ~0.010, tighter than Phase 1's ~0.054)
   and provides a much sharper practical warning than Phase 1's −38%:
   at low real-data scales, mediocre synthetic augmentation is not just
   wasted capacity; it can *corrupt* the downstream training signal.

5. **A variance-study protocol for downstream augmentation experiments**
   — the difference between n=3 and n=8 seeds turned out to be
   substantive (v3 SPADE mean 0.218 → 0.178). The dissertation
   documents the analytical shift the added seeds forced, and argues
   that n≥5 should be treated as the minimum for downstream synthetic
   augmentation claims at this data scale.

## 1.6 Structure of the dissertation

| Chapter | Content |
|---|---|
| **1. Introduction** (this doc) | Motivation, problem, thesis, contributions |
| **2. Background** | DDPMs, SPADE, PatchGAN, medical image synthesis, cross-domain MRI translation, RAovSeg |
| **3. Data and downstream pipeline** | UT-EndoMRI D1/D2, splits, sacred test set, 6-channel labels, RAovSeg architecture, preprocessing rule |
| **4. Methods** | Generator backbone, conditioning mechanisms, adversarial loop, CFG/EMA/ISCS, Phase 2 cross-domain setup |
| **5. Experiments and results** | Phase 1 generator quality (four variants), Phase 1 downstream (v1 → v2 → v3 → Options B/C), variance study, Phase 2 (exp2 catastrophic, exp2_lam05 pending) |
| **6. Discussion and conclusion** | Four headline claims, meta-lessons, limitations, future work |
| **7. Appendix — reproducibility** | GitHub repo, HPC layout, YAML configs, SLURM invocations |

The two-phase structure of the empirical work maps onto Chapters 5 and 6:
Chapter 5 lays out the ablations and their raw outcomes, Chapter 6
interprets and generalises.

## 1.7 Scope and out-of-scope

**In scope:**
- Ovary segmentation as the downstream task (RAovSeg full pipeline).
- 2D conditional DDPM as the synthesis primitive (with a 3D-coherence
  patch via ISCS at inference).
- Concat vs SPADE conditioning; PatchGAN as the adversarial variant.
- D2 T2FS in-domain (Phase 1) and D1 T2 cross-domain (Phase 2).

**Out of scope (deliberate):**
- Non-DDPM generators (GANs, VAEs, latent diffusion). The DDPM choice is
  a locked design decision to keep the ablation spine clean.
- Downstream segmenters other than RAovSeg. Whether a different segmenter
  responds differently to our synth data is a limitation, discussed in §6.
- Cyst-positive or endometrioma-positive subjects (excluded by RAovSeg's
  inclusion criteria).
- The D1 dataset for downstream evaluation. D1 is used only as generator
  training data in Phase 2; the test set remains the 8 sacred D2 subjects
  throughout.

## 1.8 Reading path recommendation

For a reader who wants the full story in dissertation order: read the
chapters as numbered. For a reader coming in for a specific claim:

- *"What did the ablation look like?"* → Chapter 4 (methods) and §5.2
  (Phase 1 quality metrics).
- *"Why did concat fail downstream while SPADE marginally helped?"* →
  §5.3 (Phase 1 downstream) and Discussion §6.1 (claim 2).
- *"Why is Phase 2 the sharpest negative result?"* → §5.5 (Phase 2 exp2)
  and Discussion §6.1 (claim 1).
- *"How would I reproduce any of this?"* → Chapter 7 (appendix).
