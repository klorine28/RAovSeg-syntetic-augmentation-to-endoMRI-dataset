# 2 — Literature Review

> **Target: 2,500 words.** Domain-first structure (clinical → imaging →
> existing methods → data problem → augmentation → generative solutions →
> gap). Modelled after the distinction-level structure in acu23ns
> (Selvam, 2025). Full reference list in the References chapter.

---

## 2.1 Overview [target: 100 words]

This chapter surveys the literature that positions the dissertation.
Section 2.2 establishes the clinical context of endometriosis and its
imaging workup. Section 2.3 reviews pelvic MRI sequences and the T2 vs
T2FS distinction that drives the cross-domain question in Chapter 4.
Section 2.4 surveys automated ovary segmentation and situates RAovSeg
(Liang et al., 2025) as this work's downstream anchor. Sections 2.5 and
2.6 introduce the data-scarcity problem and classical augmentation
responses. Sections 2.7 and 2.8 survey generative modelling in medical
imaging and the specific conditional-diffusion techniques the
dissertation ablates. Section 2.9 covers cross-domain translation.
Section 2.10 identifies the research gap.

## 2.2 Clinical context of endometriosis and pelvic MRI [target: 250 words]

Endometriosis is a chronic gynaecological condition in which
endometrial-like tissue grows outside the uterus, affecting an estimated
10% of reproductive-age women worldwide (Zondervan et al., 2020).
Definitive diagnosis remains invasive (laparoscopy), and the non-invasive
workup relies on pelvic magnetic resonance imaging (MRI). MRI is
preferred over ultrasound and CT for pelvic imaging in this population
because of its superior soft-tissue contrast and the absence of ionising
radiation.

The imaging workup revolves around a small set of pelvic organs — the
uterus, the ovaries (including any endometriomas they harbour), and
their relationships to neighbouring structures. Among these targets,
**automated ovary segmentation** is the natural quantitative anchor:
ovary volume, morphology, and endometrioma involvement inform surgical
planning, longitudinal monitoring, and phenotyping for research
cohorts (Bazot & Daraï, 2017).

Manual pelvic-organ contouring on 3D MRI is slow and expert-limited,
motivating automated methods. The clinical ceiling on any such method is
the inter-rater agreement between expert radiologists: on the UT-EndoMRI
dataset used throughout this dissertation, Liang et al. (2025) report an
inter-rater Dice similarity coefficient (DSC) of **0.48 ± 0.24** for the
ovary target. State-of-the-art automated methods sit well below this
ceiling; Liang et al.'s own RAovSeg pipeline achieves DSC 0.290 on the
8-subject held-out test set. The 0.19 gap between 0.290 and 0.48 defines
the headroom this dissertation targets.

## 2.3 Pelvic MRI sequences and cross-cohort variability [target: 200 words]

Pelvic MRI protocols acquire multiple sequences per subject. Two
T2-weighted variants dominate the endometriosis workup:

- **T2-weighted (T2)** — high signal from both water and fat. Fat
  appears bright, complicating soft-tissue contrast around fat-embedded
  organs such as the ovaries.
- **T2-weighted fat-suppressed (T2FS)** — active suppression of the fat
  signal via frequency-selective saturation or Dixon methods. Fat
  appears dark; ovarian and endometrial tissue is much more visible
  against the suppressed background.

RAovSeg operates on T2FS specifically because ovary contrast is
substantially higher there. This creates a downstream constraint that
becomes central to Chapter 4: the D2_TCPW cohort within UT-EndoMRI has
T2FS acquisitions, but the D1_MHS cohort (from Memorial Hermann) has
only T2. Cross-cohort augmentation therefore requires implicit or
explicit style transfer between T2 (bright fat) and T2FS (dark fat) — a
non-trivial cross-domain problem examined in Section 2.9.

## 2.4 Automated ovary and pelvic organ segmentation [target: 400 words]

Automated medical image segmentation has been dominated by
encoder-decoder convolutional architectures since the introduction of
U-Net (Ronneberger et al., 2015). U-Net's skip connections between
symmetric encoder and decoder blocks preserve high-resolution spatial
information while allowing deep feature extraction, and remain the
default backbone for medical segmentation tasks including pelvic MRI.

**Attention U-Net** (Oktay et al., 2018) extends U-Net with soft-
attention gates on the skip connections. The gates learn to suppress
irrelevant regions and highlight target structures, improving DSC in
small-object segmentation tasks where the target occupies only a small
fraction of the input volume. Ovaries typically account for less than
1% of pelvic MRI slice pixels — a regime where attention gates
demonstrably help.

**nnU-Net** (Isensee et al., 2021) automates the entire training
pipeline (preprocessing, architecture choice, hyperparameter selection)
around a U-Net backbone and has become the de-facto benchmark for
medical segmentation. Published nnU-Net baselines on UT-EndoMRI
achieve DSC 0.272 — comparable to RAovSeg's 0.290 — suggesting the
n=30 data-scarcity constraint, rather than architecture choice, limits
performance.

**Two-stage pipelines** decouple slice-level classification (does this
slice contain the target?) from voxel-level segmentation. Filtering out
target-negative slices before segmentation reduces false-positive rate
substantially: RAovSeg's ablation shows full-pipeline DSC of 0.290 vs
DSC 0.013 when the slice classifier is removed (Liang et al., 2025).

**RAovSeg** (Liang et al., 2025) is this dissertation's downstream
anchor. It is a two-stage pipeline consisting of:
1. **ResClass** — a 2-block ResNet slice classifier ("does this slice
   contain ovary?"), features [8, 16], BCE loss.
2. **AttUSeg** — a MONAI Attention U-Net with channels [16, 32, 64, 128],
   trained only on ovary-positive slices with Focal Tversky loss
   (α=0.8, β=0.2) that heavily penalises false negatives.
3. **Morphological postprocessing** (closing + largest connected
   component filter).

RAovSeg's preprocessing chain includes a critical intensity-enhancement
rule: post-normalisation voxels in the intensity band [0.22, 0.30] are
saturated to 1, visually highlighting ovary tissue on real T2FS. This
hidden preprocessing assumption turns out to drive success or failure
of synthetic augmentation, as Chapter 4 shows.

**Focal Tversky loss** (Abraham & Khan, 2019) is standard for
class-imbalanced medical segmentation. Its asymmetric α, β parameters
encode the clinical bias toward not missing the target — the
appropriate loss for ovary segmentation where false negatives (missing
an ovary) are clinically more damaging than false positives.

## 2.5 Data scarcity in clinical medical imaging [target: 250 words]

Clinical medical imaging datasets are chronically small compared to
natural-image benchmarks. Expert annotation is expensive, time-consuming,
and requires clinical training. Patient privacy constrains
inter-institutional sharing. The result is a data regime where 30–200
labelled subjects is common, and where n<50 real subjects is treated
as a distinct sub-regime with its own methodological challenges
(Zhang et al., 2022).

Three consequences follow at n<50:

1. **DSC ceilings sit well below inter-rater agreement.** The
   automated-vs-inter-rater gap widens as training-set size shrinks;
   the RAovSeg baseline (0.290 vs 0.48 inter-rater) is characteristic.
2. **Per-subject variance dominates cross-seed variance.** Standard
   n=3 seed reporting risks over-interpreting a single lucky draw
   (Bouthillier et al., 2021). The variance study in Chapter 4 (§4.4)
   demonstrates this: v3 SPADE mean drops from 0.218 (n=3) to 0.178
   (n=8).
3. **Universal-failure subjects.** Certain subjects consistently fail
   segmentation across all training runs, capping the achievable mean
   DSC regardless of methodological improvements.

The n<50 regime motivates augmentation strategies that go beyond
classical affine transformations, and provides the specific setting in
which this dissertation's negative result holds. Positive
augmentation results in the literature are almost always reported at
n>100 subjects.

## 2.6 Data augmentation approaches [target: 200 words]

Classical augmentation applies label-preserving geometric and
photometric transformations at training time: rotations, translations,
scaling, elastic deformation, intensity jitter (Perez & Wang, 2017;
Shorten & Khoshgoftaar, 2019). Both stages of RAovSeg use RandAffine
(±25° rotation, ±25 px translation, 5× multiplier per epoch) as its
classical augmentation baseline.

**MixUp** (Zhang et al., 2018) and **CutMix** (Yun et al., 2019) blend
pairs of training images and their labels. Effective in classification
but less commonly reported for pixel-level segmentation because the
blended targets are ambiguous.

**Model-based augmentation** synthesises training data via a generative
model, aiming to expand the effective training distribution beyond what
classical augmentation can produce (Chen et al., 2019). This is the
paradigm the dissertation evaluates. At n>100 real subjects, positive
results are common. At n<50 the evidence is mixed, with several studies
reporting neutral or negative downstream effects when the generator
distribution diverges from the real distribution — a failure mode the
dissertation quantifies precisely (Chapter 4, §4.5).

## 2.7 Generative modelling in medical imaging [target: 500 words]

Three generative paradigms have been applied to medical image synthesis:
generative adversarial networks (GANs), variational autoencoders
(VAEs), and denoising diffusion probabilistic models (DDPMs).

### 2.7.1 Generative adversarial networks

**GANs** (Goodfellow et al., 2014) train a generator against a
discriminator in a two-player minimax game. In medical imaging, GAN
variants include StyleGAN (Karras et al., 2019) for high-fidelity
unconditional synthesis, Pix2Pix (Isola et al., 2017) for paired
image-to-image translation, and CycleGAN (Zhu et al., 2017) for
unpaired translation. GANs achieve impressive sample quality on large
datasets but are notoriously mode-collapse-prone at n<50, making them
poor candidates for the low-data pelvic MRI regime this dissertation
occupies.

### 2.7.2 Variational autoencoders

**VAEs** (Kingma & Welling, 2014) model data as a latent-variable
distribution and optimise an evidence lower bound. VAEs are stable
trainers at low n but produce characteristically blurry outputs that
would not survive RAovSeg's intensity-enhancement preprocessing —
blurry synthetic ovaries would fail to fire the [0.22, 0.30]
enhancement rule, defeating the augmentation goal from the start.

### 2.7.3 Denoising diffusion probabilistic models

**DDPMs** (Ho et al., 2020) define a Markov forward process that
gradually adds Gaussian noise to an image over T timesteps, and train
a neural network to reverse it. DDPMs have become the dominant
generative paradigm for medical imaging in the past three years
(Kazerouni et al., 2023) due to three properties: stable training at
low n, high sample quality, and straightforward conditioning
mechanisms.

**DDIM** (Song et al., 2020) is a deterministic inference-time
sampler that produces high-quality samples in ~100 steps versus DDPM's
1000, making inference tractable in wall-clock time.

**Classifier-Free Guidance (CFG)** (Ho & Salimans, 2022) trains the
DDPM with 10% probability of dropped conditioning, then combines
conditional and unconditional predictions at inference to amplify the
conditioning effect. CFG substantially improves label-image spatial
correspondence and is used throughout this dissertation.

**Med-DDPM** (Dorjsembe et al., 2024) demonstrates 3D DDPMs on brain,
knee, and pelvic MRI with positive downstream augmentation results at
n>100 training subjects. **RoentGen** (Chambon et al., 2022)
synthesises chest X-rays with radiology-report conditioning, showing
clear downstream benefit on disease classification. **MONAI Generative**
(Pinaya et al., 2023) provides reference 2D and 3D DDPM
implementations for medical imaging — the base library this
dissertation's generator subclasses.

The pattern across positive DDPM augmentation results is training
cohorts substantially larger than this dissertation's n=30. Whether the
DDPM paradigm generalises to the n<50 regime is the empirical question
Chapter 4 answers.

## 2.8 Conditional diffusion for label-aware synthesis [target: 400 words]

For segmentation-augmentation applications, the DDPM must generate
images conditioned on the target anatomy label. Two conditioning
mechanisms anchor this dissertation's architectural ablation.

### 2.8.1 Concatenation conditioning

The label map is concatenated to the noisy image at the U-Net input as
extra channels. The label then propagates through the encoder and
decoder as ordinary feature-map data. The mechanism is simple, requires
no architectural surgery beyond widening the input Conv, and works well
for global scene properties (overall shape, tissue distribution). Its
weakness — empirically shown in this dissertation — is that concat
tends to use the label *globally* rather than per-organ, producing
synth without localised organ-textured content.

### 2.8.2 Spatially-Adaptive Normalization (SPADE)

**SPADE** (Park et al., 2019, "Semantic Image Synthesis with SPADE")
removes the label from the input and injects it at every decoder
ResBlock via modules that modulate feature-map normalisation with
per-pixel γ and β. The critical property is per-pixel modulation: a
label change at pixel (x, y) directly modulates the feature map at (x,
y). SPADE was originally introduced for semantic image synthesis on
natural images (Cityscapes, ADE20K) and has been adapted to medical
imaging in Semantic Diffusion Models (Wang et al., 2022).

A zero-initialisation of the γ and β heads is standard practice: SPADE
starts as pure GroupNorm (identity-like) and learns modulation from
scratch. The dissertation's first SPADE run failed because zero-init
was not applied — chaotic modulation from step 0 prevented the
diffusion loss from converging.

### 2.8.3 Adversarial regularisation of diffusion

Adding a discriminator loss to a DDPM training objective combines the
stability of denoising diffusion with the texture-realism pressure of
adversarial training. **PatchGAN** (Isola et al., 2017) is a fully
convolutional discriminator producing per-patch real/fake logits over
a 70×70 receptive field. Local discrimination enforces texture realism
without the global mode-collapse risk of scalar-output discriminators.

**Spectral normalisation** (Miyato et al., 2018) constrains each
discriminator layer's Lipschitz constant to prevent D from saturating
to 100% accuracy — a common GAN failure that would kill G's gradient.
Spectral norm is what makes stable joint DDPM + adversarial training
possible at the low λ_peak = 0.01 the dissertation uses.

The **DDGAN** family (Xiao et al., 2022) combines diffusion with
adversarial denoising, showing that adversarial regularisation can
substantially improve DDPM sample quality. The dissertation's 1c
variants apply the same idea to the medical augmentation setting.

## 2.9 Cross-domain MRI translation [target: 200 words]

Cross-domain translation aims to map images from one modality or site
to another. Two paradigms dominate:

**Paired translation** requires per-subject matched images across
domains. **Pix2Pix** (Isola et al., 2017) applies a conditional GAN to
paired data, achieving strong results when paired data exists.

**Unpaired translation** removes the paired-data requirement.
**CycleGAN** (Zhu et al., 2017) enforces bidirectional cycle
consistency; **MUNIT** (Huang et al., 2018) factorises content and
style into shared and domain-specific latents. Unpaired methods have
been applied to T1 ↔ T2 MRI translation (Yang et al., 2020) and
CT ↔ MRI translation (Wolterink et al., 2017) with mixed downstream
success.

For the T2 → T2FS translation this dissertation attempts in Phase 2,
no per-subject T2/T2FS pairs exist in UT-EndoMRI. The Phase 2 setup
therefore takes a one-way DDPM + adversarial approach with an
unconditional discriminator anchoring the D2 T2FS style. Chapter 4's
negative Phase 2 result contributes to the growing literature
documenting failure modes of unpaired cross-domain medical translation
at low real-data scales.

## 2.10 Research gap and positioning [target: 150 words]

Three gaps emerge from the review:

1. **DDPM-based augmentation at n<50 is under-evaluated.** Positive
   published results cluster at n>100 subjects. The pelvic MRI
   endometriosis regime (n=30) sits below this threshold, and its
   augmentation calculus is not established.

2. **Concat vs SPADE conditioning has not been quantitatively
   compared** in medical DDPMs on a task where per-organ localisation
   should matter. This dissertation contributes such a comparison
   using a novel Counterfactual Localisation Ratio (CLR) metric.

3. **Downstream-preprocessing-aware evaluation is missing** from the
   augmentation literature. Aggregate metrics (FID, hist_KL) do not
   measure whether generated images satisfy the downstream consumer's
   preprocessing assumptions. This dissertation shows that these
   assumptions are the dominant lever, and proposes evaluation
   protocols that surface them.

## 2.11 Summary [target: 50 words]

The pelvic MRI ovary-segmentation task at n=30 sits in a
data-scarcity regime where classical augmentation is inadequate and
generative augmentation is unproven. Conditional DDPMs with concat or
SPADE conditioning, optionally regularised by PatchGAN, are the
natural candidates. Chapter 3 details the methodology; Chapter 4
reports what happened.
