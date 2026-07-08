# 02 — Background and related work

> Theoretical foundations and prior art. Written to support a dissertation
> chapter — each block includes the "what it is + why we chose it + what
> reader needs from it" chain. Source: `../docs_archive/synthetic_mri_generator_design.md`,
> `../docs_archive/architecture_dataflow_v2.md`, `../docs_archive/TRAINING_OVERVIEW.md`, per-experiment
> summaries.

---

## 2.1 Denoising diffusion probabilistic models (DDPMs)

The generator is a conditional DDPM in the Ho et al. (2020) formulation.
A DDPM defines a forward Markov process that gradually adds Gaussian
noise to an image over T timesteps:

```
q(x_t | x_0) = N(x_t ; √(ᾱ_t) x_0, (1 − ᾱ_t) I)
```

where ᾱ_t is a decreasing schedule (we use linear β schedule, T=1000).
A neural network ε_θ(x_t, t, c) learns to predict the noise given the
noisy image, timestep, and any conditioning c. Training minimises

```
L_diff = E_{t, x_0, ε} [ ‖ε − ε_θ(x_t, t, c)‖² ]
```

At inference, we start from x_T ~ N(0, I) and denoise using DDIM (Song
et al. 2020) rather than the full DDPM sampler — a deterministic sampler
that produces high-quality samples in ~50–100 steps versus DDPM's 1000.
DDIM is standard practice; the only relevant hyperparameter we set is
`num_inference_steps = 100`.

**Why DDPM (not GAN, not VAE, not latent diffusion) as the synthesis
primitive:**
- Stability at low training data. GANs are notoriously mode-collapse-prone
  at n=30 subjects; VAEs produce blurry outputs that would not challenge
  the RAovSeg intensity enhancement rule.
- Well-understood conditioning mechanisms (concat, cross-attention, SPADE)
  that support a clean architectural ablation.
- The MONAI Generative implementation gives a mature 2D UNet backbone we
  can subclass for the SPADE variant.
- Cost tolerable at 512×512 axial slices with modest channel widths on a
  single A100.

### Classifier-Free Guidance (CFG)

Ho & Salimans (2022) introduced CFG: train the network with a small
probability (10%) of dropping the conditioning c → the same network can
predict both conditional and unconditional noise. At inference, combine
them:

```
ε̂ = ε_θ(x_t, t, ∅) + g · (ε_θ(x_t, t, c) − ε_θ(x_t, t, ∅))
```

where g is the guidance scale. Higher g amplifies the conditioning
signal at inference at the cost of some sample diversity.

CFG was added mid-Exp 1a after early samples showed weak spatial
alignment between the anatomy label and the synthesised organs. Enabling
CFG substantially improved organ-position fidelity. Per-variant optima
(from the Tier 1 inference sweep): g = 3.0 for concat variants (1a, 1c_concat),
g = 2.0 for SPADE variants (1b, 1c_spade). SPADE prefers lower guidance
because its per-pixel modulation already provides a strong spatial signal
that CFG amplification easily oversaturates.

### Exponential Moving Average (EMA)

We maintain an EMA of the generator weights with decay 0.9999. Inference
uses the EMA weights, not the raw training weights. This is a standard
diffusion-model practice that noticeably smooths sample textures. EMA
was added mid-Exp 1a alongside CFG as a quality-of-life improvement;
inherited unchanged across 1b, 1c_concat, 1c_spade for ablation parity.

## 2.2 Label-conditioning mechanisms — the crux of the ablation

Conditional DDPMs need to inject the label information somewhere. Two
canonical choices anchor our Phase 1 ablation.

### Concat conditioning (baseline, Exp 1a and 1c_concat)

The label map is concatenated to the noisy image as extra channels at the
UNet input. In our case: 1 image channel + 6 label channels = 7 input
channels. The label then propagates through the entire encoder and
decoder as ordinary feature-map data.

Properties:
- Simple, no architectural surgery required (just wider input Conv).
- The label is "ambient" — every layer sees it, but the model chooses how
  to use it. In practice it tends to use it *globally* rather than
  per-organ (see CLR results in Chapter 5).
- Works well for global scene properties (overall body shape, tissue
  intensity distributions).

### SPADE conditioning (Exp 1b and 1c_spade)

**Spatially Adaptive Normalization** (Park et al. 2019, "Semantic Image
Synthesis with SPADE"). The label is *removed* from the UNet input
entirely; input is just the noisy image (1 channel). The label instead
enters at every decoder ResBlock through SPADE modules that modulate
feature-map normalisation with per-pixel γ and β:

```
SPADE(feat, label)(x, y):
  normalised = GroupNorm(feat)(x, y)
  γ(x, y), β(x, y) = MLP_label(label at x, y)
  output(x, y) = normalised(x, y) × (1 + γ(x, y)) + β(x, y)
```

Properties:
- Per-pixel modulation: a label change at pixel (x, y) only *directly*
  modulates the feature map at (x, y). Downstream convolutions still
  spread the effect, but the modulation itself is local.
- Zero-initialising γ and β makes SPADE start as pure GroupNorm
  (identity-like); the model then *learns* modulation from scratch. Our
  first 1b run failed because default random init made γ chaotic from
  step 0 — samples remained noise after 40k steps. Zero-init is standard
  practice (DiT, Imagen, Semantic Diffusion Models) and was retroactively
  applied.
- Custom implementation required: MONAI's ResBlock forward does not
  accept a label argument, so we subclassed to build
  `DiffusionUNetSPADE` (`src/Generator/unet_spade.py`).

The empirical difference between concat and SPADE conditioning is the
central axis of the Phase 1 ablation. It shows up in the
**Counterfactual Localisation Ratio (CLR)** metric — SPADE achieves 10-30×
higher CLR than concat, meaning "removing the uterus channel changes
mostly the uterus region" is a property the SPADE variants have and the
concat variants do not.

### CLR — Counterfactual Localisation Ratio

Because standard image-quality metrics (FID, hist_KL) do not measure
per-channel controllability, we introduce a diagnostic metric:

```
For each label channel c:
  Regenerate synth with the same initial noise but channel c zeroed
  Compute per-pixel change ‖x_synth − x_synth_ablated‖²
  CLR(c) = ‖change‖² inside channel-c's mask / ‖change‖² over the whole image
```

- CLR → 1: removing channel c only changes its own region → local, clean
  conditioning.
- CLR → 0: removing channel c changes the image everywhere → global,
  mixed conditioning.

CLR is essential to the paper story because it *predicts* Phase 1
downstream failure. Concat's CLR ≈ 0.03 is the architectural reason its
downstream augmentation is unrescuable by preprocessing tuning — there is
no per-organ signal in the synth for the fixes to align with.

### OSI — Organ Specificity Index (SPADE-only)

Complementary metric for SPADE variants: for each SPADE decoder module,
compute the Pearson correlation between |γ| and each organ mask, versus
the body mask.

- OSI_max_organ_corr > 0: γ correlates with at least one organ region →
  SPADE is doing per-organ work.
- OSI_body_corr ≈ 0: γ is not merely encoding "inside vs outside body"
  (which the outside_body and body_other channels already carry).

Both 1b and 1c_spade show organ_corr ≈ 0.25 and body_corr ≈ 0. SPADE γ
heads pick up per-organ structure at the module level.

## 2.3 Adversarial regularisation of diffusion — the PatchGAN arm

### Conditional PatchGAN (Isola et al. 2017)

The Pix2Pix paper introduced PatchGAN — a fully convolutional
discriminator that outputs a grid of real/fake logits over
overlapping receptive fields (in our case 70×70 receptive field over a
32×32 output grid). Advantages over global-image D: local discrimination
enforces texture realism without a single "is this a real image" scalar
that would push toward mode collapse.

We use the **conditional** variant: D receives `concat(image, label)`
as its 7-channel input. It learns *both* image realism and image-label
consistency simultaneously.

### Spectral normalisation (Miyato et al. 2018)

Applied to every weight layer in the discriminator. Constrains the
Lipschitz constant of each layer to ≤ 1, which prevents D from
saturating to 100% training accuracy and killing G's gradient — a
well-known GAN failure mode. Spectral norm is what makes the adversarial
loss stably trainable at λ_peak = 0.01 without needing gradient penalty
or R1 regularisation.

### λ schedule

Naive addition of adversarial loss to a diffusion training loop
destabilises early training. Our schedule:

```
Warmup  0 → 10k steps:  λ = 0        (pure DDPM MSE, G stabilises)
Ramp    10k → 30k:      λ = 0 → 0.01 (linear ramp)
Steady  30k → 100k:     λ = 0.01     (joint training at full pressure)
```

Total training steps for the 1c variants: 100k, vs 80k for 1a/1b. The
extra 20k gives 90k effective adversarial steps.

### x̂_0 estimate for the adversarial pass

D judges real vs fake at the *image level*, but the generator predicts
*noise*. We invert on the fly:

```
x̂_0 = (x_t − √(1 − ᾱ_t) · ε̂) / √(ᾱ_t)
```

This is a cheap single-step estimate of the clean image from the noise
prediction — not as good as a full DDIM sampler, but adequate for the
adversarial pass and avoids the multi-step sampling cost per training
step. D then sees `concat(x̂_0, label)` on the fake side.

### Empirical takeaway from the 1c arm

PatchGAN affects the two backbones differently:
- On concat: big texture-realism gains (FID −12%, hist_KL −29%),
  localisation still poor (CLR unchanged).
- On SPADE: best perceptual realism (LPIPS 0.699, best of all 4 variants),
  localisation preserved.

This asymmetric effect is one of the two main claims of the
architectural ablation (Chapter 5): PatchGAN is not a generic realism
booster; it delivers different benefits depending on what the
conditioning mechanism *lacked*.

## 2.4 Medical image synthesis for segmentation augmentation

Prior work in the direction of using generative models to augment
downstream segmentation:

- **Med-DDPM** (Dorjsembe et al. 2024): 3D diffusion model for
  brain/knee/pelvic MRI, demonstrating that class-conditional DDPMs can
  produce useful downstream data at moderate cohort sizes (>100
  subjects).
- **RoentGen** (Chambon et al. 2022): chest X-ray synthesis with
  radiology-report conditioning — clear positive downstream result on
  disease-classification tasks.
- **MONAI Generative examples** (Pinaya et al.): 2D and 3D DDPM
  reference implementations for medical imaging, including SPADE-like
  conditioning.
- **Pix2Pix / CycleGAN** for cross-modality translation (T1 ↔ T2, CT ↔
  MRI) — image-to-image translation is the closest paradigm to our Phase
  2 setup.

Common pattern in these positive results: **large cohort sizes** (>500
subjects in the training pool), often coupled with paired training data
(same-subject T1/T2 pairs) that give the generator a strong supervised
signal.

**Our regime differs in two ways**:
1. n=30 real training subjects is at least 10× smaller than most
   published positive-augmentation studies.
2. No paired T2/T2FS data on a per-subject basis — D2 has T2FS only, D1
   has T2 only, no same-subject T2 ↔ T2FS pairs exist in UT-EndoMRI.

This regime is under-explored, and the dissertation's core empirical
finding is that the standard recipe does not carry over. §6 discusses
what would.

## 2.5 Cross-domain MRI translation (Phase 2 background)

Phase 2 tackles the cross-domain question: can a generator trained on
D1_MHS T2 (bright fat) produce D2_TCPW T2FS-styled synth (dark fat) via
an adversarial signal from a D2-anchored discriminator?

Prior art:
- **CycleGAN** (Zhu et al. 2017): unpaired image-to-image translation
  via cycle consistency. Standard in medical unpaired translation.
- **Pix2Pix** (Isola et al. 2017): paired image-to-image translation.
  Cleaner supervision when pairs exist.
- **UNIT / MUNIT** (Liu et al.): shared latent space for unpaired
  translation with style codes.

Our Phase 2 architecture is closest to a Pix2Pix + DDPM hybrid: the
generator's job is (i) reconstruct the anatomy specified by the label
via DDPM MSE (grounded in D1), and (ii) match the D2 T2FS *style* via
adversarial pressure. There is no cycle consistency — the setup is
one-way — and no paired supervision.

The Phase 2 failure (§5.5) is directly attributable to the DDPM MSE loss
on D1 T2 dominating the adversarial signal for T2FS style transfer. The
two objectives are antagonistic at λ_peak = 0.01, and the balance found
does not favour style transfer. This is discussed in Chapter 6 as
evidence that **cross-domain synthesis via DDPM + PatchGAN with the
standard adversarial schedule is architecturally insufficient** for the
n<50 real, no-pairs regime.

## 2.6 RAovSeg — the downstream anchor

Liang et al. (2025, *Scientific Data*, "Deep learning for automated
segmentation of the ovaries on endometriosis MRI: an open dataset and
methodology"). The paper introduces UT-EndoMRI and RAovSeg together.

### Architecture (two-stage)

**Stage 1 — ResClass (slice classifier):** binary classifier over slices
that predicts "does this slice contain ovary?" A small 2-block ResNet
(features [8, 16]) trained with BCE loss + RandAffine augmentation (5×
multiplier). Purpose: filter out ovary-negative slices before feeding
Stage 2, reducing false positives.

**Stage 2 — AttUSeg (segmentation):** MONAI Attention U-Net with
channels [16, 32, 64, 128] and strides (2, 2, 2). Trained only on
ovary-containing slices, Focal Tversky loss (α=0.8, β=0.2, γ=1.33) — the
asymmetric weights penalise false negatives heavily, reflecting the
clinical bias toward "do not miss the ovary."

**Stage 3 — Postprocessing:** per-volume, morphological closing
(iterations=10) + largest connected component filter.

**Stage 4 — Evaluation:** three DSC variants reported:
- `full`: ResClass + AttUSeg + postprocess → the headline metric.
- `no_postprocess`: ResClass + AttUSeg only.
- `no_resclass`: AttUSeg on every slice (skip Stage 1).

Paper's reported numbers: full = 0.290, no_postprocess = 0.235,
no_resclass = 0.013 — the drop to 0.013 when the slice classifier is
removed shows how central Stage 1 is to the pipeline's performance.

### Preprocessing — the pipeline's central assumption

RAovSeg applies a fixed preprocessing chain per subject:

```
1. Load T2FS NIfTI (SimpleITK)
2. Resample to 0.35 × 0.35 × 6.0 mm, 512 × 512 × native_z
3. Percentile clip to [1st, 99th], min-max normalise to [0, 1]
4. Ovary enhancement — the critical step:
     if 0.22 ≤ I ≤ 0.30: I = 1     (highlight ovary intensity band)
     elif I ≥ 0.5:       I = 1 − I (invert bright tissue)
     else:               I = I     (leave dark tissue)
5. Save image.npy and ov_label.npy
```

The **ovary enhancement rule at [0.22, 0.30] → 1** is the pipeline's
hidden assumption: after percentile-clip + min-max, real T2FS ovary
tissue lands in this intensity band, so the rule visually saturates the
ovary region to white. The AttUSeg then trains on images where the
ovary is a saturated white blob against a background of gradient tissue.

This rule turns out to be the single most consequential piece of the
whole pipeline for the augmentation story. When synth doesn't satisfy
the assumption (its ovary lands elsewhere in intensity), the enhancement
either doesn't fire on synth (making synth ovaries visually invisible to
AttUSeg) or fires in the wrong location (creating false training
signals). Chapter 5 traces this thread across v1 → v2 → v3 fixes.

### Inclusion criteria (for both real and synth subjects)

Paper's criteria for train_val / test membership:
1. Has T2FS image (`{id}_T2FS.nii.gz`).
2. Has ovary label (`{id}_ov.nii.gz`).
3. Does NOT have cyst label (`{id}_cy.nii.gz`).
4. Does NOT have endometrioma label (`{id}_em.nii.gz`).

Our augmentation setup adds a `--extra-train-dir` flag to `preprocess.py`
that scans a second directory for synthetic D2-9XX subjects and forces
them into the train_val split. The test split (8 sacred D2 subjects)
remains untouched.

## 2.7 Where the dissertation sits relative to the literature

- **Vs medical image synthesis for augmentation broadly**: our contribution
  is a rigorous *negative* result at n<50 subjects, complementing the
  positive results in larger-cohort settings.
- **Vs SPADE-conditioned diffusion (SDM, Semantic Diffusion Models)**:
  we replicate SPADE's per-organ localisation quantitatively (via CLR)
  and show that this localisation is *necessary but not sufficient* for
  downstream augmentation benefit.
- **Vs adversarial-augmented diffusion (DDGAN, ADM-G)**: we
  characterise PatchGAN's *asymmetric* effect on different conditioning
  mechanisms — a finding that would inform anyone combining these
  techniques.
- **Vs cross-domain MRI translation (CycleGAN, MUNIT)**: our Phase 2
  negative result documents a specific failure mode (DDPM MSE dominating
  adversarial signal in T2 → T2FS transfer at λ = 0.01, 100k steps) that
  the field's positive cross-domain results have not surfaced.
- **Vs downstream augmentation methodology**: we motivate a variance
  study protocol (n≥5 seeds, per-subject analysis) that the field's
  standard n=3 seed reporting would miss.
