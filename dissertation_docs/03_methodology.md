# 3 — Methodology

> **Target: 3,000 words.** Merges the dataset, downstream pipeline, and
> generator design into one methodology chapter (the pattern used in
> both distinction dissertations aca22mmm and acu23ns).

---

## 3.1 Overview [target: 100 words]

The methodology combines a downstream ovary-segmentation pipeline
(Section 3.2, RAovSeg recreation) with a family of conditional-DDPM
generators trained to synthesise pelvic T2FS MRI (Section 3.4).
Sections 3.3 details the UT-EndoMRI dataset, subject filtering, and
the 6-channel label design. Section 3.5 describes the training loop,
including the joint DDPM + adversarial objective. Section 3.6
introduces the body-centered vs image-centered preprocessing mismatch
that drives the augmentation experiments' failure modes. Section 3.7
specifies the cross-domain Phase 2 configuration. Section 3.8
enumerates the evaluation metrics and downstream protocol.

## 3.2 Dataset — UT-EndoMRI [target: 350 words]

The dataset used throughout this dissertation is UT-EndoMRI (Liang et
al., 2025), an open-source pelvic MRI cohort for endometriosis
research. It comprises two site cohorts:

| Cohort | Site | Sequence | Subjects | Fat suppression |
|---|---|---|---|---|
| D1_MHS | Memorial Hermann | T2-weighted | 51 | No (bright fat) |
| D2_TCPW | TCPW | T2-weighted fat-suppressed (T2FS) | ~73 | Yes (dark fat) |

Each subject volume carries per-organ manual annotations:
- `{id}_T2FS.nii.gz` or `{id}_T2.nii.gz` — the image volume.
- `{id}_ut.nii.gz` — uterus mask.
- `{id}_ov.nii.gz` — combined left+right ovary mask.
- `{id}_em.nii.gz` — endometrioma mask (present only for em-positive
  subjects).
- `{id}_cy.nii.gz` — cyst mask (present only for cyst-positive
  subjects, which are excluded from RAovSeg).

The Left vs Right ovary split for label conditioning is computed in
image-space by connected-component analysis on the combined ovary
mask; each connected component is labelled `ov_L` or `ov_R` by its
x-coordinate relative to the image midline.

### 3.2.1 Subject filtering — D2 primary cohort

Of the ~73 raw D2 subjects, three filters reduce the pool:

| Filter | Removed | Rationale |
|---|---|---|
| Missing T2FS image | 3 | Cannot train without image |
| Missing uterus or ovary mask | ~9 | Required by generator inclusion filter |
| RAovSeg test subjects (sacred) | 8 | Held out for downstream evaluation |
| **Effective generator training pool** | **32** | |

The RAovSeg training pool applies stricter criteria (excludes
em-positive and cyst-positive subjects), producing 30 train_val
subjects + 8 sacred test subjects. The generator's 32 vs RAovSeg's
30 differ by two em-positive subjects that the generator sees for
realism.

### 3.2.2 The sacred 8-subject test set

Identical across every experiment: real-only baseline, all four Phase
1 downstream runs, the n=8 variance study, and Phase 2 exp2 and
exp2_lam05.

```
D2-005, D2-015, D2-016, D2-017, D2-023, D2-024, D2-026, D2-038
```

Two subjects — **D2-005** and **D2-023** — become universal-failure
cases in the augmented pipeline (DSC = 0 across all 8 seeds of the
variance study), a fact discussed in Chapter 5 as a per-subject
limitation of DSC-mean reporting.

### 3.2.3 D1_MHS — Phase 2 generator training pool

D1 is used exclusively as generator training data in Phase 2. After
filtering (present T2, present ovary, present uterus), ~32 D1
subjects contribute. D1 is never used for downstream evaluation; the
8 D2 test subjects remain the sole evaluation target throughout.

## 3.3 Six-channel label design [target: 300 words]

Labels are stored as 6-channel one-hot tensors at 512 × 512. Each
pixel belongs to exactly one channel:

| Channel | Name | Meaning | Source |
|---|---|---|---|
| 0 | `outside_body` | Air outside body | Computed: 1 − body_mask |
| 1 | `uterus` | Uterus | Manual `_ut.nii.gz` |
| 2 | `ov_L` | Left ovary | Auto-split from `_ov.nii.gz` |
| 3 | `ov_R` | Right ovary | Auto-split from `_ov.nii.gz` |
| 4 | `em` | Endometrioma | Manual `_em.nii.gz` (em+ subjects only) |
| 5 | `body_other` | Non-target body tissue | Computed from body mask minus organs |

Early experiments used a 5-channel label (`outside_body` and
`body_other` collapsed into a single "background" channel). Generated
samples then displayed "noisy grey edges" at the body-air boundary
because the model had no unambiguous signal distinguishing "render as
tissue" from "render as air". Adding the `body_other` channel resolved
this by giving explicit conditioning: every pixel is now assigned to
one of six semantic classes, and the generator learns a clean
"fill this region with plausible non-target tissue" signal.

The 6-channel change was made mid-way through Exp 1a and inherited
unchanged into 1b, 1c_concat, 1c_spade, and Phase 2 experiments to
preserve ablation parity — a principle discussed further in §3.5.

## 3.4 Generator architecture [target: 700 words]

All four Phase 1 variants and the Phase 2 generator share a 2D U-Net
backbone adapted from MONAI Generative's `DiffusionModelUNet` (Pinaya
et al., 2023). The 2×2 ablation crosses two conditioning mechanisms
(concat vs SPADE) with two training regimes (with and without a
PatchGAN adversarial discriminator).

### 3.4.1 Shared backbone

| Attribute | Value |
|---|---|
| Input resolution | 512 × 512 axial slice |
| Levels | 4 (512² → 256² → 128² → 64²) |
| Channel widths | [64, 128, 256, 256] |
| Self-attention | Deepest level (64²) only — memory-limited |
| Diffusion schedule | Linear β from 1e-4 to 2e-2, T = 1000 |
| Inference sampler | DDIM, 100 steps |

The self-attention restriction to the deepest level is a memory
constraint: adding attention at level 2 (128²) OOMs at batch 4 on an
A100 80 GB. The 2D backbone (versus 3D) is a compute choice; 3D
coherence at inference is patched by Inter-Slice Consistent
Stochasticity (§3.5.4).

### 3.4.2 Concat conditioning (Exp 1a, 1c_concat)

The 6-channel label is concatenated to the noisy image at the U-Net
input, producing a 7-channel input. The label propagates through the
entire encoder and decoder as ordinary feature-map data. The
mechanism is architecturally minimal — no changes beyond widening the
first Conv layer — and works well for global scene properties. Its
empirical weakness (Chapter 4, §4.2) is per-organ localisation: the
network chooses to use the label globally rather than per-organ,
producing counterfactual localisation ratios (CLR) of 0.03–0.07
across target channels.

### 3.4.3 SPADE conditioning (Exp 1b, 1c_spade)

**SPADE** (Park et al., 2019) removes the label from the U-Net input
and injects it at every decoder ResBlock via Spatially-Adaptive
Normalization modules:

```
SPADE(feat, label)(x, y):
  normalised = GroupNorm(feat)(x, y)
  shared = ReLU(Conv3×3(label))     # 6 → 64 channels
  γ(x, y) = Conv3×3(shared)         # per-pixel
  β(x, y) = Conv3×3(shared)         # per-pixel
  output(x, y) = normalised × (1 + γ) + β
```

The γ and β heads are **zero-initialised**, so SPADE starts as pure
GroupNorm (identity-like) and the model learns modulation from
scratch. Without zero-init the first 1b run produced fuzzy noise
even after 40k training steps.

MONAI's ResBlock forward does not accept a label argument, so the
SPADE variant is implemented as a subclassed `DiffusionUNetSPADE`
(`src/Generator/unet_spade.py`).

### 3.4.4 PatchGAN discriminator (Exp 1c only)

The 1c variants add a conditional PatchGAN discriminator (Isola et
al., 2017) on top of both backbones — 1c_concat = 1a + PatchGAN,
1c_spade = 1b + PatchGAN — with identical D architecture across both
arms.

| Attribute | Value |
|---|---|
| Input | concat(image, label) — 7 channels |
| Architecture | 5-block PatchGAN, 70×70 receptive field, 32×32 output logits |
| Base channels | 64 → 128 → 256 → 512 → 1 (~2.7 M parameters) |
| Normalisation | Spectral norm (Miyato et al., 2018) |
| Activation | LeakyReLU(0.2) |

The conditional input (image concatenated with label) forces D to
learn both realism and image-label consistency simultaneously.
Spectral norm prevents D from saturating to 100% accuracy and killing
G's gradient — a well-known GAN failure mode.

### 3.4.5 The joint DDPM + adversarial loop

Per training step:

```
1. G forward: predict noise ε̂ from (x_t, label)
2. L_diff = MSE(ε̂, ε_true)                         # DDPM MSE
3. x̂_0 = (x_t − √(1 − ᾱ_t) ε̂) / √(ᾱ_t)           # cheap single-step invert
4. L_D = BCE(D(x_real, label), 1) + BCE(D(x̂_0.detach(), label), 0)
5. Optimizer_D.step()
6. L_adv = BCE(D(x̂_0, label), 1)                   # G wants D fooled
7. L_G = L_diff + λ_t · L_adv
8. Optimizer_G.step()
```

The `x̂_0` estimate — a cheap single-step inversion of the noise
prediction — is not as accurate as a full DDIM sample but is
adequate for D's real/fake decision and avoids the multi-step
sampling cost per training step.

### 3.4.6 Adversarial λ schedule

Naive addition of adversarial loss destabilises early diffusion
training. The schedule that works:

```
Warmup  0 → 10k:      λ = 0
Ramp    10k → 30k:    λ = 0 → 0.01 (linear)
Steady  30k → 100k:   λ = 0.01
```

Phase 2 (§3.7) inherits this schedule with λ_peak = 0.01 (exp2) or
0.05 (exp2_lam05).

## 3.5 Training strategy [target: 350 words]

### 3.5.1 Optimiser and schedule

| Component | Value |
|---|---|
| Optimiser (G) | AdamW, lr 1e-4 |
| Optimiser (D, 1c only) | AdamW, lr 2.5e-5 (¼ of G) |
| Batch size | 4 |
| Steps — 1a, 1b | 80,000 |
| Steps — 1c, exp2 | 100,000 |
| Checkpoint interval | Every 5,000 steps |

At batch 4 × 100k steps, an A100 80 GB run takes ~10–11 h wall clock.

### 3.5.2 Classifier-Free Guidance

CFG (Ho & Salimans, 2022) is applied throughout. Training-time: 10% of
batches use a zero label. Inference-time:

```
ε̂ = ε_uncond + g · (ε_cond − ε_uncond)
```

Per-variant guidance-scale optima (from a Tier 1 inference sweep):
- Concat variants (1a, 1c_concat): g = 3.0
- SPADE variants (1b, 1c_spade): g = 2.0

SPADE prefers lower guidance because per-pixel modulation already
provides strong spatial signal; g > 2.0 oversaturates the SPADE
effect.

### 3.5.3 Exponential Moving Average

EMA of generator weights with decay 0.9999. Inference uses EMA
weights. Added mid-Exp 1a and inherited across all variants.

### 3.5.4 Inter-Slice Consistent Stochasticity (ISCS)

The 2D DDPM produces independent per-slice noise samples, which would
break 3D anatomical coherence in the assembled volume. ISCS mixes a
shared and an independent noise seed:

```
ε_shared ~ N(0, I)                # one draw per volume
For each slice z:
    ε_z_indep ~ N(0, I)
    x_T[z] = 0.8 · ε_shared + 0.6 · ε_z_indep
```

The 0.8/0.6 weighting is a project choice — empirically gives
slice-to-slice coherence without over-smoothing. ISCS is plug-and-play
at inference; no retraining required.

### 3.5.5 Ablation parity — the design principle

Every mid-flight quality-of-life change (CFG, EMA, the 6-channel
label, weighted ovary-slice sampling, zero-init SPADE γ/β) was
retroactively applied to earlier variants before final reporting. The
2×2 architectural ablation is the locked spine of Phase 1;
methodological improvements are permitted only if inherited
identically across all four variants.

## 3.6 Preprocessing — the body-centered vs image-centered mismatch [target: 400 words]

Two preprocessing chains coexist in the pipeline and produce
substantially different framings.

### 3.6.1 Generator preprocessing (body-centered)

`src/Generator/preprocess_for_generator.py`:

```
1. Load raw NIfTI at native spacing/orientation
2. Compute body silhouette (threshold + morphological closing + fill)
3. Bounding-box the body silhouette, crop with 5% margin
4. Resample the crop to 512 × 512 at per-subject in-plane spacing
5. Save image, 6-channel label, and body_silhouette
```

Result: body fills **~90%** of the 512² frame. Outside-body is a thin
border of air.

### 3.6.2 RAovSeg preprocessing (image-centered)

`src/RaovSeg_recreation/preprocess.py`:

```
1. Load raw NIfTI at native spacing/orientation
2. Resample to 0.35 × 0.35 × 6.0 mm at 512 × 512 × native_z
3. Percentile clip [1st, 99th], minmax normalise to [0, 1]
4. Ovary enhancement:
     if 0.22 ≤ I ≤ 0.30:  I = 1
     elif I ≥ 0.5:        I = 1 − I
     else:                I = I
5. Save image and ov_label
```

Result: at 0.35 mm/px × 512², field of view is 179.2 mm — body fills
**~55–60%** of the frame with a substantial black border.

### 3.6.3 The intensity enhancement rule — the pipeline's hidden assumption

Step 4 of RAovSeg preprocessing sets voxels in [0.22, 0.30] to 1. On
real T2FS after percentile-clip + minmax, ovary tissue empirically
falls in this band. The rule visually saturates the ovary region to
white, so the AttUSeg trains on images where the ovary is the
brightest object by construction.

This rule turns out to be the central mechanism of Chapter 4's story:
if synth's ovary intensity does not land in [0.22, 0.30], the
enhancement does not fire on synth. The AttUSeg is then trained on
synth images where the ovary is visually indistinguishable from
surrounding tissue, defeating the augmentation. Chapter 4's v1 → v2 →
v3 preprocessing fixes all target this specific mechanism.

### 3.6.4 Assembly-time preprocessing fixes

The `src/Generator/assemble_synthetic_volumes.py` script exposes four
flags that instantiate the fixes applied progressively across
Chapter 4:

| Flag | Purpose | Introduced at |
|---|---|---|
| `--body-mask` | Kill outside-body hallucinations | v2 |
| `--histogram-match` | Match synth to real intensity distribution | v2 |
| `--resample-to-source` | Transfer synth to real subject's frame | v2 |
| `--ovary-target-intensity 0.26` | Force ovary pixels to enhancement window | v3 |

All are ON in v3. `--no-body-mask`, `--no-histogram-match`, and
`--no-resample-to-source` disable them for the v1 baseline
reproduction.

## 3.7 Phase 2 — cross-domain configuration [target: 300 words]

Phase 2 tests whether a generator trained on D1 T2 anatomy plus a
discriminator trained on D2 T2FS style can produce D2-styled synth
via adversarial pressure alone. The Phase 1 architectural best
(1c_spade — SPADE + PatchGAN) is the starting backbone.

### 3.7.1 Configuration

| Component | Value |
|---|---|
| Generator | SPADE + PatchGAN backbone |
| Generator training data | D1_MHS T2 (32 subjects) |
| Discriminator | Conditional PatchGAN with **label zeroed** (unconditional D) |
| Discriminator training data | D2_TCPW T2FS (41 subjects) |
| λ_peak | 0.01 (exp2) or 0.05 (exp2_lam05) |
| Steps | 100,000 |
| Inference | D1 r1 masks, ovary target t = 0.26 |

### 3.7.2 Why unconditional D

If D receives labels, it can learn a label-distribution shortcut ("D1
labels look like X, D2 labels look like Y — reject anything D1"), which
has nothing to do with T2 vs T2FS style. Zeroing the label input to D
forces D to judge pure pixel-level style. The cost is weaker gradient
signal, which (in retrospect, Chapter 5) contributes to Phase 2's
failure at λ_peak = 0.01.

### 3.7.3 Assembly

All three preprocessing fixes and the ovary target rescale (§3.6.4)
are inherited from Phase 1 v3. 32 synth volumes are assembled per
Phase 2 experiment, resampled into D1 raw subject frames.

## 3.8 Evaluation metrics [target: 500 words]

Two evaluation stages: generator quality (§3.8.1) and downstream
utility (§3.8.2).

### 3.8.1 Generator quality metrics

| Metric | Direction | Purpose |
|---|---|---|
| FID | ↓ | Inception feature-space distance to real |
| hist_KL | ↓ | KL divergence of pixel-intensity histograms |
| LPIPS-NN (min, mean) | ↓ | Perceptual nearest-neighbour distance to real |
| CLR (Counterfactual Localisation Ratio) | ↑ | Per-organ label-image consistency |
| OSI (Organ Specificity Index, SPADE-only) | ↑ | γ-vs-organ correlation |

**CLR** is introduced as a novel metric in this dissertation. For each
label channel c:

```
Regenerate synth with channel c zeroed, same initial noise
Δ = ‖x_synth − x_synth_c_ablated‖²
CLR(c) = Δ inside c's mask / Δ over the whole image
```

CLR → 1 indicates the network uses channel c for its own region only
(local, clean conditioning); CLR → 0 indicates the effect is global
(mixed conditioning). CLR is essential to the paper story because it
predicts Phase 1 downstream failure: concat's CLR ≈ 0.03 is the
architectural reason its augmentation is unrescuable by preprocessing.

**OSI** (SPADE-only) is the per-SPADE-module Pearson correlation
between |γ| and each organ mask vs the body mask. OSI_max_organ > 0
and OSI_body ≈ 0 together indicate SPADE γ heads pick up per-organ
structure, not just body silhouette.

FID sample count is N = 256 (noise floor ~±30 at this sample size).
CLR/OSI are computed at N = 4 test labels.

### 3.8.2 Downstream metrics — RAovSeg ovary DSC

The downstream metric is the Dice similarity coefficient (DSC) on the
8 sacred D2 test subjects. `evaluate.py` reports three DSC variants:

| Variant | Pipeline | Purpose |
|---|---|---|
| `full` | ResClass + AttUSeg + postprocess | Headline metric |
| `no_postprocess` | ResClass + AttUSeg | Isolates postprocess |
| `no_resclass` | AttUSeg on every slice | Isolates ResClass |

Paper benchmarks (Liang et al., 2025): full 0.290, no_postprocess
0.235, no_resclass 0.013.

### 3.8.3 Variance study protocol

Standard n=3 seed reporting produces wide confidence intervals at this
data scale. The variance study reruns the v3 SPADE configuration for
5 additional seeds (seeds 3–7) and reports:
- Aggregate mean and standard deviation across seeds.
- Per-subject DSC distribution across the 8 test subjects.
- Universal-failure identification (subjects with DSC = 0 across all
  8 seeds).

The variance study (Chapter 4, §4.4) revises the v3 SPADE mean from
0.218 (n=3) to 0.178 (n=8), demonstrating why n≥5 should be the
minimum reporting standard for augmentation experiments at n<50 real
subjects.
