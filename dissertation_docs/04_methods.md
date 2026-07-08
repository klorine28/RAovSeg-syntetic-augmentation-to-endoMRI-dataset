# 04 — Methods

> Full generator design across all four Phase 1 variants and the Phase 2
> cross-domain configuration. Source: `../docs_archive/EXP1B_SUMMARY.md`, `../docs_archive/EXP1C_SUMMARY.md`,
> `../docs_archive/RESULTS_2x2.md`, `../docs_archive/synthetic_mri_generator_design.md`,
> `../docs_archive/architecture_dataflow_v2.md`, memory files.

---

## 4.1 The design principle — ablation as the spine

Every method-side choice below exists to be *isolated*. The 2×2
architectural ablation (concat vs SPADE conditioning × ±PatchGAN
adversarial regularisation) is the locked spine of Phase 1. Every
quality-of-life improvement discovered mid-flight (CFG, EMA, 6th
`body_other` channel, zero-initialisation of SPADE γ/β heads) was
retroactively inherited across all four variants to preserve ablation
fairness. This principle — "the ablation is the spine; QoL changes must
be inherited across variants" — is explicit in the project's memory
files and governs every methodological decision below.

## 4.2 Shared backbone — 2D conditional DDPM

All four Phase 1 variants and the Phase 2 generator share the same 2D
UNet backbone, adapted from MONAI Generative's `DiffusionModelUNet`.

### 4.2.1 Backbone dimensions

| Attribute | Value | Rationale |
|---|---|---|
| Input resolution | 512 × 512 axial slice | Post-body-centered-resample native resolution |
| Levels | 4 (512² → 256² → 128² → 64²) | Standard depth for 512² diffusion |
| Channel widths | `[64, 128, 256, 256]` | Balanced for a single A100 80GB at batch 4 |
| Self-attention | Deepest level (64²) only | Memory budget — attention at level 2 (128²) OOMs at batch 4 |
| Time embedding | Sinusoidal positional + 2-layer MLP | Standard DDPM practice |
| Total parameters (concat) | 25.3 M | 1a |
| Total parameters (SPADE) | 23.9 M | 1b (slightly fewer because SPADE moves label out of the input) |
| Discriminator (1c only) | 5-block PatchGAN, ~2.7 M | Separate optimiser |

### 4.2.2 Diffusion schedule

Standard Ho et al. (2020) formulation. Linear β schedule from β_start =
1e-4 to β_end = 2e-2, T = 1000 timesteps. Training loss:
`MSE(ε̂, true_noise)`. Inference uses DDIM (Song et al. 2020) at 100
steps, deterministic sampler with η = 0.

### 4.2.3 Optimiser and schedule

| Attribute | Value |
|---|---|
| Optimiser (G) | AdamW, lr = 1e-4, weight_decay = 0 |
| Optimiser (D, 1c only) | AdamW, lr = 2.5e-5 (¼ of G's), weight_decay = 0 |
| Batch size | 4 |
| Effective batch (grad accumulation) | 4 (no accumulation) |
| Training steps — 1a, 1b | 80k |
| Training steps — 1c_concat, 1c_spade, exp2 | 100k |
| Checkpoint interval | Every 5k steps |
| Sample grid interval | Every 5k steps |

At batch 4 × 100k steps, an A100 80GB run takes ~10–11 hours wall clock
(the 20k extra steps for 1c add ~2 hours vs 80k for 1a/1b).

## 4.3 Concat conditioning (Exp 1a, Exp 1c_concat)

The label is concatenated to the noisy image at the UNet input as
6 extra channels. Input is 7 channels total: 1 image + 6 label.

### 4.3.1 Input pipeline

```
x_t (1 × 512 × 512) ⊕ label (6 × 512 × 512) → concat → (7 × 512 × 512)
   ↓
DiffusionModelUNet (7 input channels)
   ↓
predicted noise (1 × 512 × 512)
```

### 4.3.2 Properties

- **Simple** — no architectural changes required beyond widening the first
  Conv layer.
- **Ambient conditioning** — the label propagates through the entire
  encoder + decoder as ordinary feature-map data. Every layer "sees" the
  label, but chooses how (and whether) to use it.
- **Global tendency** — empirically the concat variants use the label
  globally rather than per-organ. Their per-channel CLR is ≈ 0.03–0.07
  (essentially no per-organ localisation), meaning removing the uterus
  channel changes the image everywhere, not preferentially inside the
  uterus region.

The concat variant works well for global scene properties (overall body
shape, tissue intensity distributions) — see FID and hist_KL numbers in
§5.2. But its lack of per-organ localisation is the architectural
reason the downstream augmentation (Chapter 5) fails and cannot be
rescued by preprocessing tuning.

## 4.4 SPADE conditioning (Exp 1b, Exp 1c_spade)

The label is *removed* from the input entirely. UNet input is 1 channel
(the noisy image). The label enters via SPADE modules at every decoder
ResBlock.

### 4.4.1 SPADE module (Park et al. 2019)

Per-pixel spatially-adaptive normalisation:

```
Input:  feat (C × H × W), label (6 × H × W)
Step 1: normed = GroupNorm(feat)
Step 2: shared = ReLU(Conv3×3(label))     # 6 → 64
Step 3: γ = Conv3×3(shared)               # 64 → C
        β = Conv3×3(shared)               # 64 → C
Step 4: output = normed × (1 + γ) + β
```

The label is upsampled to the local feature-map resolution before entering
each SPADE module — SPADE at level 64² sees a 64² downsampled label; at
level 256² sees the 256² downsampled label.

### 4.4.2 Zero-initialisation — the failure mode of the first 1b run

Default random initialisation of γ and β heads gives chaotic per-pixel
modulation at step 0, and the diffusion loss cannot recover. Our first
1b run trained for 40k steps and produced fuzzy noise; abandoning it,
we applied the standard practice (used in DiT, Imagen, Semantic
Diffusion Models) of **zero-initialising the γ and β heads**:

```
nn.init.zeros_(spade.gamma_head.weight)
nn.init.zeros_(spade.gamma_head.bias)
nn.init.zeros_(spade.beta_head.weight)
nn.init.zeros_(spade.beta_head.bias)
```

At step 0, γ = β = 0 everywhere, so `output = normed × 1 + 0 = normed`
— SPADE starts as pure GroupNorm (identity-like). The model then
learns modulation from scratch. This fixed 1b.

The failed first run is preserved at `1b/v1_first/` for the doc drift
record; the corrected run is `1b/current/`.

### 4.4.3 Custom implementation

MONAI's `DiffusionModelUNet` ResBlock forward does not accept a label
argument, so we subclassed:

- `src/Generator/unet_spade.py` — `DiffusionUNetSPADE`, replaces
  `DiffusionModelUNet` for SPADE variants.
- `src/Generator/spade.py` — the SPADE module.

Only the decoder-side ResBlocks are SPADE. The encoder remains
label-agnostic (standard GroupNorm), which matches Park et al.'s original
formulation and preserves comparable parameter count (23.9 M vs 25.3 M
for concat).

### 4.4.4 Empirical properties

- **Per-pixel local modulation**: label change at (x, y) directly
  modulates feature at (x, y). Downstream convolutions still spread the
  effect, but the modulation itself is local.
- **CLR 0.30–0.53** across channels — 10–30× higher than concat.
- **OSI organ_corr ≈ 0.25, body_corr ≈ 0** — γ heads pick up per-organ
  structure, not just body silhouette.
- **Slight FID cost** (200.1 vs 188.2 for 1a — within noise floor) but
  better hist_KL (6.89 vs 8.15) and better LPIPS_mean (0.745 vs 0.824).

## 4.5 Conditional PatchGAN (Exp 1c only)

Added on top of both backbones — 1c_concat = 1a + PatchGAN, 1c_spade =
1b + PatchGAN — with the *same* discriminator architecture across both,
isolating the adversarial contribution from the conditioning mechanism.

### 4.5.1 Discriminator architecture

| Attribute | Value | Rationale |
|---|---|---|
| Input | `concat(image, label)` (7 channels) | Conditional variant — D learns both realism AND image-label consistency |
| Architecture | 5-block PatchGAN (Isola et al. 2017), 70×70 receptive field, 32×32 output logits | Standard pix2pix discriminator |
| Base channels | 64 → 128 → 256 → 512 → 1 | ~2.7 M parameters |
| Normalisation | **Spectral norm** on all weight layers (Miyato et al. 2018) | Prevents D from saturating to 100% accuracy and killing G gradient |
| Output | Per-patch real/fake logit map | Local discrimination → texture realism without global mode collapse |
| Activation | LeakyReLU(0.2) | Standard discriminator activation |

Source: `src/Generator/patchgan.py`.

### 4.5.2 Joint G + D training loop

Per step:

```
1. G forward: predict noise ε̂ from (x_t, label)
2. L_diff = MSE(ε̂, ε_true)                       # same as 1a/1b
3. x̂_0 = (x_t − √(1 − ᾱ_t) · ε̂) / √(ᾱ_t)       # cheap single-step estimate
4. D forward on real: D(concat(x_real, label)) → logit_real
5. D forward on fake: D(concat(x̂_0.detach(), label)) → logit_fake
6. L_D = BCE(logit_real, 1) + BCE(logit_fake, 0)  # standard non-saturating GAN loss
7. Optimizer_D.step()
8. L_adv = BCE(D(concat(x̂_0, label)), 1)         # G wants D to call fakes real
9. L_G_total = L_diff + λ_t × L_adv
10. Optimizer_G.step()
```

Key details:
- **x̂_0 estimate** — the noise-to-image inversion is done in closed form
  in a single step per iteration, not by running the full DDIM sampler.
  Cheap but noisy; adequate for D's decision because D only needs to
  judge "does this look like a real image at all," not "is this a
  high-quality generation."
- **Detached fake in D update** — standard non-saturating GAN loss on
  the detached x̂_0 prevents D gradients from flowing back through G.
- **Same forward for G's adversarial pass** — the *non-detached* x̂_0
  goes through D again to compute L_adv, so G gradients flow through
  both the DDPM MSE (via ε̂) and the adversarial signal (via x̂_0 → D).

### 4.5.3 Adversarial weight (λ) schedule

Naive addition of adversarial loss to a diffusion training loop
destabilises early training. The schedule that works in our setup:

```
Warmup    0 → 10k:      λ = 0        Pure DDPM MSE; G stabilises before any adversarial pressure
Ramp      10k → 30k:    λ = 0 → 0.01 Linear ramp
Steady    30k → 100k:   λ = 0.01     Joint training at full adversarial pressure
```

Total training steps: 100k. This gives 90k effective adversarial steps
(vs 80k pure DDPM for 1a/1b). Peak λ = 0.01 is empirically the
largest value at which G still trains stably in this setup.

Phase 2 (exp2) inherits this schedule with λ_peak = 0.01. The
exp2_lam05 diagnostic explores λ_peak = 0.05 as a tuning knob.

## 4.6 Cross-cutting improvements (all four variants)

### 4.6.1 Classifier-Free Guidance (CFG)

Training-time: with probability 10%, drop the label conditioning (set to
all-zeros) so the same network learns both conditional and unconditional
noise prediction.

Inference-time:

```
ε_cond = G(x_t, label)
ε_uncond = G(x_t, ∅)
ε̂ = ε_uncond + g · (ε_cond − ε_uncond)
```

**Per-variant guidance optima** (Tier 1 inference sweep):
- 1a, 1c_concat: g = 3.0
- 1b, 1c_spade: g = 2.0

SPADE variants prefer lower g because per-pixel modulation already
provides strong spatial signal; g > 2.0 oversaturates the SPADE effect.
Concat variants need higher g because their conditioning is ambient
and requires amplification to visibly shape the output.

CFG was added mid-Exp 1a and inherited across all four variants.

### 4.6.2 Exponential Moving Average (EMA)

EMA of the generator weights with decay 0.9999. Inference uses EMA
weights, not raw training weights. Visibly cleaner textures. Added
mid-Exp 1a; inherited across all four variants.

### 4.6.3 6-channel labels — the `body_other` addition

Details in §3.4. Added mid-Exp 1a to fix "noisy grey edges" artefact;
inherited across all four variants. Both design docs
(`../docs_archive/synthetic_mri_generator_design.md`, `../docs_archive/architecture_dataflow_v2.md`)
originally described a 5-channel label; the actual implementation is
6-channel throughout Phase 1 and Phase 2.

### 4.6.4 Weighted ovary-slice sampling

Ovary-containing slices are ~13% of the raw slice pool. The generator's
data loader boosts them 3× at sampling time → they become ~30% of
training batches. Prevents the diffusion loss from being dominated by
easy ovary-absent slices at the expense of the ovary-slice quality.

### 4.6.5 Fixed-labels resampling for periodic sample grids

Early sample-grid rendering used 4 randomly chosen labels held constant
across training. About 24% of random draws produced 4 all-background
slices → sample grids looked blank. Fix: resample up to 20 batches
looking for one with foreground content. Cosmetic but prevents the
misdiagnosis "the model is broken because the grids are empty" during
mid-training debugging.

## 4.7 Inter-Slice Consistent Stochasticity (ISCS) at inference

DDPM inference starts from x_T ~ N(0, I) per slice. If we generate slice
z independently from slice z+1, adjacent slices in a 3D volume have
independent noise seeds → visible slice-to-slice discontinuity that
breaks 3D anatomical coherence.

**ISCS fix** (`src/Generator/assemble_synthetic_volumes.py`):

```
ε_shared ~ N(0, I)                # one draw per volume
for each slice z:
    ε_z_indep ~ N(0, I)          # per-slice
    ε_z = 0.8 · ε_shared + 0.6 · ε_z_indep
    x_T[z] = ε_z
```

The 0.8 shared / 0.6 independent weighting empirically gives inter-slice
coherence without over-smoothing. Plug-and-play at inference; **no
retraining required**. Standard practice in 3D synthesis from a 2D
diffusion prior; the specific weighting is a project choice.

Volumes are then generated slice-by-slice via DDIM at 100 steps with
per-variant g, stacked into a 3D image, and saved as NIfTI.

## 4.8 Body-centered vs image-centered preprocessing (recap)

Detailed in §3.5. Generator preprocessing is body-centered (body fills
~90% of the 512² frame). RAovSeg preprocessing is image-centered (body
fills ~60%). The mismatch is the primary distribution-shift lever that
v2's Fix 3 (resample synth to source real subject's frame) addresses in
Chapter 5.

## 4.9 Phase 2 — cross-domain configuration (exp2)

Phase 2 tests whether a generator trained on D1 T2 anatomy plus a
discriminator trained on D2 T2FS style can produce D2-styled synth via
adversarial pressure alone.

### 4.9.1 Configuration

| Component | Configuration |
|---|---|
| Generator backbone | `DiffusionUNetSPADE` (SPADE conditioning, inherited from 1c_spade) |
| Generator training data | D1_MHS T2 (32 subjects, non-fat-suppressed) |
| Discriminator | Conditional PatchGAN (same architecture as 1c) with **label zeroed** — unconditional D |
| Discriminator training data | D2_TCPW T2FS (41 subjects — RAovSeg training pool + em-positive subjects that generator sees) |
| Dual dataloader | Yes — G sees D1 batches, D sees D2 batches, interleaved per step |
| λ schedule | Warmup 10k, ramp 30k, steady 0.01 (exp2) or 0.05 (exp2_lam05) |
| Steps | 100k |
| Inference target | Generate D2-styled synth conditioned on D1 anatomy labels (r1 masks) |

### 4.9.2 Why unconditional D

If D receives labels alongside the image, it can learn a
label-distribution shortcut: "D1 labels look like X, D2 labels look
like Y — reject anything that looks D1." That shortcut has nothing to
do with T2 vs T2FS style transfer.

Zeroing the label input to D forces D to judge pure pixel-level style
without any label crutch. Cost: D provides a weaker gradient than a
conditional D would, because it has less signal per image. This is a
known trade-off and (in retrospect, §5.5.3) part of why exp2 failed:
unconditional D at λ_peak = 0.01 was not enough to counter DDPM MSE.

### 4.9.3 Inference — D1 masks at ovary target t

At inference we use D1 r1 masks (single-rater subset, matches D2's
single-rater setup for consistency). Ovary intensity target t (post-
percentile-clip normalisation) is **re-calibrated per generator** — the
Phase 1 t = 0.26 was tuned to 1c_spade's synth distribution and does
not carry over.

For exp2 the intended workflow was: generate a pilot batch, measure the
ovary-voxel intensity distribution after RAovSeg's preprocess, choose t
= median of that distribution. In practice exp2 failed before the pilot
was informative (synth samples had no clear ovary structure to
calibrate against), and t = 0.26 was used as a default. This does not
affect the final DSC — the synth quality is the bottleneck, not t.

## 4.10 What is deliberately not tested (method scope)

| Not tested | Why excluded |
|---|---|
| GAN-only or VAE-only generators | Stability at n=30; keeps ablation spine clean |
| Latent diffusion (LDM) | 2D pixel-space is tractable at 512²; adds VAE stage complexity |
| Cross-attention conditioning | Not part of the concat vs SPADE ablation spine |
| Cycle consistency (CycleGAN) | Phase 2 tests one-way translation; cycle would be a separate experiment |
| Paired-cohort translation (Pix2Pix) | Requires per-subject T2/T2FS pairs, absent from UT-EndoMRI |
| 3D UNet | 2D + ISCS is cheaper and gives adequate 3D coherence for downstream |
| Latent-DDPM at higher resolution | 512² is the native resolution of the RAovSeg pipeline |
| Nyquist-scale higher-order noise schedules | β-schedule was not a variable in the ablation |

## 4.11 Design principle recap

Every design decision in this chapter follows the rule: the 2×2
architectural ablation is the locked spine, and everything else is a
quality-of-life or parity choice that must be applied identically across
all four variants. Deviating from this — e.g. adding CFG to only 1c and
not 1a — would silently corrupt the ablation. The mid-flight additions
(CFG, EMA, 6-channel, ovary-slice weighting, fixed-labels resampling,
zero-init SPADE γ/β) were all retroactively applied to earlier variants
before final reporting.

The full 4-variant training corpus (~40 GPU hours on A100) was executed
under this parity constraint. The Phase 1 results in Chapter 5 are
directly comparable across variants because of it.
