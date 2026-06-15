# Exp 1b — 2D SPADE-Conditioned DDPM: Planning Notes

A forward-looking checklist of what to design, decide, and verify *before*
implementing 1b. Drafted from what we learned in 1a so we don't relearn
the same things. The companion to `EXP1A_NOTES.md`.

---

## 1. What 1b is

A 2D conditional DDPM that **replaces** concat conditioning with
SPADE (Spatially-Adaptive Normalization, Park et al. 2019). SPADE
injects label information into the decoder *at every level* via
spatially-varying per-channel γ and β modulation, rather than just
at the input via channel concatenation.

The ablation contract:

> **1a vs 1b answers**: does spatially-adaptive normalisation produce
> better label-image alignment than channel concatenation, holding
> everything else equal?

So everything else **must** stay equal between 1a and 1b — same data,
same backbone capacity, same hyperparameters, same CFG/EMA setup,
same evaluation protocol.

---

## 2. Inherited from 1a (do not change in 1b)

### Data + split

- **6-channel one-hot label** (outside_body, uterus, L-ov, R-ov, em, body_other)
- **Same train/test split** (RAovSeg test sacred; train = RAovSeg train ∪ em-positive minus missing-file candidates → ~32 effective train subjects)
- **Same `preprocess_for_generator.py` outputs** — no re-preprocessing needed
- **Body silhouette is part of the input** — do *not* drop the 6th channel; it's no longer the variable under test
- **Ovary L/R distinction is mostly midline-split** — same limitation; will affect 1b the same way as 1a so the comparison stays fair

### Training schedule and optimisation

- **Batch size 4** — established memory budget; verify SPADE fits before increasing
- **80,000 steps** — same total compute
- **AdamW, lr 1e-4, weight_decay 0.0, grad_clip 1.0** — DO NOT TUNE for 1b
- **bf16 autocast**
- **CFG dropout 0.1, guidance scale 3.0**
- **EMA decay 0.9999**
- **DDIM 50 inference steps**
- **Fixed_labels resampling** — keeps in-training grids interpretable

### MONAI Generative quirks

- `DDPMScheduler(schedule=..., beta_start=..., beta_end=...)` — NOT `beta_schedule=`
- Schedule name is `"linear_beta"`, NOT `"linear"`
- `torch.cuda.amp.autocast` deprecation warning from `vector_quantizer.py` is harmless library noise; ignore

### SLURM / env

- Same conda env (`synth_mri`); same modules; same `MKL_THREADING_LAYER=GNU`; same `set +u; source activate; set -u` pattern
- Same partition + qos (`gpu` + `qos=gpu`) + 24h wall
- Same `git rev-parse HEAD || echo "(not a git repo)"` provenance failure — harmless
- Use `gpu-h100-nvl` (96 GB) only if SPADE OOMs

---

## 3. Open design decisions to make before implementing

### 3.1 Pure SPADE or hybrid (concat + SPADE)?

| Option | Ablation cleanliness | Risk |
|---|---|---|
| **Pure SPADE** — input is 1 channel (noisy image only); labels enter only through SPADE | Cleanest "concat vs SPADE" contrast | Loses some early-layer label awareness; may need stronger SPADE |
| Concat + SPADE | Best raw quality | Muddies the ablation: "concat vs concat+SPADE" is a weaker scientific claim |

**Recommendation: pure SPADE** for the dissertation experiment. The
ablation contract is clean and matches the SPADE paper's framing.

This means **`model.in_channels` drops back to 1** for 1b, but
`num_label_channels` stays at 6 (passed separately into the SPADE
modules, not concatenated at input).

### 3.2 SPADE at which U-Net levels?

Standard practice: SPADE in every decoder block, often also in the
bottleneck. Adding it to the encoder gives marginal extra gain at
extra cost.

| Variant | Where SPADE fires |
|---|---|
| **Decoder-only** | All 4 up-blocks + bottleneck |
| Decoder + bottleneck |  Same as above (default) |
| Encoder + decoder | All down-blocks + bottleneck + up-blocks |

**Recommendation: decoder + bottleneck.** Matches the SPADE GAN paper.
Encoder-side SPADE is rarely worth the extra compute.

### 3.3 SPADE module internals

The standard SPADE block is:

```
y = GroupNorm(x)                    # parameter-less normalisation
gamma, beta = SmallConvNet(label_downsampled)
out = y * (1 + gamma) + beta
```

Open choices:

- **Label downsampling**: bilinear vs nearest. Use **nearest** — labels are categorical, bilinear blurs them.
- **SmallConvNet structure**: usually 2 conv layers (3×3 → ReLU → 3×3) producing γ and β as separate heads or one head split. The SPADE paper uses 128 hidden channels — that's overkill for our 6 input channels. **Try 64 hidden channels first.**
- **Normalisation base**: SPADE normally replaces BatchNorm; in our UNet we use **GroupNorm (32 groups)**. SPADE's spatial modulation goes on top of GroupNorm. Verify no double-norm.

### 3.4 Backbone parity with 1a

To make the comparison fair:

- **Same U-Net depths** (4 levels)
- **Same channel widths** ([64, 128, 256, 256])
- **Same num_res_blocks (2)**
- **Same attention configuration** — attention at level 3 (64²) only;
  level 2 was removed in 1a for memory, and SPADE doesn't change
  that calculus

The parameter count will *increase* (~10-30%) because of the SPADE
modulation networks. That's the point — SPADE is supposed to be a
"better way to use the parameters." Don't try to match 1a's exact
parameter count by shrinking the backbone.

### 3.5 How does SPADE interact with CFG?

CFG drops the label tensor to all-zeros with probability 0.1.

In a concat setup that means "no label info at input." In a SPADE
setup it means "no spatial modulation at any level — γ=0, β=0
becomes y = GroupNorm(x) · 1 + 0 = y." The model learns the
unconditional distribution naturally because the SPADE-multiplier
collapses to identity.

This is the standard behaviour. **Keep CFG as-is.** Drop the entire
label tensor to zeros, not selectively per channel. Standard CFG
inference (guidance scale 3.0) applies unchanged.

---

## 4. Memory and performance forecast

SPADE adds at every level:

- A label-downsampler: cheap (1 conv or interpolation)
- A `SmallConvNet`: at level `L` with feature resolution `H×W`, output is `(B, C_L, H, W)` γ + same for β.

At each level, memory cost vs current:

| Level | Resolution | Feature C | SPADE γ+β extra activations (batch 4, fp32) |
|---|---|---|---|
| 0 (up) | 256×256 | 256 | ~67 MB |
| 1 (up) | 128×128 | 128 | ~8 MB |
| 2 (up) | 256×256 | 128 | ~33 MB |
| 3 (up) | 512×512 | 64 | ~67 MB |
| bottleneck | 64×64 | 256 | ~4 MB |

Total extra: ~180 MB of activations cached for backward. **Trivial vs
the current ~50 GB at peak.** SPADE will NOT push us over the OOM
ceiling at batch 4.

Compute cost per step: probably +15-20% wall clock for the extra
SPADE convs. So 1b run might take **~11-12h** instead of 1a's 9-10h.

---

## 5. Implementation plan (in order)

1. **Write `spade.py` (~80 lines)** with:
   - `SPADENorm(nn.Module)` — takes `(features, label)`, applies GroupNorm + γ/β modulation
   - `LabelEmbedder` — downsamples one-hot label to per-level resolution

2. **Subclass `DiffusionModelUNet`** in a new file `model_spade.py`:
   - Override decoder blocks to receive `label` alongside `temb`
   - Insert `SPADENorm` in place of the standard GroupNorm at each decoder layer
   - DO NOT touch encoder unless we decide on encoder SPADE
   - Keep the same `forward(x, timesteps)` signature externally — but add an extra `label` argument

   Alternative if subclassing is fragile: **fork** MONAI Generative's
   `DiffusionModelUNet` into a local file and edit directly. This is
   the SPADE GAN paper's pattern.

3. **Update `model.py`**:
   - `build_unet_spade(model_cfg)` factory returning the SPADE UNet
   - New `SPADEConditionedDDPM(nn.Module)` wrapper analogous to
     `ConcatConditionedDDPM` but `predict_noise(x_t, label, t)` does NOT concat
   - `sample()` signature unchanged externally — CFG works the same way

4. **Branch the config**:
   - New file `exp1b.yaml` mirroring `exp1a.yaml` with:
     - `model.in_channels: 1` (was 7)
     - `model.spade_hidden: 64`  (new)
     - `model.spade_levels: [false, true, true, true, true]`  (bottleneck + 4 up-blocks)
   - Everything else identical to 1a

5. **Adjust `train.py`**:
   - Conditional import: `from .model import SPADEConditionedDDPM`
     when running 1b config
   - The CFG label dropout (`lbl * keep.view(-1, 1, 1, 1)`) stays
     identical — labels still pass through unchanged structure, just
     to SPADE instead of cat
   - The fixed_labels resampling already excludes channel 5; no change

6. **Branch SLURM script**:
   - `scripts/train_exp1b.sh` mirroring `train_exp1a.sh` with `--config src/Generator/exp1b.yaml` and `--job-name=exp1b_spade`

7. **Smoke test** — `smoke_test.py` updated to take a config path and run 20 steps with the SPADE backbone. Verify:
   - Model builds
   - Forward pass works
   - Memory at batch 4 fits
   - Loss decreases over 20 steps

8. **Full run** — submit overnight. Watch for:
   - Sample grids at step 5000 to confirm SPADE is producing
     anatomy (not just noise)
   - Loss curve compared to 1a's at the same step — should be
     comparable (~0.04-0.10 at step 5000)

---

## 6. Things to measure in 1b vs 1a

### Qualitative (eyeball)

- [ ] Body silhouette still clean and bounded?
- [ ] Outside-body region still uniformly dark?
- [ ] Within-body organ-shaped content matches *organ identity*, not just "tissue here"?
- [ ] Edge sharpness comparable to 1a?
- [ ] Sample-to-sample variation reasonable?

### Quantitative (start computing)

- [ ] **FID** between synthetic and real D2 T2FS slices (on the train pool;
  test pool kept sacred until end-to-end evaluation)
- [ ] **Boundary DSC** — segment the synthetic image with a pre-trained
  segmentation model, compare against input label. Tests whether
  SPADE actually improved organ-level spatial fidelity.
- [ ] **Nearest-neighbour LPIPS** — memorisation check; flag any synthetic
  with NN distance to a training sample < 0.1
- [ ] **Intensity histogram** vs real, with attention to the [0.22, 0.3]
  ovary window for downstream RAovSeg compatibility

### Speed of convergence

- [ ] At step 30k, is 1b's sample grid clearly better than 1a's at 30k?
  (i.e. does SPADE accelerate learning?)

### Comparison artefact

- Generate one `validate_step_080000_6chan.png`-equivalent for 1b
  with the SAME deterministic labels (the same 4 high-foreground
  slices `inference_validate.py` picks). Direct A/B comparison
  becomes a one-PNG-vs-one-PNG read.

---

## 7. Risks and open questions

- **R1: SPADE may overfit faster.** Extra modulation power on 32
  training subjects could mean step 80k is too long. Plan: watch
  the sample grids — if they plateau or degrade after step 50k,
  reduce `total_steps` for 1b retroactively.

- **R2: SPADE doesn't ship with MONAI Generative.** Either subclass
  carefully or fork the UNet file. Subclassing is cleaner; forking
  is more robust to MONAI's internal refactors.

- **R3: Encoder-side parity with 1a.** If we keep concat conditioning
  at the input (hybrid) by accident, the comparison is invalid.
  Verify with explicit assert: `model.in_channels == 1` in 1b.

- **R4: SPADE's `label_downsampled` for the bottleneck** — at 64²,
  the 6-channel one-hot becomes mostly zero (organs are tiny
  fractions of the image at low resolution). The bottleneck SPADE
  may effectively get blank input. Consider whether to skip
  bottleneck SPADE if testing shows no signal there.

- **R5: Fair CFG comparison.** Make sure CFG dropout in 1b zeros the
  6-channel tensor identically to 1a. If you accidentally introduce
  a "null label" learned vector, 1b becomes CFG+learned-null vs 1a's
  CFG+zero-null. Same dropout style as 1a.

---

## 8. After 1b, before 1c

Once 1b is done:

- Generate the 4-image triplet: 1a-final-validate / 1b-final-validate / 1c-final-validate (after 1c)
- Compute FID / boundary DSC for both 1a and 1b
- Write the methodology section while it's all fresh

Then 1c adds the PatchGAN. Memory-wise it adds discriminator gradients;
likely fits but be ready to drop batch to 2 if it doesn't.

---

## 9. Open data questions (worth answering before 1b for the writeup)

- **How well does the body_other channel match real anatomy?** Currently
  computed from a fixed threshold (0.05) on the normalised image. Has
  anyone visually verified it on all 32 subjects? Worth doing — an
  `extract_silhouette.py` viewer script that overlays the body mask
  on the image for a quick sanity sweep.

- **What's the actual distribution of em coverage in the train pool?**
  Roughly how many em-positive subjects, how many em-positive slices,
  total em voxels. For the dissertation we should be able to say
  "endometrioma channel was trained on N voxels from K subjects."

- **What does the loss curve look like over 80k steps for 1a-final?**
  Did it plateau early? If yes, 1b might not need 80k steps either —
  save 2-3 hours by stopping at the actual plateau.
