# Training Overview — How We Generate Synthetic Pelvic MRI

A plain-English explanation of what the generator is, what data it sees,
how the train/test split works, and what each conditioning input does.
The companion docs `src/Generator/EXP1A_NOTES.md` and `EXP1B_NOTES.md`
hold the detailed bug log and design decisions; this file is the
"sit-down-and-read-once" version.

---

## 1. What the model is

A conditional Denoising Diffusion Probabilistic Model (DDPM). You give
it a label map (a multi-channel anatomical mask), and it generates a
synthetic T2-weighted fat-suppressed pelvic MRI slice that matches that
anatomy. It does not memorise images; it learns the *distribution* of
pelvic MRI conditioned on anatomy.

We now have **four trained variants** in a 2×2 ablation (status as of
June 2026):

- **Exp 1a — Concat conditioning** *(80k steps, done)*: the label is
  glued to the noisy image at the network input as extra channels
- **Exp 1b — SPADE conditioning** *(80k steps, done)*: the label enters
  at every decoder layer through Spatially-Adaptive Normalization modules
- **Exp 1c-concat — Exp 1a + PatchGAN** *(100k steps, done)*: same
  concat backbone as 1a plus a conditional PatchGAN discriminator that
  judges both image realism and image-label consistency
- **Exp 1c-spade — Exp 1b + PatchGAN** *(100k steps, done)*: same
  SPADE backbone as 1b plus the same PatchGAN discriminator

Same data, same preprocessing, same backbone hyperparameters within each
arm. The 2×2 ablation isolates *how* the label gets used (concat vs SPADE)
AND *whether adversarial pressure helps* (with/without PatchGAN).

**Key findings** (full details in [RESULTS_2x2.md](RESULTS_2x2.md)):
- Concat variants produce more globally-realistic textures (best FID, hist_KL)
- SPADE variants achieve 10-30× higher per-channel label localisation
- PatchGAN gives the biggest realism wins on concat, the biggest perceptual
  realism wins on SPADE
- No single winner across all metrics — choice depends on downstream use case

Per-variant inference defaults: 1a/1c-concat use g=3.0, 1b/1c-spade use g=2.0,
DDIM steps=100 shared. See `src/Generator/TIER1_TUNING_AND_EXPLAINABILITY.md`
for the rationale.

---

## 2. What the model is trained on

**Both real images AND masks together** — not masks alone. The model
learns what pelvic MRI *looks like* from the real T2FS images, and
learns *where to put anatomy* from the masks. The two are inseparable
in training.

Per training step:

```
1. Take a real MRI slice and its matching label
2. Add Gaussian noise to the image (random timestep t ∈ [0, 1000])
3. Ask the model: "given this noisy image + this label, predict the noise"
4. Loss = MSE(predicted_noise, actual_noise)
```

At inference time the model starts from pure noise and gradually
denoises it over 50 DDIM steps, with the label steering every step.

---

## 3. The data — what's used, what's excluded

UT-EndoMRI Dataset 2 (D2_TCPW), single institution. Of the **~73
subjects** on disk:

| Filter | Subjects removed | Why |
|---|---|---|
| Missing T2FS image | 3 | Can't train without the image |
| Missing uterus or ovary mask | ~9 | Required by `build_generator_split.py` filter |
| RAovSeg test subjects (sacred) | 8 | Held out for downstream evaluation, never trained on |
| **Effective training pool** | **32 subjects** | All pass T2FS + uterus + ovary |

Test set (sacred — never seen during training, not even in 1a or 1b):

```
D2-005, D2-015, D2-016, D2-017, D2-023, D2-024, D2-026, D2-038
```

**Slice usage:** all ~1,143 axial slices from the 32 training subjects
contribute, but a **weighted sampler** boosts ovary-containing slices
**3×** during sampling. So ovary slices go from a natural 13% of the
slice pool to ~30% of training batches. This compensates for the
natural class imbalance (most pelvic slices don't show ovaries).

---

## 4. The conditioning channels (6-channel labels)

Each label is a 6-channel one-hot mask at 512×512. Exactly one channel
is `1` per pixel:

| Channel | Name | What it captures | Source |
|---|---|---|---|
| 0 | `outside_body` | Air outside the body silhouette | Computed from image (1 − body) |
| 1 | `uterus` | Target organ | Manual annotation (`_ut.nii.gz`) |
| 2 | `ov_L` | Left ovary (image-space split) | Auto-split from combined `_ov.nii.gz` |
| 3 | `ov_R` | Right ovary (image-space split) | Auto-split from combined `_ov.nii.gz` |
| 4 | `em` | Endometrioma | Manual annotation when present (`_em.nii.gz`) |
| 5 | `body_other` | Body tissue that isn't a target organ (fat, muscle, bowel, bladder wall…) | Computed from image (threshold + morphological closing + fill, minus target organs) |

The `body_other` channel was added between the early 1a runs and the
final 6-channel runs. Without it, the model didn't have a clear signal
for "this region is air vs body tissue at the image edges" and produced
noisy grey corners. Adding it gives explicit "fill this region with
plausible non-target tissue" conditioning.

---

## 5. Train / test split details

Split is built from `data/processed/manifest.csv` by
`src/Generator/build_generator_split.py`:

| Set | Count | Members |
|---|---|---|
| Train | 32 | RAovSeg's train_val ∪ em-positive subjects, minus subjects with missing files, minus test |
| Test (sacred) | 8 | Identical to RAovSeg's test split |

Split file: `data/splits/d2_generator_split.json`. The `_meta` block
records the exact subject IDs and the dropped-for-missing-files list.

Within the 32 training subjects, **no held-out validation set is used
during training** — we train on all 32, watch the loss curve and the
periodic sample grids for convergence, and treat the 8 RAovSeg test
subjects as the *only* held-out evaluation set.

---

## 6. Improvements added during 1a (in chronological order)

1. **Memory budget**: removed self-attention at level 2 (128² resolution),
   kept it only at the deepest level (64²). Without this 1a OOMs at
   batch 4 on an A100 80GB.
2. **Classifier-Free Guidance (CFG)** — 10% of training batches use a
   zero label; at inference we combine conditional + unconditional
   predictions with guidance scale 3.0. Substantially improved spatial
   alignment of synthesised anatomy to the input label.
3. **EMA of model weights** — decay 0.9999. Visibly cleaner sample
   textures.
4. **Body silhouette channel (6th label channel)** — fixed the "noisy
   grey edges" artefact by giving explicit air-vs-tissue conditioning.
5. **Fixed-labels resampling** — the 4 in-training visualisation labels
   used to be a single random draw at startup. ~24% of the time we got
   4 background-only slices and the periodic grids looked blank. Now we
   resample up to 20 batches looking for one with foreground content.

These changes were inherited unchanged into 1b for ablation parity.

---

## 7. Sample images

The in-training sample grids show 4 fixed labels held constant for all
80 k steps so progress is visually comparable.

### 1a evolution — early CFG-only run (before EMA and 6-channel)

Step 5,000 — model has just learned coarse pelvic texture:

![1a CFG step 5k](exp1a_samples_cfg/step_005000.png)

Step 80,000 — recognisable anatomy emerging, but the model only uses the
labels globally ("yes, generate pelvis") not locally ("uterus here
specifically"):

![1a CFG step 80k](exp1a_samples_cfg/step_080000_final.png)

### 1a final — CFG + EMA + 6-channel body silhouette

Step 80,000:

![1a CFG+EMA+6chan step 80k](exp1a_samples_cfg_ema_6chan/step_080000_final.png)

Notice the body silhouette is now clean and well-bounded; outside-body
regions are uniformly dark; within-body content has organ-shaped bright
and dark areas where the label says they should be.

### 1a final validation grid (deliberately picked high-anatomy labels)

This is the post-hoc validation run we did against the step-80 k
checkpoint, with `inference_validate.py` picking 4 labels by highest
target-organ voxel count. The argmax column on the right shows the full
6-channel structure clearly:

![1a CFG+EMA+6chan validation](exp1a_samples_cfg_ema_6chan/validate_step_080000_6chan.png)

The overlay column (yellow uterus, red L-ovary, blue R-ovary, green em)
demonstrates that the synthetic content actually tracks the labelled
organ positions — the CFG amplification is working.

### 1b first attempt — SPADE without identity-init (broken)

This is the first SPADE run we did, before applying the
zero-initialisation fixes. By step 40,000 the model has barely learned
anything — body silhouettes are fuzzy and within-body content is mostly
uniform noise:

![1b first attempt step 40k](exp1b_samples_first/step_040000.png)

This run was abandoned. Diagnosis: SPADE's γ and β heads use default
random initialisation, so the spatial modulation is chaotic from step 0.
Standard practice (used in DiT, Imagen, SDM) is to zero-init both heads
so SPADE starts as identity-like and the model gradually learns to use
it. After the patch the second 1b run is in progress.

---

## 8. How to read a sample grid

Each row in a sample grid is one (label, generated image) pair. With
the recent change, each row has 4 columns:

| Col | Title | What it is |
|---|---|---|
| 0 | `real (source of the label)` | The actual MRI slice that the label was extracted from. **Ground-truth reference.** |
| 1 | `synthetic` | What the diffusion model generated when given the label at col 3 as conditioning |
| 2 | `overlay (Y=ut, R=L-ov, B=R-ov, G=em)` | Col 1 with the organ masks painted on top in colours. Lets you check spatial alignment |
| 3 | `input label (argmax)` | The 6-channel one-hot label collapsed to a single colour per pixel — shows the conditioning |

Older grids (from before this change) only have columns 1, 2 and 3.

---

## 9. What we deliberately don't use

| Available | Why excluded |
|---|---|
| Cyst masks (`_cy.nii.gz`) | Cyst-positive subjects are excluded by the RAovSeg inclusion filter; cysts aren't a conditioning channel |
| D1 dataset (51 more subjects from Memorial Hermann) | Different scanner site, different protocols — adding it would help with data scarcity but introduces a site-shift confound |
| Subjects without uterus annotation | Uterus is a conditioning channel; a missing `_ut` would teach the model "no uterus here," which corrupts the training signal |
| The 8 RAovSeg test subjects | Sacred — held out for the downstream RAovSeg+synthetic evaluation |

---

## 10. Quick glossary

- **DDPM** — Denoising Diffusion Probabilistic Model. Adds noise to
  images during training, learns to undo the noise step by step,
  generates new images by starting from pure noise and denoising
- **DDIM** — Faster sampler at inference time (50 steps vs DDPM's 1000)
- **CFG** — Classifier-Free Guidance. Training trick that lets you
  amplify the label conditioning at inference time
- **EMA** — Exponential Moving Average of the model weights, used for
  inference. Smooths out optimisation noise
- **SPADE** — Spatially-Adaptive Normalization. A way of injecting
  label conditioning at every layer of the decoder instead of just
  the input
- **`one-hot label`** — Each pixel belongs to exactly one of the 6
  classes; the label is a stack of 6 binary masks
- **`body_other`** — The 6th channel; "inside body, not a target organ."
  Lets the model render fat / muscle / bowel without being told
  specifically where each goes
