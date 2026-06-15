# Exp 1a — 2D Concat-Conditioned DDPM: Implementation Notes

A record of what was built for Exp 1a, the issues encountered along the way,
and the rationale behind each design decision. This is the technical
companion to the planned writeup; if you can't remember *why* a value or
file looks the way it does, it should be in here.

---

## 1. What Exp 1a is

A 2D conditional Denoising Diffusion Probabilistic Model (DDPM) that
generates synthetic T2-weighted fat-suppressed pelvic MRI slices,
conditioned on a multi-channel anatomical label map via **channel
concatenation** at the U-Net input. It is the baseline ("floor") of the
Phase 1 ablation. Exp 1b will replace the conditioning mechanism with
SPADE; Exp 1c will add a PatchGAN adversarial loss on top.

Final architecture:

| Component | Value |
|---|---|
| U-Net | MONAI Generative `DiffusionModelUNet` |
| Levels (resolutions) | 4 — 512², 256², 128², 64² |
| Channel widths | 64, 128, 256, 256 |
| Self-attention levels | Deepest only (64²) — level 2 (128²) removed for memory |
| Norm | GroupNorm (32 groups) |
| Total params | ~25.3 M |
| Conditioning | Concat: 1 image channel + 6 label channels → 7 input channels |
| Diffusion timesteps | 1000, linear-beta schedule |
| Prediction target | ε (noise) |
| Loss | MSE on predicted vs added noise |
| Sampling | DDIM, 50 inference steps |

Training:

| | |
|---|---|
| Optimiser | AdamW, lr 1e-4 |
| Batch size | 4 |
| Total steps | 80,000 |
| Mixed precision | bf16 autocast |
| Hardware | 1× A100 80 GB on Sheffield Stanage |
| Wall clock | ~9-10 h |

---

## 2. Final 6-channel label layout

One-hot per voxel (exactly one channel == 1):

| Idx | Name | Meaning |
|---|---|---|
| 0 | `outside_body` | Air / outside the body silhouette |
| 1 | `uterus` | Target organ |
| 2 | `ov_L` | Left ovary (image-space split of the combined `ov` mask) |
| 3 | `ov_R` | Right ovary |
| 4 | `em` | Endometrioma (empty for most subjects; em-positive ones included) |
| 5 | `body_other` | Inside body silhouette, no target organ — fat, muscle, bowel, bladder wall, etc. |

Priority for overlap resolution among target organs:
`em > L-ov > R-ov > uterus` (endometriomas physically sit inside ovary
contours, so they win those voxels).

The body silhouette (channel 5 + channel 0 boundary) is computed
automatically from the normalised T2FS image (threshold > 0.05,
per-slice binary closing × 3 iters, fill internal holes). UT-EndoMRI
ships no body silhouette annotation; this is a cheap deterministic
proxy that works because pelvic T2FS has reliably dark air after
percentile-clip + minmax normalisation.

---

## 3. Issues encountered and how they were resolved

### 3.1 Environment / HPC setup

| Issue | Resolution |
|---|---|
| Login node defaults to Python 2 | `module load Anaconda3/2024.02-1` + `source activate synth_mri` |
| `source activate` updates the prompt but NOT `$PATH` → `python` still points at base Anaconda's binary | Workaround: define `SYNTH_PY=/mnt/parscratch/users/ijp25lg/anaconda/.envs/synth_mri/bin/python` and call it explicitly. Affects interactive shells; SLURM jobs are unaffected. |
| `pip install pandas` triggers source build (requires GCC ≥ 9.3; Stanage login has 4.8.5) | Use `conda install -y -n synth_mri -c conda-forge pandas` for binary install |
| Conda's MKL activation script reads unset env vars (`MKL_INTERFACE_LAYER`) and trips `set -u` in SLURM scripts | Wrap activation with `set +u; source activate synth_mri; set -u` |
| MKL/OpenMP collision on login node | `export MKL_THREADING_LAYER=GNU` |
| Missing Python deps for training: `yaml`, `tensorboard`, `matplotlib`, `tqdm`, `monai-generative` | `conda install -y -n synth_mri -c conda-forge tensorboard matplotlib tqdm pyyaml` plus `$SYNTH_PY -m pip install monai-generative` |
| SLURM partition `sheffield` was a guess | Verified with `sinfo -o "%P"` — confirmed as the default CPU partition |

### 3.2 MONAI Generative API compatibility

| Issue | Resolution |
|---|---|
| `DDPMScheduler(beta_schedule=...)` — keyword forwarded to `_linear_beta(beta_schedule=...)` which rejects it | Renamed the kwarg to `schedule=` (YAML key `beta_schedule` retained for readability and mapped at call site) |
| `schedule="linear"` not in MONAI Generative's `NoiseSchedules` component registry | Registry key is `"linear_beta"` (matches the underlying `_linear_beta` function name without the underscore prefix) |

### 3.3 GPU memory (CUDA OOM)

The U-Net's self-attention over the spatial sequence is the dominant
memory cost. At a 128² feature map the score tensor has shape
`B × n_heads × N × N` where N = 16,384. In fp32 that's:

- B=8: ~64 GiB allocation → OOM (A100 has 80 GiB)
- B=4: ~32 GiB allocation → still OOM because ~53 GiB of activations
  from earlier down-blocks were already cached for backward
- B=4, no level-2 attention: 64² attention only → ~2 GiB → fits comfortably

**Decision**: keep batch 4, remove self-attention at level 2 (128²),
keep self-attention only at the deepest level (64²,
`attention_levels: [false, false, false, true]`,
`num_head_channels: [0, 0, 0, 32]`). This matches mainstream
pixel-space diffusion practice (Imagen, SD-pixel) and reduces the
parameter count from a hypothetical 80 M to 25.3 M without affecting
the validity of the 1a vs 1b ablation, since 1b/1c will use the
same backbone.

Other options considered and rejected for now:

- **Gradient checkpointing** — preserves all attention levels, costs
  ~25-35% wall clock. Worth doing if the level-2 attention turns out
  to matter for image quality.
- **H100 NVL partition** (96 GB GPU memory) — would fit batch 4 with
  level-2 attention. Saved for later if we hit memory walls again on
  larger architectures or 3D extensions.
- **Multi-GPU (DDP)** — standard pattern works on Stanage but
  requires ~50-80 lines of refactoring. Not justified for 25 M
  params at batch 4.

### 3.4 Train/test split

The reference RAovSeg paper splits 30 train+val / 8 test from 38
subjects passing inclusion criteria (T2FS + ovary label, no cyst, no
endometrioma). The authors did not publish the canonical subject IDs.

`build_generator_split.py` constructs the generator's split from the
RAovSeg manifest:

- **Test set**: identical to RAovSeg's 8-subject test split — sacred,
  never trained on, kept intact across all experiments
- **Training set**: RAovSeg's 30 train_val subjects ∪ all em-positive
  subjects from disk, minus any that overlap with the test set, minus
  any missing T2FS / uterus / ovary labels

The em-positive subjects are added so channel 4 of the label tensor
has real signal (most RAovSeg-aligned subjects are em-negative). 9
candidates failed the file-completeness filter, leaving an effective
~32-subject training set after silent skips were eliminated.

### 3.5 In-training visualisation grid: blank-label problem

The training script froze 4 labels at startup and reused them for every
visualisation grid. With the 3× weighted sampler boosting ovary slices
to ~30% of the marginal, a single batch of 4 had a ~24% chance of
landing on entirely background-only slices. This happened twice in a
row across two separate training runs, making the in-training grids
uninterpretable.

**Resolution**: at startup, resample up to 20 batches looking for one
where at least `max(2, n_grid // 2)` samples carry target-organ
voxels. If no batch qualifies, fall back to the best one seen and
print a warning. Scoring uses channels 1-4 only (target organs);
channel 5 (body) is non-zero almost everywhere and would otherwise
defeat the score.

### 3.6 Classifier-Free Guidance (CFG)

Initial results showed the model used the label *globally* ("generate
pelvic content") but not *locally* ("put the uterus here"). CFG is the
inference-time amplifier for label conditioning:

- During training: each sample has 10% probability of being
  replaced with the null label (all zeros), so the model learns both
  p(x | label) and p(x | ∅).
- At inference: combine ε_cond + ε_uncond with
  `ε_guided = ε_uncond + w · (ε_cond − ε_uncond)`. We use w = 3.0;
  Stable Diffusion uses 7.5 but small medical datasets typically
  benefit from a lower scale.

**Effect verified**: synthetic images visibly track the label
positions of uterus, L/R ovary and endometrioma. The "spatial
awareness" symptom the original run lacked is substantially fixed.

### 3.7 Exponential Moving Average (EMA) of weights

Standard practice in DDPM literature (DDPM, ADM, EDM, Imagen, SD).
After each `optim.step()`, an EMA copy of the model is updated:

`ema = decay · ema + (1 − decay) · train`

Decay = 0.9999 averages the last ~10 k steps. The EMA copy is used for
all inference (in-training grids and post-hoc `inference_validate.py`),
the training weights for the optimisation step. Saved in every
checkpoint under the `"ema"` key; `inference_validate.py` prefers EMA
weights with `--no-ema` to override for comparison.

**Effect**: visibly cleaner samples — high-frequency optimisation
noise smooths out, producing sharper organ contours and less grain.

### 3.8 6th channel: body silhouette

Final design improvement, motivated by a clear artefact in the
CFG+EMA outputs: the central region (covered by foreground labels)
showed detailed anatomy, but the edges/corners (where labels are all
zero) reverted to noisy grey patches. Cause: with only the 5 organ
labels, the model had no explicit signal about *where the body
silhouette is* — beyond the labelled organs it was sampling from the
marginal distribution of pelvic edge content.

The 6th channel `body_other` (inside body but not a target organ)
gives the model an explicit *anatomical envelope* signal everywhere,
not just at organ regions. Edges become explicitly modelled as
"outside body, generate uniform background" instead of being left
as implicit guesswork.

**Trade-off accepted**: this changes the input shape for all of
Phase 1. Exp 1b (SPADE) and Exp 1c (SPADE + PatchGAN) will also use
the 6-channel input. The contract for the ablation becomes "concat
vs SPADE vs SPADE+PatchGAN, all on 6-channel input." The original
5-channel 1a results are kept on the HPC under
`runs/exp1a_cfg_ema` as a deprecated baseline reference.

---

## 4. Final hyperparameter summary

```yaml
data:
  num_label_channels: 6        # outside_body, uterus, L-ov, R-ov, em, body_other
  ovary_oversample_weight: 3.0 # weighted sampler 3× boost on ovary-containing slices
  image_size: 512

model:
  in_channels: 7               # 1 image + 6 labels
  out_channels: 1              # noise prediction on image channel
  num_channels: [64, 128, 256, 256]
  attention_levels: [false, false, false, true]   # 64² only
  num_res_blocks: 2
  num_head_channels: [0, 0, 0, 32]
  norm_num_groups: 32

diffusion:
  num_train_timesteps: 1000
  beta_schedule: "linear_beta"  # MONAI Generative registry key
  beta_start: 0.0001
  beta_end: 0.02
  prediction_type: "epsilon"

training:
  batch_size: 4                 # OOM at 8 with current attention; 4 fits comfortably
  total_steps: 80000
  lr: 1.0e-4
  weight_decay: 0.0
  grad_clip: 1.0
  amp: true                     # bf16 autocast
  cfg_dropout_prob: 0.1         # CFG label-dropout probability
  ema_decay: 0.9999             # EMA decay for inference weights

sampling:
  num_inference_steps: 50       # DDIM
  num_samples_per_grid: 4
  guidance_scale: 3.0           # CFG guidance scale (1.0 = no CFG)
```

---

## 5. Files in `src/Generator/`

| File | Role |
|---|---|
| `preprocess_for_generator.py` | Per-subject preprocessing: resample, normalise, compute body silhouette, build 6-channel one-hot label, save NIfTI |
| `build_generator_split.py` | Build the train/test JSON from the RAovSeg manifest, filtering candidates with missing files |
| `dataset.py` | `D2SliceDataset`: caches volumes in RAM, indexes (subject, z) slices, weighted sampler factory |
| `model.py` | `ConcatConditionedDDPM` wrapper, `EMAModel` class, scheduler builders |
| `train.py` | Main training loop with CFG dropout, EMA updates, periodic sampling, checkpointing, resume-from-latest, auto-clean on fresh runs |
| `smoke_test.py` | 20-step pipeline check before submitting a real job |
| `inference_validate.py` | Post-hoc sampling with deliberately chosen high-anatomy labels; uses EMA weights by default |
| `exp1a.yaml` | All hyperparameters |
| `EXP1A_NOTES.md` | This file |

`scripts/preprocess_generator.sh` and `scripts/train_exp1a.sh` are the
SLURM submit files.

---

## 6. Inputs / outputs

**Inputs (assumed on disk before training):**

- `UT-EndoMRI/D2_TCPW/D2-XXX/D2-XXX_{T2FS,ut,ov,em,cy}.nii.gz`
- `data/processed/manifest.csv` (from RAovSeg preprocess) — used to identify em-positive candidates and which subjects passed RAovSeg's inclusion criteria

**Outputs (per training run):**

- `runs/exp1a/ckpt/step_XXXXXX.pt` — model + EMA + optim state, every 5,000 steps
- `runs/exp1a/samples/step_XXXXXX.png` — visualisation grid (synth, overlay, argmax label), every 5,000 steps
- `runs/exp1a/tb/` — TensorBoard event files (loss, it/s)
- `runs/exp1a/config_used.yaml` — snapshot of the YAML used for this run

---

## 7. Known limitations of 1a (not 1a's job to fix)

- **Spatial coherence still imperfect** — even with CFG=3 and body silhouette, sample-to-sample variation in how strictly anatomy follows the label is high. SPADE in 1b is the architectural fix.
- **No quantitative metrics yet** — judgement is by eye; FID and boundary DSC are Phase 1 endpoints to implement.
- **No augmentation** — small dataset (~32 subjects, ~1,143 slices). Adding coupled image+label augmentation (horizontal flip with L↔R channel swap, small affine) is a quick boost that we've deferred.
- **L/R ovary split is mostly synthetic** — 92.5% of subjects in the
  source data have only one ovary labelled; the L/R distinction is
  largely a midline-split artefact rather than a true anatomical L/R.
  Acknowledged in the methodology.
- **Test set is 8 subjects** — small for downstream DSC evaluation;
  matches the paper, but means individual outlier subjects can swing
  reported metrics noticeably.
