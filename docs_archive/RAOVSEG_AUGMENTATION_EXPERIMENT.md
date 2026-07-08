# RAovSeg Augmentation Experiment — design, pipeline, results, and pressure points

> Document the downstream experiment in full: how the RAovSeg recreation
> works, how we tried to augment it with synthetic data from `exp1c_concat`
> and `exp1c_spade`, what we got, and where the suspected failure modes are.
> Written after the first 6-run augmentation completed and produced
> meaningfully worse DSC than the real-only baseline.

---

## 0. TL;DR

| Variant | DSC (n=3 seeds) | vs baseline 0.290 |
|---|---|---|
| **Baseline** (real-only RAovSeg) | **0.290** | — |
| **1c_concat augmentation** | 0.150 ± 0.006 | **−48% (worse)** |
| **1c_spade augmentation** | 0.138 ± 0.049 | **−52% (worse)** |

Augmentation **hurt** ovary segmentation in every seed × variant
combination. The cause is most likely a distribution mismatch between
real and synthetic data (preprocessing pipeline differences) compounded
by several smaller factors listed in Section 8 below.

---

## 1. Why we ran this experiment

The whole project ladder:
1. Train a conditional DDPM that synthesises pelvic T2FS MRI from labels
2. Compare conditioning mechanisms (concat vs SPADE, ±PatchGAN) → done in
   Phase 1 (results in `RESULTS_2x2.md`)
3. **Test whether the synthetic data actually improves the downstream
   ovary segmentation task** ← this experiment
4. If yes: paper has a positive augmentation result
5. If no: explain why, possibly fix preprocessing or reframe the paper

The downstream task is RAovSeg (Liang et al. 2025, *Sci Data*) ovary
segmentation. Published full-pipeline DSC is **0.290** on the 8 D2 test
subjects. Our recreation reproduces that baseline.

---

## 2. RAovSeg architecture (the recreation)

Two-stage pipeline. All 2D, slice-level.

### 2.1 Stage 1 — ResClass (slice classifier)

Binary classifier: "does this slice contain ovary?" — used to filter out
ovary-negative slices before segmentation.

| Attribute | Value |
|---|---|
| Model | 2-block ResNet, MONAI `ResNetBlock`, features `[8, 16]` |
| Stem | Conv7×7 s=2 → BN → ReLU → MaxPool3×3 s=2 |
| Body | ResNetBlock(8 → 8) → ResNetBlock(8 → 16, stride 2) |
| Head | GAP → Dropout(0.2) → Linear(16, 1) |
| Loss | BCEWithLogitsLoss |
| Optim | Adam, lr=1e-3, weight_decay=1e-4 |
| Batch | 32 |
| Epochs | up to 50, save best by val F1 |
| Augmentation | RandAffine: ±25° rotation, ±25 px translation, prob=1.0, multiplier=5× |
| Train/val split | 60/40 subject-level (no slice-level leakage) |
| Inference threshold | 0.6 (paper unspecified; tuned on val) |

Source: [src/RaovSeg_recreation/train_resclass.py](src/RaovSeg_recreation/train_resclass.py)

### 2.2 Stage 2 — AttUSeg (segmentation)

Per-slice ovary mask predictor. Trained only on ovary-containing slices.

| Attribute | Value |
|---|---|
| Model | MONAI `AttentionUnet` |
| Channels | `[16, 32, 64, 128]` |
| Strides | `(2, 2, 2)` |
| Dropout | 0.2 |
| Loss | Focal Tversky (α=0.8, β=0.2, γ=1.33) |
| Optim | Adam, lr=1e-3, weight_decay=1e-4 |
| Batch | 16 |
| Epochs | 100, save best by val Dice |
| Augmentation | RandAffine: ±25° rotation, ±25 px translation, prob=1.0, multiplier=5×, mode=["bilinear", "nearest"] |
| Train/val split | 80/20 subject-level, ovary-containing slices only |

Source: [src/RaovSeg_recreation/train_attuseg.py](src/RaovSeg_recreation/train_attuseg.py)

### 2.3 Stage 4 — Post-processing

| Step | Implementation |
|---|---|
| Morphological closing | `closing_iterations=10` (paper unspecified; tuned) |
| Largest connected component | Keep only the largest CC per volume |

Both via `RAovSeg_tools.postprocess_()`. Source: external repo at `RAovSeg/RAovSeg_tools.py`.

### 2.4 Evaluation

`evaluate.py` reports DSC under three conditions:
- **`full`**: ResClass filter + AttUSeg + postprocessing → headline metric
- **`no_postprocess`**: ResClass + AttUSeg (no morphological cleanup)
- **`no_resclass`**: AttUSeg on every slice (skip the slice classifier)

Paper benchmarks: full=0.290, no_postprocess=0.235, no_resclass=0.013.

Source: [src/RaovSeg_recreation/evaluate.py](src/RaovSeg_recreation/evaluate.py)

---

## 3. RAovSeg preprocessing pipeline

Applied to every input subject (real or synth) in [src/RaovSeg_recreation/preprocess.py](src/RaovSeg_recreation/preprocess.py).

### 3.1 Subject inclusion criteria

Per the paper:
1. Has a T2FS sequence file (`{subject_id}_T2FS.nii.gz`)
2. Has an ovary label (`{subject_id}_ov.nii.gz`)
3. Does **NOT** have a cyst label (`{subject_id}_cy.nii.gz`)
4. Does **NOT** have an endometrioma label (`{subject_id}_em.nii.gz`)

Subjects passing all four go into the 30 train_val / 8 test split
(deterministic with `SPLIT_SEED=42`).

The 8 sacred test subjects: **D2-005, D2-015, D2-016, D2-017, D2-023, D2-024, D2-026, D2-038**.

### 3.2 Per-subject preprocessing steps

```python
1. Load T2FS NIfTI                                          (SimpleITK)
2. Resample to 0.35 × 0.35 × 6.0 mm at 512 × 512 × native_z (in-plane only)
3. Intensity clip to 1st–99th percentile, min-max → [0, 1]
4. Ovary enhancement (custom):
     if 0.22 ≤ I ≤ 0.30:  I = 1                            (highlight ovary range)
     elif I ≥ 0.5:        I = 1 − I                        (invert bright tissue)
     else:                I = I                            (leave dark tissue)
5. Save as image.npy
6. Resample ovary label NIfTI to same grid (nearest-neighbour)
7. Save as ov_label.npy
```

Output layout per subject:
```
data/processed/{train_val,test,excluded}/D2-XXX/
  image.npy        (Z, 512, 512) float32   ← preprocessed + enhanced
  ov_label.npy     (Z, 512, 512) int8       ← binary ovary mask
```

### 3.3 Augmentation modification (NEW for this experiment)

Added `--extra-train-dir` flag to `preprocess.py`:
- Scans a second directory for D2-XXX subjects
- Forces matching subjects into `train_val` split (skips the 30/8 shuffle)
- Real test split (8 sacred subjects) preserved

This lets synthetic D2-9XX subjects join the real D2-NNN subjects in
train_val without disturbing the test split.

---

## 4. The synthetic data pipeline

How synthetic NIfTI volumes get from the generator into RAovSeg.

### 4.1 Synth volume assembly

[src/Generator/assemble_synthetic_volumes.py](src/Generator/assemble_synthetic_volumes.py)

For each of the 30 train subjects in the generator split (`data/splits/d2_generator_split.json`):

```
1. Load the subject's generator-preprocessed 6-channel label NIfTI
   (body-centered, 512², per-subject in-plane spacing)
2. For each slice z:
     a. Inter-Slice Consistent Stochasticity (ISCS) noise:
        ε_shared ~ N(0, I)   shared across all slices in volume
        ε_z_indep ~ N(0, I)  per-slice
        ε_z = 0.8 · ε_shared + 0.6 · ε_z_indep
     b. DDIM sampling (100 steps, CFG guidance per variant)
        - 1c_concat: g=3.0
        - 1c_spade:  g=2.0
3. Stack slices → 3D synth image
4. Binarise ovary mask = label[ov_L] ∪ label[ov_R]
5. Save NIfTI:
     synth_volumes/exp1c_{variant}/D2-9NN/
       D2-9NN_T2FS.nii.gz    ← synthetic image
       D2-9NN_ov.nii.gz      ← binary ovary mask
```

Source subject D2-001 → synth subject D2-900; D2-002 → D2-901; etc.

### 4.2 SLURM scripts for assembly

- [scripts/assemble_synth_1c_concat.sh](scripts/assemble_synth_1c_concat.sh)
- [scripts/assemble_synth_1c_spade.sh](scripts/assemble_synth_1c_spade.sh)

Resources: 1× A100, 82G RAM, 4 CPUs, `--time=02:00:00`.

### 4.3 RAovSeg augmentation runs

Six SLURM scripts, one per (variant, seed):
- `scripts/run_raovseg_aug_concat_seed{0,1,2}.sh`
- `scripts/run_raovseg_aug_spade_seed{0,1,2}.sh`

Each runs the full pipeline (`preprocess → train_resclass → train_attuseg → evaluate`) with:
- `preprocess.py --data-dir UT-EndoMRI/D2_TCPW --extra-train-dir synth_volumes/exp1c_{variant} --output-dir $OUT_BASE/processed`
- `train_resclass.py --data-dir $OUT_BASE/processed/train_val --output-dir $OUT_BASE/models --seed $SEED`
- `train_attuseg.py --data-dir $OUT_BASE/processed/train_val --output-dir $OUT_BASE/models --seed $SEED`
- `evaluate.py --test-dir $OUT_BASE/processed/test --models-dir $OUT_BASE/models --output-dir $OUT_BASE/predictions`

Each writes to its own `$OUT_BASE = runs/raovseg_aug_{variant}_seed{N}/` so jobs don't clobber each other.

Resources: 1× A100, 82G RAM, 4 CPUs, `--time=08:00:00`. Actual elapsed ~35 min each.

---

## 5. All paths used (HPC layout)

```
/mnt/parscratch/users/ijp25lg/synth_mri/
  EndometriosisDataset/                              ← project repo
    UT-EndoMRI/D2_TCPW/                              ← raw real subjects D2-NNN
    data/
      processed_generator/D2/                        ← generator's preprocessed inputs
      splits/d2_generator_split.json                 ← generator's train/test split
    src/Generator/                                   ← diffusion code
    src/RaovSeg_recreation/                          ← recreation pipeline
      preprocess.py
      train_resclass.py
      train_attuseg.py
      evaluate.py
    RAovSeg/                                         ← external repo (postprocess_, dsc_cal_np)
      RAovSeg_tools.py
    scripts/                                         ← all SLURM scripts
    logs/                                            ← SLURM stdout/stderr

  runs/                                              ← all training outputs
    exp1a/  exp1b/                                   ← original Phase 1
    exp1c_concat/ exp1c_spade/                       ← PatchGAN runs
    raovseg_aug_concat_seed{0,1,2}/                  ← THIS experiment
    raovseg_aug_spade_seed{0,1,2}/                   ← THIS experiment
      processed/
        manifest.csv                                 ← split assignments
        train_val/D2-NNN/{image,ov_label}.npy
        test/D2-NNN/{image,ov_label}.npy
      models/
        resclass_best.pth                            ← Stage 1 weights
        attuseg_best.pth                             ← Stage 2 weights
      predictions/
        D2-NNN_pred.npy                              ← per-test-subject masks

  synth_volumes/                                     ← Stage 1 of THIS experiment
    exp1c_concat/D2-9NN/{T2FS,ov}.nii.gz
    exp1c_spade/D2-9NN/{T2FS,ov}.nii.gz
```

---

## 6. Commands used

### Synth volume assembly (Stage 1)

```bash
# Resources: 1 A100, 2 h limit. Actual time: timed out at 2 h having
# generated 16 (concat) / 19 (spade) of 30 expected volumes.
sbatch scripts/assemble_synth_1c_concat.sh
sbatch scripts/assemble_synth_1c_spade.sh
```

Direct python invocation (from `EndometriosisDataset/`):
```bash
python -m src.Generator.assemble_synthetic_volumes \
  --config src/Generator/exp1c_concat.yaml \
  --ckpt   /mnt/parscratch/users/$USER/synth_mri/runs/exp1c_concat/ckpt/step_100000.pt \
  --gen-preprocessed-root /mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset/data/processed_generator/D2 \
  --gen-split-file        /mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset/data/splits/d2_generator_split.json \
  --out-dir               /mnt/parscratch/users/$USER/synth_mri/synth_volumes/exp1c_concat \
  --iscs-alpha 0.8 --noise-seed 0
```

### RAovSeg augmentation runs (6 jobs)

```bash
for V in concat spade; do
  for S in 0 1 2; do
    sbatch scripts/run_raovseg_aug_${V}_seed${S}.sh
  done
done
```

Each SLURM internally runs:
```bash
python src/RaovSeg_recreation/preprocess.py \
  --data-dir UT-EndoMRI/D2_TCPW \
  --extra-train-dir /mnt/.../synth_volumes/exp1c_${VARIANT} \
  --output-dir $OUT_BASE/processed

python src/RaovSeg_recreation/train_resclass.py \
  --data-dir $OUT_BASE/processed/train_val \
  --output-dir $OUT_BASE/models \
  --seed $SEED

python src/RaovSeg_recreation/train_attuseg.py \
  --data-dir $OUT_BASE/processed/train_val \
  --output-dir $OUT_BASE/models \
  --seed $SEED

python src/RaovSeg_recreation/evaluate.py \
  --test-dir $OUT_BASE/processed/test \
  --models-dir $OUT_BASE/models \
  --output-dir $OUT_BASE/predictions
```

### Diagnostics

```bash
# Job state + elapsed
sacct -u $USER --starttime=now-48hours --name=raov_aug_concat_s0,raov_aug_concat_s1,raov_aug_concat_s2,raov_aug_spade_s0,raov_aug_spade_s1,raov_aug_spade_s2 \
  --format=JobID,JobName%24,State,Elapsed,ExitCode

# Synth count
ls -d synth_volumes/exp1c_concat/D2-* | wc -l
ls -d synth_volumes/exp1c_spade/D2-*  | wc -l

# DSC extraction from logs (evaluate.py prints to stdout, no JSON)
for V in concat spade; do
  for S in 0 1 2; do
    LATEST=$(ls -t logs/raov_aug_${V}_s${S}_*.out | head -1)
    grep -E "^\s+full\s*:" "$LATEST"
  done
done
```

---

## 7. Results — what we actually got

### 7.1 Synth volume counts

| Variant | Expected | Actually assembled | Reason |
|---|---|---|---|
| exp1c_concat | 30 | **16** | 2 h SLURM timeout |
| exp1c_spade | 30 | **19** | 2 h SLURM timeout |

Per-volume cost: 34-44 slices × 100 DDIM steps × CFG (2× forwards) × ~50 ms/step ≈ 5-7 min/volume. 30 volumes × 6 min = 3 h. The 2 h limit cut off the last 11-14 volumes.

### 7.2 Per-job preprocess output

| Variant / seed | train_val subjects | test subjects |
|---|---|---|
| concat seed 0/1/2 | 30 real + 16 synth = **46** | 8 |
| spade seed 0/1/2 | 30 real + 19 synth = **49** | 8 |

(Real D2 dataset has 73 subjects total; 35 are excluded/skip by inclusion criteria, 30 → train_val, 8 → test.)

### 7.3 DSC across all 6 augmentation runs

Full pipeline DSC on the 8 sacred D2 test subjects:

| variant     | seed | DSC (full) | DSC (no_pp) | DSC (no_rc) |
|---|---|---|---|---|
| concat | 0 | 0.1466 | 0.1628 | 0.1429 |
| concat | 1 | 0.1567 | 0.1728 | 0.0978 |
| concat | 2 | 0.1473 | 0.1832 | 0.0794 |
| **concat mean** | — | **0.1502 ± 0.006** | 0.1729 | 0.1067 |
| spade  | 0 | 0.1108 | 0.0748 | 0.1093 |
| spade  | 1 | 0.1087 | 0.1009 | 0.1200 |
| spade  | 2 | 0.1947 | 0.1542 | 0.1951 |
| **spade mean** | — | **0.1381 ± 0.049** | 0.1100 | 0.1415 |
| **Baseline (real-only)** | — | **0.290** | 0.235 | 0.013 |

**Both augmentation variants halved the DSC.** Concat is more
reproducible across seeds (very low std); SPADE shows more seed-to-seed
variance.

### 7.4 Per-subject pattern

Same 3 subjects always near zero across all 6 runs:
- **D2-005, D2-023** — consistently DSC = 0.000
- **D2-024** — almost always 0 (one seed got 0.28)
- **D2-026** — usually 0 (sometimes the postprocess column has signal)
- **D2-015, D2-038** — hit or miss
- **D2-016, D2-017** — best performers, often 0.4–0.7 when they hit

The model trains successfully (AttUSeg validation DSC reaches 0.44) but
fails to generalise to most test subjects. This is a **distribution-
generalization failure**, not a training failure.

---

## 8. Pressure points — things that may be causing the bad DSC

Ranked roughly by suspected impact, top-down. The top 3 are the strongest
hypotheses; everything below is plausible but secondary.

### 8.1 Body-centered vs image-centered preprocessing mismatch (PRIMARY SUSPECT)

The generator was trained on body-centered preprocessing. Its output is
512×512 with the body filling **~90% of frame** at per-subject in-plane
spacing.

The synthetic NIfTI is saved with that body-centered spacing. RAovSeg's
preprocess.py then resamples to (0.35, 0.35, 6.0) at 512×512 — which
preserves the body-centered framing for synth subjects.

But real subjects start at their original raw spacing (~0.5-0.6 mm/px
typical for pelvic MRI), with body filling ~60% of the original FOV.
After RAovSeg's resample they're at 512×512 / 0.35 mm/px → 179.2 mm FOV —
the body still fills only ~60% of the frame.

**Result**: during training the model sees two completely different
"pelvis" framings. Real: body 60% of frame, lots of context. Synth: body
90% of frame, little context. The model can't reconcile them and fails
to generalise to the framing it sees less often (real).

### 8.2 The outside-body hallucinations we deferred fixing

The 1c models have well-documented hallucinations: bright fat-like
structures generated OUTSIDE the body silhouette. This was diagnosed
earlier and the post-process body mask fix (Tier A1 in NEXT_STEPS.md)
was deferred per the user's "leave things as is" instruction.

These hallucinations train RAovSeg to expect structured noise outside
the body silhouette. Real test images don't have this. → confused model.

### 8.3 Intensity distribution shift between synth and real T2FS

RAovSeg's enhancement step uses **fixed intensity windows** tuned for
real T2FS:
- `o1 = 0.22, o2 = 0.30` → these voxel intensities get highlighted
  (set to 1) because real T2FS ovaries fall there

If the synth produces images whose ovary intensities are NOT in [0.22,
0.30] (e.g. our synth has slightly different intensity statistics —
visible in the high hist_KL of 5.79–8.15 vs real), the enhancement step
may either:
- Not highlight the synth ovary (which is at a different intensity)
- Or worse: highlight the wrong region, creating false "ovary" signals

This could explain why some test subjects get DSC=0 (model trained on
synth where the enhancement highlighted wrong areas).

### 8.4 Synth may visually resemble T2 (not T2FS) due to high FID

Our 1c models had FID 166–200 vs real T2FS. That's substantially off
from the real distribution. Possible the synth doesn't faithfully
reproduce the *fat-suppressed* character of T2FS — fat may not be
suppressed enough in synth, making synth look more like T2 (bright fat)
than T2FS (dark fat).

If the model trains on synth-looking-like-T2, it learns to expect bright
fat — then on real T2FS test images (dark fat), it doesn't find what it
expects.

### 8.5 Only 16/19 synth volumes (not the planned 30)

Partial assembly cuts the synth:real ratio:
- concat: 16/30 = 35% synth
- spade: 19/30 = 39% synth

The user's intuition that "not enough patients" might be a factor: the
*real* paper had 30 train subjects and got DSC=0.290. We're using
46-49 train_val (30 real + 16-19 synth). If the synth subjects are
adding NOISE (per the hypotheses above), more of them might HURT more.
But if they're adding SIGNAL, more might HELP.

Currently impossible to disentangle until we re-run with the full 30.

### 8.6 Train/val split changed proportions

Baseline: 30 train_val subjects → at 60% train ratio: 18 train + 12 val.

Augmented: 46-49 train_val → at 60% train ratio: 27-29 train + 19-20 val.

The validation set might now contain too many synthetic subjects. If
val is dominated by synth and the model picks "best by val Dice", we may
be selecting checkpoints that fit synth distribution best — which would
hurt real test performance. The AttUSeg val DSC reached 0.44 during
training; real test DSC was 0.15. **Classic overfit signature.**

### 8.7 The synth labels are exact copies of real labels — no novel anatomy

We generate one synth per real train subject, conditioned on that
subject's existing label. So the synth doesn't add new anatomical
configurations — it just adds more *images* with the same label
distribution.

Augmentation theory: adding more images of the same labels helps if the
images are diverse enough. But our synth + real of the same subject are
visually similar (same anatomy), so the diversity gain is mostly in
texture / noise patterns. If those are also off-distribution (per 8.1-8.4),
we're not adding signal.

### 8.8 Inclusion criteria might filter some synth

`preprocess.py` excludes subjects that have `_cy.nii.gz` or `_em.nii.gz`
labels. Our synth subjects only have `_T2FS.nii.gz` + `_ov.nii.gz`, so
they should pass. **Verified in manifest** — all assembled synth (16 / 19)
made it into train_val.

So this isn't an issue, but worth verifying that no further filtering
happens silently.

### 8.9 Loss function may not balance synth + real well

Focal Tversky (α=0.8, β=0.2, γ=1.33) is asymmetric — heavy penalty for
false negatives. If synth ovary masks have slightly different shapes
than real ovary masks (per spatial body-centering), the loss may push
the model to predict masks that fit synth shape, hurting real.

### 8.10 Augmentation pipeline interaction

RAovSeg applies RandAffine (±25° rotation, ±25 px translation, 5×
multiplier) during training. On synth subjects (body-centered, body fills
90% of frame), a ±25 px translation might push the body out of frame
much more than on real subjects (body in middle 60%). This could create
out-of-distribution training samples specifically from synth.

### 8.11 ResClass overfitting to synth slice patterns

ResClass classifies each slice as ovary+/ovary−. Training stdout shows
precision/recall stuck at 0.0 / 0.0 for many epochs (model just predicts
"no ovary" for everything), then occasionally jumping. If the classifier
learned that "synth-looking slices = ovary" and "real-looking slices =
not ovary", it would suppress almost everything on real test data — which
matches the per-subject pattern of DSC=0 for many subjects.

### 8.12 Spacing-induced resampling artifacts in synth

`assemble_synthetic_volumes.py` saves synth NIfTI with header `CopyInformation()`
from the generator's preprocessed image (per-subject body-centered spacing).
RAovSeg's preprocess does NN resampling for labels but bilinear for images.
The combination could produce edge artifacts where synth image and
synth label disagree on organ boundaries — confusing the model.

---

## 8b. Diagnostic confirmation — side-by-side comparison after RAovSeg preprocess

We loaded `data/processed/train_val/D2-001/image.npy` (real, post-RAovSeg-preprocess)
and `data/processed/train_val/D2-900/image.npy` (synth from `1c_concat`, post-same-
preprocess) and inspected them side-by-side. The figure
[synth_vs_real_after_raovseg_preprocess.png](synth_vs_real_after_raovseg_preprocess.png)
confirms **every** primary suspect from Section 8:

### Confirmed: Problem 8.1 — FOV mismatch
- Real D2-001: body sits in the **middle** of the frame, surrounded by clear
  black background. Body fills ~55-60% of the frame.
- Synth D2-900: body **fills the entire frame** (~90%). The body-centered
  preprocessing from the generator persists straight through RAovSeg's
  resampling step.

### Confirmed: Problem 8.2 — Outside-body hallucinations surviving preprocess
- Real D2-001: outside-body is **black** (uniform dark).
- Synth D2-900: outside-body is filled with **structured grainy noise** —
  the same hallucination pattern visible in the earlier explainability
  diagnostics, now amplified by the percentile-clip and normalization
  inside RAovSeg's preprocess step.

### Confirmed: Problem 8.3 + 8.4 — The intensity enhancement step is failing on synth (MOST DAMAGING)

The RAovSeg enhancement rule is: voxels in `[0.22, 0.30]` → set to **1**
(highlighted as ovary). Then the model trains on images where the ovary is
visually obvious — a saturated white blob the model just has to localise.

The intensity histograms differ dramatically:
- **Real D2-001 histogram**: large mass near 0 (background), small bump
  0.0-0.2 (body soft tissue), and a **massive spike at 1.0** — the
  enhancement firing strongly on real ovary tissue.
- **Synth D2-900 histogram**: background mass near 0 is much smaller
  (because outside-body has noise, not zeros), big mass spread from 0.2 to
  0.5 (mostly above the o1=0.22 threshold but not cleanly inside the
  [0.22, 0.30] window), and the spike at 1.0 is **much smaller**.

Looking at the overlay panels:
- **Real**: red ovary overlay sits on a **bright white (enhanced) region**
  — the model is shown "the ovary is THIS bright thing here."
- **Synth**: red ovary overlay sits on a **medium-gray region
  indistinguishable from surrounding tissue** — the enhancement didn't
  fire, so the model has no visual hint where the ovary is.

This is the smoking gun. RAovSeg was designed around the assumption that
ovary intensity lands in [0.22, 0.30] post-normalization. Our synth
doesn't satisfy this assumption — its ovary intensities are distributed
elsewhere — so the enhancement makes synth's ovary look *less* visible,
not more.

### Resulting training dynamic
The model trains on a mix of:
- Real images where ovary = obvious bright blob (enhancement fired) → easy
- Synth images where ovary = average gray tissue (enhancement missed) → hard

The model likely learns "predict ovary at the brightest blob," which:
- Works on real test data where the enhancement fires (D2-016, D2-017 get DSC=0.5+)
- Fails on real test data where the enhancement misses or is ambiguous
  (D2-005, D2-023, D2-024 consistently get DSC=0)

### Decision adopted from this diagnostic
We commit to **Option B** (fix the three pressure points and re-run):
1. Apply a body silhouette mask to the synth before saving (kills outside-body noise)
2. Histogram-match the synth to the source real subject's intensity distribution
   (so the ovary lands in RAovSeg's [0.22, 0.30] enhancement window)
3. Resample the synth NIfTI to the source real subject's frame
   (kills the body-centered → image-centered FOV mismatch)

Plus bump the synth-assembly SLURM time limit to 4 h so all 30 volumes
assemble (instead of timing out at 16-19).

---

## 8c. v2 fix attempt — three corrective fixes applied

Applied three fixes to [src/Generator/assemble_synthetic_volumes.py](src/Generator/assemble_synthetic_volumes.py):

1. **Body silhouette mask** (`--no-body-mask` to disable): set synth pixels
   to 0 where the `outside_body` label channel is 1, killing the
   outside-body hallucinations.
2. **Histogram match** (`--no-histogram-match` to disable): match the
   synth's intensity distribution to the source real subject's raw
   intensity distribution (after the same percentile-clip that RAovSeg
   will apply), so the synth ends up at a similar [0, 1] distribution
   shape post-preprocess.
3. **Resample to source real frame** (`--no-resample-to-source` to
   disable): use SimpleITK's `ResampleImageFilter` with the raw real
   subject's NIfTI as the reference, transferring the synth (body-centered
   frame) into the real's image-centered frame at matching spacing/origin/
   direction. After RAovSeg's preprocess, the synth has the same FOV
   framing as real.

Bumped `assemble_synth_*.sh` time limit from 2 h → 4 h. All 30 synth
volumes assembled per variant (previously was 16-19 due to timeout).

### 8c.1 Visual confirmation — the fixes worked

Re-ran the diagnostic comparison
([synth_vs_real_v2.png](synth_vs_real_v2.png)). All three fixes
visually accomplished their stated goal:

- **FOV match**: synth body now fills ~60% of frame, same as real (was 90%)
- **Outside-body dark**: synth outside-body is now clean black (was filled
  with structured noise)
- **Histogram shape**: synth and real histograms now have very similar
  overall shapes — big mass near 0, decay through 0.0-0.3, small mass
  at higher intensities

### 8c.2 DSC results after v2 fixes

| Variant | v1 (no fixes) | v2 (3 fixes) | Δ vs v1 | vs baseline 0.290 |
|---|---|---|---|---|
| **concat** | 0.150 ± 0.006 | **0.044 ± 0.039** | −71% (WORSE) | −85% |
| **spade** | 0.138 ± 0.049 | **0.169 ± 0.037** | +22% (marginal) | −42% |

The fixes had **opposite effects** on the two variants. SPADE improved
modestly; concat collapsed. concat seed 2 specifically returned DSC =
0.000 on every test subject.

### 8c.3 Why the v2 fixes didn't recover baseline

Despite matching real visually, the **intensity enhancement step is still
firing much less on synth** than on real:

| | Pixels at intensity 1.0 (post-enhancement) |
|---|---|
| Real | ~400k |
| concat synth v2 | ~100k (1/4 of real) |
| spade synth v2 | ~150k (1/3 of real) |

Root cause: **histogram matching is rank-based, not semantic**. It aligns
the synth's brightest pixels with the real's brightest pixels by rank.
- In real T2FS, the rank-95+ pixels happen to be ovary tissue (which is
  what the enhancement targets at [0.22, 0.30]).
- In synth, the rank-95+ pixels are **whatever the diffusion model
  decided to make bright** — not necessarily the ovary. For SPADE
  (which has weak per-organ localisation, CLR ~0.4) those bright pixels
  are at least near the ovary. For concat (CLR ~0.04, essentially no
  per-organ localisation) the bright pixels are randomly distributed.

Result: histogram-matched concat synth has enhanced regions in WRONG
locations → model trains to look for ovary at wrong locations → fails
on real test.

### 8c.4 Decision: try Path B, fall back to Path C

After the visual confirmation that the framing/intensity-shape fixes
work but the enhancement assumption isn't satisfied for the synth ovary
specifically, we adopt this plan:

**Path B (try first) — label-aware ovary intensity rescaling.** Add an
explicit step that uses the synth's ovary label mask to ENSURE the synth
ovary pixels land at intensity ~0.26 (middle of the [0.22, 0.30]
enhancement window) after RAovSeg's normalization. Computes a per-volume
additive offset on the ovary region only. Implemented as
`--ovary-target-intensity 0.26` flag in `assemble_synthetic_volumes.py`,
default ON.

This directly targets the failure mode: instead of hoping histogram
matching will place the ovary pixels at the right intensity by rank
luck, we explicitly force them there using the label.

**Path C (fall back if Path B is futile) — disable enhancement for synth
subjects.** Modify `preprocess.py` to skip the o1/o2 enhancement step
when the subject ID matches `D2-9XX`. Both real and synth then have the
same intensity scale (percentile clip + minmax only). This trades the
enhancement-as-localisation hint that benefits real for a consistent
training distribution. Likely also hurts the real baseline, so this is
a fallback to confirm whether the enhancement-mismatch hypothesis is
correct, not a path to a positive result.

If Path B recovers baseline (~0.290) or beats it, we have a positive
augmentation result and Path C is unnecessary. If Path B still under-
performs, Path C tells us whether removing the enhancement entirely
makes synth useful (proving the enhancement step is the actual bottleneck)
or whether the synth quality itself is the limit (in which case we accept
the negative result for the paper).

---

## 8d. v3 results — Path B (label-aware ovary intensity rescaling)

### 8d.1 Trajectory across all versions

| Version | concat | spade |
|---|---|---|
| **Baseline** (real-only) | **0.290** | **0.290** |
| v1 (no fixes) | 0.150 ± 0.006 | 0.138 ± 0.049 |
| v2 (3 preprocessing fixes) | 0.044 ± 0.039 | 0.169 ± 0.037 |
| **v3 (3 fixes + Path B)** | **0.053 ± 0.056** | **0.218 ± 0.057** |

**SPADE seed 0 in v3: DSC = 0.276** — within 0.014 of the real-only
baseline. SPADE's trajectory 0.138 → 0.169 → 0.218 shows Path B is
genuinely helping.

**Concat stuck around 0.05 regardless of fixes.** Path B did not move
the needle for concat.

### 8d.2 Why the architectural split makes sense

This maps cleanly to the CLR (Counterfactual Localisation Ratio) finding
from Phase 1:

| Variant | CLR_uterus | Path B outcome |
|---|---|---|
| concat | 0.013–0.069 (essentially no per-organ localisation) | Doesn't help — forces "ovary region" to be enhanced, but concat's synth doesn't actually contain ovary content at that location. Path B puts a bright blob where the label says, disconnected from surrounding synth tissue. Model can't learn from this. |
| SPADE | 0.407–0.494 (real per-organ localisation) | Helps — SPADE's synth ovary region DOES have ovary-shaped content. Path B just ensures it lands at the right intensity for RAovSeg's enhancement to fire. |

**Path B works when the generator can localise the ovary.** Concat can't
→ intensity shifting doesn't rescue it. SPADE can → intensity shifting
turns "correctly-located but wrong-intensity" ovary into "correctly-
located, correctly-highlighted" ovary that RAovSeg can find.

### 8d.3 The paper story is now cohesive

- **Phase 1** (generator ablation) established that SPADE achieves 5-10×
  higher CLR than concat.
- **Phase 4** (downstream augmentation, this experiment) confirms that
  the per-organ localisation difference has real downstream consequences:
  concat is architecturally locked out of useful augmentation, SPADE can
  approach the real-only baseline once the preprocessing pipeline mismatch
  is corrected.
- **Meta-lesson**: for label-aware downstream tasks, per-organ localisation
  at the generator matters more than raw image realism.

## 8e. Decision: do Options B AND C to complete the story

Two remaining experiments to determine whether SPADE augmentation can
match or beat the real-only baseline of 0.290:

### Option B — sweep the ovary target intensity (SPADE-only)

The current Path B target `0.26` (middle of RAovSeg's [0.22, 0.30]
enhancement window) was a first guess. Values at the edges of the window
might work better if there's an interaction with the percentile-clip that
shifts things. Try:
- `--ovary-target-intensity 0.22` (lower edge of enhancement window)
- `--ovary-target-intensity 0.28` (upper edge)

Compare against v3's 0.26 default. If either wins → adopt it.

Only run for SPADE (concat is a lost cause per §8d.2).

### Option C — disable enhancement for synth subjects (SPADE-only)

Modify `preprocess.py` to skip the o1/o2 enhancement step when the
subject ID matches `D2-9*` (our synth prefix). Real subjects still get
the enhancement.

Rationale: even with Path B pushing the synth ovary to the target
intensity, the enhancement step may still misfire on synth subjects
because their body-tissue distribution differs from real. Skipping
enhancement for synth means the synth trains as-is (percentile-clipped
+ minmax'd only), while real still benefits from enhancement at both
train and test time.

This tests whether the enhancement step itself is the bottleneck. If
Option C beats Option B, enhancement was actively hurting synth
utility. If Option C is worse, enhancement was neutral or helpful even
without perfect alignment.

Only run for SPADE.

### What we'll conclude after both

- **If Option B (target=0.22 or 0.28) or Option C beats v3's 0.218 and
  approaches 0.290**: positive result, SPADE augmentation can match
  real-only baseline with the right pipeline configuration. Paper
  headline is a clean "generator architecture + preprocessing alignment
  enables useful augmentation."
- **If neither exceeds 0.218**: v3 (0.218) is the best SPADE
  augmentation can do at this synth quality / data scale. Paper
  headline stays as "SPADE approaches but doesn't fully match baseline;
  per-organ localisation matters."
- **If both hurt SPADE**: v3 was the peak, further tuning made things
  worse. Same paper conclusion as above.

## 8f. Options B and C results — v3 (t=0.26) was the ceiling

### 8f.1 Full trajectory including B and C

| Config | seed0 | seed1 | seed2 | Mean | Std |
|---|---|---|---|---|---|
| **Baseline** (real-only) | — | — | — | **0.290** | — |
| v3 SPADE (t=0.26) | 0.2755 | 0.1620 | 0.2167 | **0.2181** | 0.057 |
| Opt B SPADE t=0.22 | 0.1192 | 0.1008 | 0.2753 | 0.1651 | 0.096 |
| Opt B SPADE t=0.28 | 0.1363 | 0.1236 | **0.3061** | 0.1887 | 0.102 |
| Opt C SPADE no-enh | 0.1345 | 0.2349 | 0.1405 | 0.1700 | 0.056 |

### 8f.2 What the sweep + toggle told us

1. **t=0.26 was serendipitously optimal.** Moving the ovary target
   intensity away from the middle of the enhancement window (down to
   0.22 or up to 0.28) reduced the mean DSC. Confirms "put the ovary
   in the middle of the enhancement window" was the right heuristic.

2. **The enhancement step is helpful, not hurtful.** Option C (disable
   enhancement for synth) got DSC 0.170 — worse than v3's 0.218. So
   when the enhancement DOES fire correctly on synth (via Path B), it
   IS beneficial. Removing enhancement doesn't rescue synth utility.

3. **We've hit the synth-quality ceiling.** Multiple targeted
   interventions (framing fix, outside-body mask, intensity
   distribution match, ovary-specific rescale, enhancement toggle) all
   converge on ~0.17-0.22 mean DSC for SPADE augmentation. That's the
   ceiling of what SPADE synth at this quality level (FID ~188,
   hist_KL ~7.2) can contribute.

4. **One tantalising signal (later invalidated)**: Opt B t=0.28 seed 2
   got DSC = 0.306, ABOVE baseline. v3 seed 0 got 0.276, just below.
   Original interpretation was that variance was masking a real
   augmentation benefit. **The n=8 variance study in §8g overturned
   this**: seeds 0/1/2 were cherry-picked upward — expanding to 8 seeds
   drops the v3 mean from 0.218 to 0.178, and the "variance drags the
   mean down" argument doesn't survive.

### 8f.3 DSC picture (all versions × both variants) — pre-variance-study numbers

*The v3 SPADE cell is corrected in §8g.3 once n=8 data lands.*

| Version | concat | spade (n=3) |
|---|---|---|
| Baseline (real-only) | 0.290 | 0.290 |
| v1 (no fixes) | 0.150 ± 0.006 | 0.138 ± 0.049 |
| v2 (3 preprocessing fixes) | 0.044 ± 0.039 | 0.169 ± 0.037 |
| v3 (3 fixes + Path B, t=0.26) | 0.053 ± 0.056 | 0.218 ± 0.057 |
| Opt B (Path B, t=0.22) | — | 0.165 ± 0.096 |
| Opt B (Path B, t=0.28) | — | 0.189 ± 0.102 |
| Opt C (no enhancement for synth) | — | 0.170 ± 0.056 |

### 8f.4 The paper story (superseded by §8g)

The claims below were the pre-variance-study reading. §8g documents
which survive and which don't after n=8.

1. **Concat augmentation is architecturally broken** at every
   intervention level (v1, v2, v3 all ≈ 0.05-0.15). Concat's lack of
   per-organ localisation (CLR ~0.03) means the label-aware fixes
   have nothing to align. — *STILL HOLDS after n=8.*

2. **SPADE augmentation approaches baseline with correct preprocessing**.
   — *WEAKENED*: n=8 shows 0.178, not 0.218. Still trends upward with
   preprocessing fixes but does NOT close the gap. See §8g.

3. **Preprocessing pipeline alignment matters more than raw synth
   quality** — meta-lesson: synth generators for downstream augmentation
   must be designed with awareness of the downstream consumer's
   assumptions. — *STILL HOLDS*: this is a story about relative
   improvement across v1 → v2 → v3, which the n=8 confirms.

### 8g. Variance study — n=8 seeds of v3 SPADE (t=0.26)

**Motivation**: n=3 seed samples give a very wide CI (std 0.057, so the
mean is only known to ±0.06). To decide whether variance is masking a
real augmentation benefit, we ran 5 more seeds (3-7) of the same v3
config and analyzed both cross-seed and per-subject variance.

#### 8g.1 Aggregate n=8 result

| | n=3 (§8d,e,f) | **n=8 (§8g)** |
|---|---|---|
| Mean DSC | 0.2181 | **0.1783** |
| Std across seeds | 0.0570 | 0.0537 |
| Best seed | 0.2755 (s0) | 0.2755 (s0) |
| Gap to baseline (0.290) | −25% | **−38%** |

The extra 5 seeds averaged 0.1544 ± 0.0378 — well below the original
three. The original 0.218 mean was **luck**, not a stable estimate.

#### 8g.2 Per-subject variance dwarfs cross-seed variance

Across the 8 sacred test subjects × 8 seeds:

| Subject | Seeds hitting DSC > 0.1 | Behavior |
|---|---|---|
| D2-017 | **8/8** (all ~0.5) | Reliably segmented |
| D2-016 | 6/8 (2 total failures) | Usually great |
| D2-015 | 7/8 (variable 0.14–0.60) | Seed-dependent |
| D2-038 | 2/8 | Rare hit |
| D2-024 | 1/8 (only s0) | One-hit wonder |
| D2-026 | 1/8 (only s7) | One-hit wonder |
| **D2-005** | **0/8** | **Universal failure** |
| **D2-023** | **0/8** | **Universal failure** |

Two test subjects (D2-005 and D2-023) get DSC = 0 across every seed —
structural failure, not variance. That alone caps the achievable mean
at ~0.22 even if every other subject were perfectly segmented.

Std WITHIN a seed (across 8 subjects) is ~0.24, **4× larger than
std ACROSS seeds** (~0.054). The dominant variance axis is per-subject,
not per-training-run.

#### 8g.3 Corrected v3 SPADE cell

The v3 SPADE row in §8f.3 should read (after n=8):

| Version | concat | spade (n=8) |
|---|---|---|
| v3 (3 fixes + Path B, t=0.26) | 0.053 ± 0.056 | **0.178 ± 0.054** |

#### 8g.4 What §8g tells the paper

1. **The "variance masks a real benefit" story is dead.** SPADE augmentation
   under Phase 1 conditions robustly underperforms the real-only baseline
   by ~38%. The gap (0.11) is 2× the cross-seed std (0.054).

2. **The DSC ceiling for Phase 1 augmentation is ~0.18** — no further
   preprocessing tuning will move this meaningfully. Confirmed by the
   Options B/C sweep (all landed lower) and now by the variance study.

3. **Per-subject structural failure** on D2-005 and D2-023 suggests
   these subjects are outside the augmented pipeline's competence
   entirely. Worth checking against the baseline's per-subject numbers
   in future analysis — if baseline also fails on them, it's a dataset
   property; if only the augmented pipeline fails, it's a distribution-shift
   artefact of the synth data.

4. **Phase 1 is complete for the paper.** Any narrative "synth helps
   downstream segmentation" now REQUIRES Phase 2 (cross-domain D1→D2)
   to move the needle. Continuing to tune Phase 1 augmentation has no
   remaining upside.

### 8h. Phase 2 — cross-domain (D1 gen + D2 disc) downstream result

**Configuration**: exp2 = SPADE generator trained on D1 T2 preprocessed
(32 subjects), PatchGAN discriminator trained on D2 T2FS (~41 subjects),
D unconditional (label zeroed), λ_peak=0.01, 100k steps.

**Assembly**: 32 synth volumes from D1 preprocessed masks, resampled into
D1 raw subject frames, ovary target intensity t=0.26 (inherited from
Phase 1 v3), body silhouette mask + histogram match + resample-to-source
all ON.

**Downstream DSC (n=3 seeds)**:

| Seed | DSC (full) | DSC (no_pp) | DSC (no_rc) |
|---|---|---|---|
| 0 | 0.0266 | ? | ? |
| 1 | 0.0255 | ? | ? |
| 2 | 0.0089 | ? | ? |
| **Mean** | **0.0203** | | |
| Std | ~0.010 | | |

vs Phase 1 v3 SPADE (n=8): 0.178 ± 0.054
vs real-only baseline: 0.290
**Gap to baseline: −93%.**

#### 8h.1 Why Phase 2 collapsed

Sample grids at every step from 5k → 95k show the generator plateaued at
"gray-blob body silhouette with textured noise" — no distinct T2FS style
acquired, no visible organ structure. Diagnosis:

1. **DDPM MSE loss on D1 T2 dominated adversarial signal at λ=0.01.**
   MSE says "reconstruct D1 T2 (bright, non-fat-sup)"; adversarial says
   "look more like D2 T2FS (dark, fat-sup)". The two objectives are
   antagonistic and MSE won.
2. **Unconditional D provided weaker gradient than the Phase 1 conditional
   D.** Zeroing the label (to avoid D1/D2 label-distribution shortcuts)
   removed the label-consistency signal, leaving D judging pure style at
   pixel level. Not enough to counter MSE.
3. **Fewer training subjects (32 D1 vs 41 D2 in Phase 1) + larger domain
   gap (T2→T2FS vs D2→D2)** compounded the two above.

#### 8h.2 Why the −93% DSC drop is stable, not noise

- 3 seeds within std ~0.010 (much tighter than Phase 1's ~0.054).
- All 3 seeds land in the same failure mode: model predicts near-zero
  ovary on essentially every test subject.
- Bad synth doesn't just waste training capacity — it corrupts the real
  signal. At n=30 real, ~30 pieces of gray-blob "training data" is enough
  to break the model's ovary-detection prior.

#### 8h.3 What Phase 2 tells the paper

1. **Bad synth is worse than no synth.** At data-scarce clinical scales,
   augmentation quality is not optional — a mediocre generator poisons
   training rather than neutrally not-helping. This is the strongest
   practical lesson from the whole two-phase study.
2. **Naive DDPM + adversarial cross-domain translation for MRI style
   transfer does not work** with the standard schedule (λ_peak=0.01,
   PatchGAN base_channels=64, 100k steps). The MSE reconstruction loss
   is too dominant.
3. **The −93% DSC is a much cleaner negative-result claim than the
   Phase 1 −39%.** Paper story: Phase 1 shows synth augmentation helps
   marginally then plateaus below baseline; Phase 2 shows cross-domain
   extension actively harms. Together they map the design space where
   synth augmentation fails in this regime.

#### 8h.4 exp2_lam05 (Track 2, tuning attempt)

Retrained with `lambda_peak: 0.05` (5× exp2). Status: [PENDING —
paste result when SLURM job completes].

Expected outcome: even if λ=0.05 makes the samples look more T2FS-like,
downstream DSC is unlikely to recover to Phase 1 levels (~0.18), let
alone baseline (0.290). Track 2 is a diagnostic — did we abandon exp2
prematurely? — not a rescue attempt. If DSC lands in [0.05, 0.15], we
report the tuning as a marginal improvement over exp2 but still a Phase 2
negative result. If ≥0.15, we'd reconsider λ-only tuning as a research
direction for the paper's future-work section.

---

## 9. What we could try to fix it

In rough order of likely impact:

| Fix | Addresses | Effort | Likely impact |
|---|---|---|---|
| A. **Save synth NIfTI at image-centered spacing matching real** | 8.1 | ~50 lines | HIGH (primary suspect) |
| B. **Apply post-process body mask before saving synth** | 8.2 | ~30 lines | MEDIUM |
| C. **Bump assembly --time to 4 h, get all 30 synth** | 8.5 | 1 line | LOW-MEDIUM |
| D. **Reduce synth:real ratio (use fewer synth per training run)** | 8.6 | config change | LOW-MEDIUM |
| E. **Inspect a synth slice after RAovSeg preprocess vs a real one** | 8.1, 8.3, 8.4 | quick diagnostic | LOW cost, HIGH info |
| F. **Histogram-match synth intensities to real before saving** | 8.3, 8.4 | ~30 lines | MEDIUM |
| G. **Skip the RAovSeg enhancement step for synth subjects** | 8.3 | preprocess.py change | UNKNOWN |

**Recommendation**: do (E) first as a 30-min diagnostic, then commit to
either (A) + (B) + (C) (full fix → re-run) or accept the negative result
and write it up.

---

## 10. Negative result framing for the paper

Even if we don't fix the bad DSC, the experimental design + negative
result is publishable:

> *"We tested whether synthetic data from our four 2D conditional DDPM
> variants improves downstream ovary segmentation (RAovSeg pipeline). At
> 1:0.5 augmentation ratio (16-19 synthetic volumes added to 30 real),
> both PatchGAN-augmented variants reduced ovary DSC by ~50% compared
> to the real-only baseline (0.150 ± 0.006 for concat+GAN, 0.138 ± 0.049
> for SPADE+GAN, vs. baseline 0.290; n=3 seeds per variant). Per-subject
> analysis showed the model failed to predict any ovary for 3 of 8 test
> subjects consistently. We attribute the failure primarily to a
> preprocessing pipeline mismatch: the generator's body-centered output
> has the body filling ~90% of the frame, while the downstream pipeline
> expects images where the body fills ~60% of the frame. This
> distribution shift, combined with intensity-histogram mismatch (KL ≈
> 6-8 vs real), prevents the synthetic data from contributing positively
> to segmentation training."*

This is honest, replicable, and tells the field something useful: **the
synthetic generator must match the downstream consumer's preprocessing
expectations, or augmentation backfires.**

---

## 11. Files / artifacts

```
Documentation:
  RAOVSEG_AUGMENTATION_EXPERIMENT.md          ← this file
  NEXT_STEPS.md                               ← the original forward plan
  RESULTS_2x2.md                              ← Phase-1 generator results
  EXP1B_SUMMARY.md, EXP1C_SUMMARY.md          ← per-experiment context

Code:
  src/Generator/assemble_synthetic_volumes.py ← NEW: builds synth NIfTI volumes
  src/RaovSeg_recreation/preprocess.py        ← MODIFIED: --extra-train-dir
  src/RaovSeg_recreation/train_resclass.py    ← MODIFIED: --seed
  src/RaovSeg_recreation/train_attuseg.py     ← MODIFIED: --seed
  src/RaovSeg_recreation/evaluate.py          ← UNCHANGED

SLURM:
  scripts/assemble_synth_1c_concat.sh         ← NEW (Stage 1)
  scripts/assemble_synth_1c_spade.sh          ← NEW (Stage 1)
  scripts/run_raovseg_aug_concat_seed{0,1,2}.sh   ← NEW (Stage 3)
  scripts/run_raovseg_aug_spade_seed{0,1,2}.sh    ← NEW (Stage 3)

Outputs on HPC (not pulled to local):
  /mnt/parscratch/users/ijp25lg/synth_mri/runs/raovseg_aug_*_seed*/
  /mnt/parscratch/users/ijp25lg/synth_mri/synth_volumes/exp1c_*/
```
