# 03 — Data and downstream pipeline

> UT-EndoMRI datasets, splits, sacred test set, label design, and the
> RAovSeg downstream pipeline in enough detail to reproduce and to reason
> about the failure modes. Source: `../docs_archive/RAOVSEG_AUGMENTATION_EXPERIMENT.md`,
> `../docs_archive/TRAINING_OVERVIEW.md`, `../docs_archive/architecture_dataflow_v2.md`,
> `../docs_archive/synthetic_mri_generator_design.md`.

---

## 3.1 UT-EndoMRI overview

The Liang et al. (2025) paper releases UT-EndoMRI as an open dataset
paired with the RAovSeg pipeline. It comprises two site cohorts:

| Cohort | Site | Sequence | Subjects on disk | Fat suppression |
|---|---|---|---|---|
| **D1_MHS** | Memorial Hermann | T2-weighted | 51 | No (bright fat) |
| **D2_TCPW** | TCPW | T2-weighted fat-suppressed (T2FS) | ~73 | Yes (dark fat) |

Both cohorts are pelvic MRI acquired for endometriosis workup. Each
subject volume comes with per-organ manual annotation files:

- `{subject}_T2FS.nii.gz` (or `_T2.nii.gz` for D1) — the image volume.
- `{subject}_ut.nii.gz` — uterus mask.
- `{subject}_ov.nii.gz` — combined left+right ovary mask.
- `{subject}_em.nii.gz` — endometrioma mask (present only for
  em-positive subjects).
- `{subject}_cy.nii.gz` — cyst mask (present only for cyst-positive
  subjects, which are excluded from RAovSeg).

The Right vs Left ovary split for our labels is not stored explicitly by
the annotators. It is done in image-space by
`src/Generator/preprocess_for_generator.py` — connected-component
analysis on the combined ovary mask, then labelling each CC as `ov_L` or
`ov_R` by its x-coordinate relative to the image midline. This is a
pragmatic split; a small fraction of subjects with unusual ovary
positioning may have swapped labels — not consequential for downstream
DSC (which is measured on the *combined* ovary mask).

## 3.2 Subject filtering — from raw to usable

### 3.2.1 D2_TCPW — the primary cohort

Of the ~73 raw D2 subjects on disk, several filters reduce the pool:

| Filter | Removed | Why |
|---|---|---|
| Missing T2FS image | 3 | Can't train without the image |
| Missing uterus OR combined ovary mask | ~9 | Required by `build_generator_split.py` inclusion filter |
| RAovSeg test subjects (sacred) | 8 | Held out for downstream evaluation, never touched by the generator |
| **Effective generator training pool** | **32** | Pass all three checks |

The **32 D2 subjects** are what the four Phase 1 generators (1a, 1b,
1c_concat, 1c_spade) train on. Every slice from each subject
contributes to training, but a weighted sampler over ovary-containing
slices boosts them 3× at sampling time — so ovary-slices go from a
natural ~13% of the slice pool to ~30% of training batches. This
compensates for the class imbalance (most pelvic slices don't show
ovaries).

### 3.2.2 RAovSeg's own inclusion criteria (paper)

Different from the generator's filter. RAovSeg further excludes
subjects with cyst or endometrioma labels present:

1. Has `_T2FS.nii.gz`.
2. Has `_ov.nii.gz`.
3. Does NOT have `_cy.nii.gz`.
4. Does NOT have `_em.nii.gz`.

Subjects passing all four go through a deterministic split (`SPLIT_SEED
= 42`) into 30 train_val + 8 test.

The gap between the generator's 32 and RAovSeg's 30 is because the
generator *does* accept em-positive subjects (since `em` is a
conditioning channel in the generator label; not present as an organ to
segment for RAovSeg, so RAovSeg excludes them). Both sets share the
same 8-subject test held-out set.

### 3.2.3 D1_MHS (Phase 2 only)

D1 is used exclusively as generator training data in Phase 2. Filter:
present T2 + present ovary + present uterus; result ~32 D1 subjects for
Phase 2 generator training. Sizes are similar to D2's 32, but the
domain gap (T2 vs T2FS) is precisely the challenge Phase 2 is designed
to test.

D1 is **not** used for downstream evaluation at any point. The 8 D2 test
subjects remain the only evaluation target throughout the dissertation.

## 3.3 The sacred 8-subject test set

Identical across every experiment: real-only baseline, all four Phase 1
downstream runs, the Phase 1 variance study (n=8 seeds), Phase 2 exp2,
and Phase 2 exp2_lam05. The list:

```
D2-005, D2-015, D2-016, D2-017, D2-023, D2-024, D2-026, D2-038
```

Two subjects (**D2-005** and **D2-023**) are universal-failure cases in
the augmented pipeline — DSC = 0 across all 8 seeds of the v3 SPADE
variance study. This alone caps the achievable mean DSC at ~0.22 even
if every other subject were segmented perfectly. Chapter 6 flags this
as a per-subject-analysis limitation of the current evaluation
protocol.

The sacred test set is never seen during generator training, generator
preprocessing, or any preprocessing calibration. It is the invariant
against which all synth-augmentation claims are measured.

## 3.4 The 6-channel label design

Labels are stored as 6-channel one-hot tensors at 512 × 512. Exactly one
channel is `1` per pixel:

| Channel | Name | Semantic meaning | Source |
|---|---|---|---|
| 0 | `outside_body` | Air outside the body silhouette | Computed: `1 − body_mask` from image |
| 1 | `uterus` | Target organ (RAovSeg-relevant even though not the segmentation target) | Manual annotation `_ut.nii.gz` |
| 2 | `ov_L` | Left ovary (image-space split from combined mask) | Auto-split from `_ov.nii.gz` |
| 3 | `ov_R` | Right ovary | Auto-split from `_ov.nii.gz` |
| 4 | `em` | Endometrioma (present only for em+ subjects) | Manual annotation `_em.nii.gz` |
| 5 | `body_other` | Body tissue that isn't a target organ (fat, muscle, bowel, bladder wall, etc.) | Computed: threshold + morphological closing + fill, minus organ channels |

### Why 6 channels — the body_other addition

Early Exp 1a runs used 5 channels: [bg, ut, ov_L, ov_R, em]. The
"background" was ambiguously "outside body" plus "body tissue that is
not a target organ." The generator handled the "clearly outside the body"
regions fine (uniformly dark) but produced **noisy grey edges** where
the ambiguous background transitioned from body tissue to air — because
the model had no explicit conditioning distinguishing "render as body
tissue" from "render as outside body."

Adding the `body_other` channel gave explicit conditioning. Every pixel
is now unambiguously assigned to one of six semantic classes, and the
generator has a clean "fill this region with plausible non-target
tissue" signal for body_other and a clean "fill this region with air"
signal for outside_body.

This is one of the mid-flight changes flagged in
`project_overview.md`: it was added during Exp 1a and then inherited
unchanged into 1b, 1c_concat, 1c_spade for ablation parity. The
architecture design docs originally described a 5-channel label; the
actual implementation is 6 channels throughout — a documentation drift
to be aware of.

### Why the em channel is present even in the augmentation setup

The generator is trained on subjects that include em-positive cases (32
D2 subjects for the generator vs 30 for RAovSeg), but RAovSeg's
inclusion criteria excludes em-positive subjects. This means:

- Generator training sees em labels (for realism — em is common in the
  UT-EndoMRI cohort).
- Downstream RAovSeg training does NOT see em-positive subjects.
- Synth subjects generated for RAovSeg augmentation come from generator
  training subjects that *don't* have em (since we generate one synth per
  RAovSeg train_val subject → we only use em-free source labels).

The em channel exists in the generator's label space but is always zero
in the synth-for-augmentation pipeline. It is neither an artefact nor a
deliberate probe of augmentation utility — just an inheritance from the
generator design.

## 3.5 Body-centered vs image-centered — the preprocessing framing

This is one of the most consequential technical decisions in the whole
dissertation, because it is the primary distribution-shift lever that
Chapter 5's v1 → v2 fixes address.

### 3.5.1 Generator preprocessing (body-centered)

`src/Generator/preprocess_for_generator.py`:

```
1. Load raw T2FS NIfTI at native spacing/resolution/orientation
2. Compute body silhouette mask (threshold + morphological closing + fill)
3. Bounding-box the body silhouette
4. Crop with 5% margin around the bbox
5. Resample the cropped region to 512 × 512, per-subject in-plane spacing
6. Save image.nii.gz + 6-channel label.nii.gz + body_silhouette.nii.gz
```

**Result**: body fills ~90% of the 512 × 512 frame; outside-body regions
are a thin border of air.

### 3.5.2 RAovSeg preprocessing (image-centered, unmodified)

`src/RaovSeg_recreation/preprocess.py`:

```
1. Load raw T2FS NIfTI at native spacing/resolution/orientation
2. Resample to 0.35 × 0.35 × 6.0 mm at 512 × 512 × native_z
   (in-plane resample only; z stays at native slicing)
3. Percentile clip to [1st, 99th], min-max normalise to [0, 1]
4. Ovary enhancement rule (see §3.6)
5. Save image.npy + ov_label.npy
```

**Result**: at 512 × 512 / 0.35 mm/px, the field of view is
179.2 mm — the body fills ~55–60% of the frame; substantial black
border around the body.

### 3.5.3 The distribution mismatch

When synth (body-centered, body 90% of frame) is fed through RAovSeg's
preprocess.py, the resample step preserves the body-centered framing
because it operates on the synth's stored spacing/origin/direction —
which the generator inherited from its body-centered preprocessing, not
from the raw subject's original frame. So the synth ends up as
body-fills-90% *even after* RAovSeg's preprocessing.

Real subjects go through the exact same preprocess.py steps and end up
as body-fills-60%.

The AttUSeg trains on a mix of body-fills-60% (real) and body-fills-90%
(synth). Extreme framing mismatch → the model cannot generalise from
"synth-framing" to "real-framing" reliably.

This is the mismatch **Fix 3 of v2** addresses: resample the synth
NIfTI to the raw source subject's frame at save time in
`assemble_synthetic_volumes.py`. Details in §5.3.2.

## 3.6 The ovary intensity enhancement rule — the pipeline's hidden assumption

Step 4 of RAovSeg preprocessing:

```python
if 0.22 <= I <= 0.30:  I = 1       # ovary highlight
elif I >= 0.5:         I = 1 - I   # invert bright tissue
else:                  I = I       # leave dark tissue
```

**What it does on real T2FS after percentile-clip + min-max**:
- The [0.22, 0.30] range corresponds empirically to the intensity of
  ovary tissue on T2FS (post-clip-normalise).
- Voxels in that range get set to 1 — visually saturated white blobs
  where ovaries are.
- The AttUSeg then trains on images where the ovary is *the brightest
  thing in the slice by construction*, learning "predict ovary at the
  bright blob."

**Why this rule matters for the augmentation story**:
- If synth's ovary intensity does *not* land in [0.22, 0.30] after the
  same percentile-clip + min-max, the enhancement does NOT fire on
  synth. The AttUSeg then trains on synth images where the ovary is
  visually indistinguishable from surrounding tissue → the segmenter
  learns nothing useful from that synth.
- Worse: if some other tissue in the synth *does* happen to land in
  [0.22, 0.30] (via rank-luck after histogram matching), enhancement
  fires in the wrong location → the segmenter is actively trained on
  incorrect signal.

The v1 → v2 → v3 preprocessing fixes (Chapter 5) are all ultimately
trying to get the synth ovary intensity into the enhancement window,
first via distribution-level histogram matching (v2) and then via
label-aware ovary-region intensity rescaling (v3 Path B). §5.3 traces
this thread.

**Meta-lesson (Chapter 6, claim 4)**: preprocessing pipeline alignment
matters more than raw image realism. FID / hist_KL do not measure
whether the synth ovary lands in the right intensity band — but that
band is the pipeline's central assumption.

## 3.7 RAovSeg pipeline in detail

Three trainable components + one postprocess block + one evaluator.

### 3.7.1 Stage 1 — ResClass (slice classifier)

`src/RaovSeg_recreation/train_resclass.py`.

| Attribute | Value |
|---|---|
| Task | Binary: "does this slice contain any ovary voxel?" |
| Backbone | MONAI `ResNetBlock`, features `[8, 16]` |
| Stem | Conv7×7 stride=2 → BN → ReLU → MaxPool3×3 stride=2 |
| Body | ResNetBlock(8→8) → ResNetBlock(8→16, stride 2) |
| Head | GAP → Dropout(0.2) → Linear(16, 1) |
| Loss | BCEWithLogitsLoss |
| Optimiser | Adam, lr=1e-3, weight_decay=1e-4 |
| Batch | 32 |
| Epochs | up to 50, save best by val F1 |
| Augmentation | RandAffine (±25° rotation, ±25 px translation, prob=1.0), 5× multiplier |
| Train/val split | 60/40 subject-level (no slice-level leakage) |
| Inference threshold | 0.6 (paper unspecified; tuned on val) |

The ResClass filter is critical: without it (no_resclass evaluation
column), full-pipeline DSC drops from 0.290 to 0.013 — a 20× degradation.
The classifier suppresses ovary-negative slices before they can produce
false-positive predictions.

**Why 60/40 subject-split at this stage**: the classifier is trained on
slices from ~18 subjects and validated on slices from ~12 subjects. The
alternative — slice-level split — leaks per-subject appearance across
train/val and inflates val F1 without generalisation. The paper does not
specify this ratio; we adopted 60/40 as a sensible default for a
per-subject 30-count pool.

### 3.7.2 Stage 2 — AttUSeg (segmentation)

`src/RaovSeg_recreation/train_attuseg.py`.

| Attribute | Value |
|---|---|
| Backbone | MONAI `AttentionUnet` |
| Channels | `[16, 32, 64, 128]` |
| Strides | `(2, 2, 2)` |
| Dropout | 0.2 |
| Loss | Focal Tversky (α=0.8, β=0.2, γ=1.33) |
| Optimiser | Adam, lr=1e-3, weight_decay=1e-4 |
| Batch | 16 |
| Epochs | 100, save best by val Dice |
| Augmentation | RandAffine (±25° rotation, ±25 px translation, prob=1.0), 5× multiplier, `mode=["bilinear", "nearest"]` |
| Train/val split | 80/20 subject-level, ovary-containing slices only |

The 80/20 subject-split gives more train subjects than ResClass's 60/40
because the class balance is much sharper (all training slices contain
ovary, by ResClass-filter). Focal Tversky (α > β) heavily penalises
false negatives, encoding the clinical bias toward "do not miss the
ovary."

### 3.7.3 Stage 3 — Postprocessing

`RAovSeg/RAovSeg_tools.py::postprocess_()` (external repo, unchanged).

Per-volume:
1. Morphological closing (`closing_iterations=10`). Fills small holes
   in the predicted mask.
2. Largest connected component. Keeps only the single largest
   3D-connected mask, discarding scattered false-positive fragments.

Postprocessing lifts DSC by ~0.05 on average (0.235 → 0.290 in the
paper). It is unmodified across our experiments.

### 3.7.4 Stage 4 — Evaluation

`src/RaovSeg_recreation/evaluate.py`. Reports three DSC variants:

| Variant | Pipeline | Purpose |
|---|---|---|
| `full` | ResClass + AttUSeg + postprocess | Headline metric |
| `no_postprocess` | ResClass + AttUSeg | Isolates postprocess contribution |
| `no_resclass` | AttUSeg on every slice | Isolates ResClass contribution |

Paper benchmarks: full = 0.290, no_postprocess = 0.235, no_resclass = 0.013.
Our real-only baseline reproduces these numbers within noise.

## 3.8 Augmentation modifications to preprocess.py

For the augmentation experiments, we added a **single flag** to
`preprocess.py`:

```
--extra-train-dir <path to synth_volumes/exp1c_variant/>
```

Behaviour when set:
- Scan the extra directory for subdirectories matching `D2-9XX` (our synth
  naming convention).
- For each match, run the same preprocessing (percentile-clip + min-max +
  ovary enhancement) as for real subjects.
- Force matching subjects into `train_val` split (skip the 30/8 shuffle for
  synth).
- Real 8-subject test set preserved.

The synth naming convention: `D2-001 → D2-900`, `D2-002 → D2-901`, ...
(one synth subject per real train subject). This lets us report
synth:real ratios exactly per experiment (e.g. Phase 1 v3 had 30 synth +
30 real = 1:1).

## 3.9 Data pipeline summary — the full chain

```
Raw NIfTI (UT-EndoMRI D2_TCPW/D2-NNN/)
   │
   ├─── Generator preprocessing (body-centered)
   │        preprocess_for_generator.py → data/processed_generator/D2/
   │        → generator training pool (32 subjects)
   │
   ├─── RAovSeg preprocessing (image-centered)
   │        preprocess.py → data/processed/
   │        → RAovSeg train_val (30) + test (8, sacred)
   │
   └─── Synth pipeline (Phase 5)
            assemble_synthetic_volumes.py
            → synth_volumes/exp1c_variant/D2-9NN/
            → preprocess.py --extra-train-dir
            → RAovSeg train_val (30 real + N synth)
            → train_resclass.py → train_attuseg.py → evaluate.py
```

Every arrow in this diagram is a potential distribution-shift injection
point. Chapter 5 traces which ones caused the observed downstream
failures and which fixes worked or did not.

## 3.10 Files and paths

**Repo-side (this working copy):**
- `data/splits/d2_generator_split.json` — generator train/test split
- `src/Generator/preprocess_for_generator.py` — body-centered preprocess
- `src/RaovSeg_recreation/preprocess.py` — image-centered preprocess with
  `--extra-train-dir`
- `src/RaovSeg_recreation/train_resclass.py`
- `src/RaovSeg_recreation/train_attuseg.py`
- `src/RaovSeg_recreation/evaluate.py`
- `RAovSeg/RAovSeg_tools.py` — external, unchanged; provides
  `postprocess_()` and `dsc_cal_np()`

**HPC-side** (see Chapter 7 for full layout):
- `/mnt/parscratch/users/ijp25lg/synth_mri/EndometriosisDataset/UT-EndoMRI/`
- `/mnt/parscratch/users/ijp25lg/synth_mri/EndometriosisDataset/data/processed/`
- `/mnt/parscratch/users/ijp25lg/synth_mri/EndometriosisDataset/data/processed_generator/D2/`
