# Appendix — Reproducibility

> **Word count: uncounted** (appendices are excluded from the
> 9,000–14,000 word range). Contains code repository pointer, HPC
> layout, YAML config schema, SLURM invocation patterns, and
> per-experiment reproduction recipes. Referenced by Chapter 3
> (Methodology) for implementation detail.
>
> Source: repo structure, `../docs_archive/stanage_cheatsheet.md`,
> per-experiment SLURM scripts. The Research Diary and Reflection
> appendix lives in `08_research_diary.md` (separate 1,000-word budget).

---

## 7.1 Code repository

**GitHub URL**: *[placeholder — insert public URL when repo is published]*

**Repository name**: `EndometriosisDataset` (working copy at
`/Users/lorenzogarduno/Documents/EndometriosisDataset/`).

**Contents summary**:

| Path | Contents |
|---|---|
| `src/Generator/` | Diffusion training + assembly + explainability code |
| `src/RaovSeg_recreation/` | RAovSeg recreation (preprocess + ResClass + AttUSeg + evaluate) |
| `RAovSeg/` | External RAovSeg repo (postprocess and DSC utilities, unchanged) |
| `scripts/` | SLURM scripts (one per experiment/seed) |
| `data/splits/` | Deterministic train/test splits |
| `data/processed_generator/` | Body-centered preprocessed generator inputs (NOT versioned; regenerable) |
| `data/processed/` | Image-centered RAovSeg preprocessed inputs (NOT versioned; regenerable) |
| `1a/`, `1b/`, `1c/` | Per-variant training outputs (checkpoints, samples, quality.json, radiologist_review) |
| `dissertation_docs/` | This dissertation's 7 master docs |
| `../metrics/master_metrics.csv` | Aggregate Phase 1 quality metrics |
| `../metrics/variance_study_summary.json` | n=8 seed summary for v3 SPADE |
| `../metrics/exp2_dsc_summary.json` | n=3 seed summary for exp2 |
| `requirements.txt` | Python dependencies |

**Files intentionally excluded from the public repo**:
- Raw UT-EndoMRI NIfTI volumes (large, dataset-license-controlled).
- Trained checkpoints (large, ~1–2 GB each; available on request from
  the corresponding author).
- Per-run RAovSeg predictions (regenerable in ~35 min per run).

## 7.2 HPC layout (Sheffield Stanage)

```
/mnt/parscratch/users/ijp25lg/synth_mri/
├── EndometriosisDataset/                       # git clone of the repo
│   ├── UT-EndoMRI/D1_MHS/                     # raw D1 subjects (51 volumes)
│   ├── UT-EndoMRI/D2_TCPW/                    # raw D2 subjects (~73 volumes)
│   ├── data/
│   │   ├── processed/                          # RAovSeg-preprocessed (30 real + N synth per run)
│   │   ├── processed_generator/D2/             # body-centered generator inputs
│   │   ├── processed_generator/D1/             # (Phase 2 only) D1 generator inputs
│   │   └── splits/
│   │       ├── d2_generator_split.json         # 32 train + 8 test
│   │       └── raovseg_split.json              # 30 train_val + 8 test
│   ├── src/Generator/                          # diffusion training code
│   ├── src/RaovSeg_recreation/                 # recreation pipeline
│   ├── RAovSeg/                                # external repo (postprocess + DSC)
│   ├── scripts/                                # all SLURM scripts
│   └── logs/                                   # SLURM stdout/stderr per job
│
├── runs/                                        # all training outputs
│   ├── exp1a/                                  # 1a checkpoints, samples, quality.json
│   ├── exp1b/                                  # 1b (v1_first is historical failure)
│   ├── exp1c_concat/                           # 1c concat + PatchGAN
│   ├── exp1c_spade/                            # 1c SPADE + PatchGAN
│   ├── exp2_d1_gen_d2_disc/                    # Phase 2 exp2
│   ├── exp2_lam05/                             # Phase 2 λ=0.05 diagnostic
│   ├── raovseg_aug_concat_seed{0,1,2}/         # v1 concat augmentation
│   ├── raovseg_aug_spade_seed{0..7}/           # v3 SPADE, seeds 0-2 initial + 3-7 for variance study
│   ├── raovseg_aug_spade_t022_seed{0,1,2}/     # Option B, t=0.22
│   ├── raovseg_aug_spade_t028_seed{0,1,2}/     # Option B, t=0.28
│   ├── raovseg_aug_spade_pathC_seed{0,1,2}/    # Option C, skip enhancement
│   ├── raovseg_aug_exp2_seed{0,1,2}/           # Phase 2 exp2 downstream
│   └── raovseg_aug_exp2_lam05_seed{0,1,2}/     # Phase 2 exp2_lam05 downstream
│
└── synth_volumes/                               # generator outputs → downstream input
    ├── exp1c_concat/D2-9NN/{T2FS,ov}.nii.gz    # 30 synth volumes per variant
    ├── exp1c_spade/D2-9NN/{T2FS,ov}.nii.gz     # v3 (with all fixes) baseline SPADE
    ├── exp1c_spade_t022/D2-9NN/                # Option B, t=0.22 variant
    ├── exp1c_spade_t028/D2-9NN/                # Option B, t=0.28 variant
    ├── exp2/D2-9NN/{T2FS,ov}.nii.gz            # Phase 2 exp2 synth
    └── exp2_lam05/D2-9NN/{T2FS,ov}.nii.gz      # Phase 2 exp2_lam05 synth
```

Each `runs/raovseg_aug_*_seed*/` contains its own isolated
`{processed/, models/, predictions/}` so parallel jobs don't clobber
each other.

## 7.3 Software environment

**Base environment**: Conda + pip.

**Python**: 3.10.

**Key dependencies** (`requirements.txt` full list):

| Package | Version | Purpose |
|---|---|---|
| torch | 2.0+ | Deep learning framework |
| monai | 1.3+ | Medical imaging framework, `DiffusionModelUNet`, `AttentionUnet`, `ResNetBlock` |
| monai-generative | latest | Diffusion model components |
| numpy | 1.24+ | Array operations |
| SimpleITK | 2.2+ | NIfTI I/O + resampling |
| scipy | 1.10+ | Morphological ops, connected components |
| scikit-image | 0.20+ | Body silhouette computation |
| matplotlib | 3.7+ | Sample grid rendering |
| lpips | 0.1+ | LPIPS-NN metric |
| pytorch-fid | 0.3+ | FID metric |
| tensorboard | 2.13+ | Training logs |
| PyYAML | 6.0+ | Config files |

**HPC modules loaded** (Sheffield Stanage):
```
module load Anaconda3/2024.06-1
source activate synth_mri  # local conda env with the pip-installed deps
```

**A100 GPU allocation**: 1× A100 80GB, 4 CPUs, 82G RAM per training job.
Assembly jobs same but reduced walltime.

## 7.4 Full data-pipeline reproduction

Steps to go from raw UT-EndoMRI to Phase 1 v3 SPADE downstream DSC.

### 7.4.1 Preprocessing (once)

**Generator-side (body-centered)**:
```bash
python -m src.Generator.preprocess_for_generator \
  --data-dir UT-EndoMRI/D2_TCPW \
  --out-dir data/processed_generator/D2 \
  --frame-margin 0.05

python -m src.Generator.build_generator_split \
  --preprocessed-dir data/processed_generator/D2 \
  --raovseg-split data/splits/raovseg_split.json \
  --out data/splits/d2_generator_split.json
```

**RAovSeg-side (image-centered)** — real only, no synth yet:
```bash
python src/RaovSeg_recreation/preprocess.py \
  --data-dir UT-EndoMRI/D2_TCPW \
  --output-dir data/processed \
  --split-seed 42
```

Result: `data/processed/{train_val, test, excluded}/D2-NNN/{image,
ov_label}.npy`.

### 7.4.2 Real-only baseline (RAovSeg)

```bash
sbatch scripts/train_raovseg_baseline_seed0.sh   # ResClass + AttUSeg + eval, seed 0
sbatch scripts/train_raovseg_baseline_seed1.sh
sbatch scripts/train_raovseg_baseline_seed2.sh
```

Each internally:
```bash
python src/RaovSeg_recreation/train_resclass.py \
  --data-dir data/processed/train_val --output-dir models/ --seed $SEED

python src/RaovSeg_recreation/train_attuseg.py \
  --data-dir data/processed/train_val --output-dir models/ --seed $SEED

python src/RaovSeg_recreation/evaluate.py \
  --test-dir data/processed/test --models-dir models/ --output-dir predictions/
```

Expected result: DSC (full) ≈ 0.290.

### 7.4.3 Phase 1 generator training (four variants)

```bash
sbatch scripts/train_1a.sh                # concat, 80k steps
sbatch scripts/train_1b.sh                # SPADE, 80k steps (needs zero-init)
sbatch scripts/train_exp1c_concat.sh      # concat + PatchGAN, 100k steps
sbatch scripts/train_exp1c_spade.sh       # SPADE + PatchGAN, 100k steps
```

Each internally invokes:
```bash
python -m src.Generator.train \
  --config src/Generator/{exp_name}.yaml \
  --output-dir runs/{exp_name}/ \
  --resume-if-exists
```

Config files: `src/Generator/exp1a.yaml`, `exp1b.yaml`,
`exp1c_concat.yaml`, `exp1c_spade.yaml`. Each specifies the backbone,
conditioning, PatchGAN block, λ schedule, and training steps.

Runtime per experiment: ~10 h (80k steps) or ~11 h (100k steps).

### 7.4.4 Phase 1 quality metric computation

```bash
for V in exp1a exp1b exp1c_concat exp1c_spade; do
  python -m src.Generator.quality_metrics \
    --config src/Generator/$V.yaml \
    --ckpt   runs/$V/ckpt/step_{final}.pt \
    --out    runs/$V/quality.json \
    --n-samples 256

  python -m src.Generator.explain \
    --config src/Generator/$V.yaml \
    --ckpt   runs/$V/ckpt/step_{final}.pt \
    --out    runs/$V/explain/ \
    --n-samples 4
done

python -m src.Generator.aggregate_metrics \
  --runs runs/exp1a runs/exp1b runs/exp1c_concat runs/exp1c_spade \
  --out master_metrics.csv
```

Result: `../metrics/master_metrics.csv` with the columns in §5.2.2.

### 7.4.5 Phase 1 v3 synth assembly

For the v3 configuration (all three preprocessing fixes ON + Path B
label-aware rescale at t=0.26):

```bash
sbatch scripts/assemble_synth_1c_spade.sh          # v3 SPADE @ t=0.26
sbatch scripts/assemble_synth_1c_spade_t022.sh     # Option B variant
sbatch scripts/assemble_synth_1c_spade_t028.sh     # Option B variant
```

Direct invocation (per variant):
```bash
python -m src.Generator.assemble_synthetic_volumes \
  --config src/Generator/exp1c_spade.yaml \
  --ckpt   /mnt/parscratch/users/$USER/synth_mri/runs/exp1c_spade/ckpt/step_100000.pt \
  --gen-preprocessed-root /mnt/.../data/processed_generator/D2 \
  --gen-split-file        /mnt/.../data/splits/d2_generator_split.json \
  --raw-data-root         /mnt/.../UT-EndoMRI/D2_TCPW \
  --out-dir               /mnt/.../synth_volumes/exp1c_spade \
  --iscs-alpha 0.8 \
  --noise-seed 0 \
  --ovary-target-intensity 0.26          # Path B, v3 default
  # implicit: body-silhouette-mask ON, histogram-match ON, resample-to-source ON
```

Runtime: ~4 h at SLURM walltime, ~5–7 min per volume × 30 volumes.

### 7.4.6 Phase 1 v3 downstream training

Seeds 0–2 (initial) and 3–7 (variance study):
```bash
for S in 0 1 2 3 4 5 6 7; do
  sbatch scripts/run_raovseg_aug_spade_seed${S}.sh
done
```

Each internally:
```bash
OUT_BASE=/mnt/.../runs/raovseg_aug_spade_seed${SEED}

python src/RaovSeg_recreation/preprocess.py \
  --data-dir UT-EndoMRI/D2_TCPW \
  --extra-train-dir /mnt/.../synth_volumes/exp1c_spade \
  --output-dir $OUT_BASE/processed

python src/RaovSeg_recreation/train_resclass.py \
  --data-dir $OUT_BASE/processed/train_val \
  --output-dir $OUT_BASE/models --seed $SEED

python src/RaovSeg_recreation/train_attuseg.py \
  --data-dir $OUT_BASE/processed/train_val \
  --output-dir $OUT_BASE/models --seed $SEED

python src/RaovSeg_recreation/evaluate.py \
  --test-dir $OUT_BASE/processed/test \
  --models-dir $OUT_BASE/models \
  --output-dir $OUT_BASE/predictions
```

Runtime per seed: ~35 min. n=8 seeds total → ~5 h of aggregate wall time.

### 7.4.7 Phase 2 exp2 pipeline

```bash
sbatch scripts/train_exp2.sh                    # generator training, ~11 h
sbatch scripts/assemble_synth_exp2.sh           # synth assembly, ~4 h
for S in 0 1 2; do
  sbatch scripts/run_raovseg_aug_exp2_seed${S}.sh
done
```

exp2_lam05 pipeline (parallel structure):
```bash
sbatch scripts/train_exp2_lam05.sh
sbatch scripts/assemble_synth_exp2_lam05.sh
for S in 0 1 2; do
  sbatch scripts/run_raovseg_aug_exp2_lam05_seed${S}.sh
done
```

### 7.4.8 DSC extraction (for reporting)

`evaluate.py` prints DSC to stdout, no JSON. To collect across seeds:

```bash
for V in concat spade exp2 exp2_lam05; do
  for S in 0 1 2 3 4 5 6 7; do
    LATEST=$(ls -t logs/raov_aug_${V}_s${S}_*.out 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
      grep -E "^\s+full\s*:" "$LATEST"
    fi
  done
done
```

Aggregate n=8 SPADE:
```bash
python scripts/summarise_variance_study.py \
  --logs logs/raov_aug_spade_s{0..7}_*.out \
  --out variance_study_summary.json
```

Aggregate n=3 exp2:
```bash
python scripts/summarise_exp2_dsc.py \
  --logs logs/raov_aug_exp2_s{0..2}_*.out \
  --out exp2_dsc_summary.json
```

## 7.5 SLURM script inventory

Located in `scripts/`. All follow the same header pattern (1× A100,
4 CPUs, 82G RAM, appropriate walltime).

### 7.5.1 Generator training

| Script | Experiment | Walltime |
|---|---|---|
| `train_1a.sh` | Exp 1a — concat | 10 h |
| `train_1b.sh` | Exp 1b — SPADE | 10 h |
| `train_exp1c_concat.sh` | Exp 1c concat + PatchGAN | 12 h |
| `train_exp1c_spade.sh` | Exp 1c SPADE + PatchGAN | 12 h |
| `train_exp2.sh` | Phase 2 — cross-domain λ=0.01 | 12 h |
| `train_exp2_lam05.sh` | Phase 2 — cross-domain λ=0.05 | 12 h |

### 7.5.2 Synth assembly

| Script | Config | Notes |
|---|---|---|
| `assemble_synth_1c_concat.sh` | 1c concat, v3 pipeline | 4 h |
| `assemble_synth_1c_spade.sh` | 1c SPADE, v3 pipeline (t=0.26) | 4 h |
| `assemble_synth_1c_spade_t022.sh` | Option B, t=0.22 | 4 h |
| `assemble_synth_1c_spade_t028.sh` | Option B, t=0.28 | 4 h |
| `assemble_synth_exp2.sh` | Phase 2 exp2 | 4 h |
| `assemble_synth_exp2_lam05.sh` | Phase 2 exp2_lam05 | 4 h |

### 7.5.3 Downstream RAovSeg augmentation

One per (variant, seed) combination:

| Script pattern | Seeds | Notes |
|---|---|---|
| `run_raovseg_aug_concat_seed{0,1,2}.sh` | 0-2 | v1 concat |
| `run_raovseg_aug_spade_seed{0..7}.sh` | 0-7 | v3 SPADE + variance study |
| `run_raovseg_aug_spade_t022_seed{0,1,2}.sh` | 0-2 | Option B t=0.22 |
| `run_raovseg_aug_spade_t028_seed{0,1,2}.sh` | 0-2 | Option B t=0.28 |
| `run_raovseg_aug_spade_pathC_seed{0,1,2}.sh` | 0-2 | Option C skip enhancement |
| `run_raovseg_aug_exp2_seed{0,1,2}.sh` | 0-2 | Phase 2 exp2 |
| `run_raovseg_aug_exp2_lam05_seed{0,1,2}.sh` | 0-2 | Phase 2 exp2_lam05 |

Walltime per script: 8 h allocated, ~35 min actual.

## 7.6 Configuration files

Located in `src/Generator/`. YAML format.

### 7.6.1 Fields common to all Phase 1 configs

```yaml
seed: 42
data:
  preprocessed_root: /mnt/.../data/processed_generator/D2
  split_file: /mnt/.../data/splits/d2_generator_split.json
  ovary_weight_multiplier: 3.0    # weighted sampler boost for ovary slices

backbone:
  resolution: 512
  levels: 4
  channels: [64, 128, 256, 256]
  self_attention_levels: [3]      # deepest level only (64x64)
  label_channels: 6

diffusion:
  T: 1000
  beta_start: 1e-4
  beta_end: 2e-2
  loss: mse

cfg:
  train_dropout_prob: 0.1
  inference_guidance: 3.0         # per-variant: 3.0 for concat, 2.0 for SPADE

ema:
  decay: 0.9999

training:
  optimizer: adamw
  lr: 1e-4
  batch_size: 4
  total_steps: 80000              # or 100000 for 1c
  checkpoint_interval: 5000
  sample_interval: 5000

inference:
  sampler: ddim
  steps: 100
```

### 7.6.2 Concat-specific block (1a, 1c_concat)

```yaml
conditioning:
  mode: concat                    # 7-channel input
```

### 7.6.3 SPADE-specific block (1b, 1c_spade)

```yaml
conditioning:
  mode: spade
  spade:
    hidden_channels: 64
    modules_at_levels: [1, 2, 3]  # decoder ResBlocks at these levels
    zero_init: true               # critical — see §4.4.2
```

### 7.6.4 PatchGAN block (1c only)

```yaml
discriminator:
  arch: patchgan
  base_channels: 64
  n_layers: 5
  spectral_norm: true
  lr: 2.5e-5

lambda_schedule:
  warmup_end: 10000
  ramp_end: 30000
  peak: 0.01                      # or 0.05 for exp2_lam05
```

### 7.6.5 Phase 2 dual-dataloader block (exp2, exp2_lam05)

```yaml
data:
  # generator training pool
  preprocessed_root: /mnt/.../data/processed_generator/D1
  split_file: /mnt/.../data/splits/d1_generator_split.json

  # discriminator anchor pool
  disc_preprocessed_root: /mnt/.../data/processed_generator/D2
  disc_split_file:        /mnt/.../data/splits/d2_generator_split.json

discriminator:
  unconditional: true             # label zeroed at D input
```

## 7.7 Assembly-time preprocessing fix flags

The `assemble_synthetic_volumes.py` script controls the v1/v2/v3 fixes
via CLI flags. All are ON by default in v3 config.

| Flag | Default | Purpose |
|---|---|---|
| `--body-mask` | ON (`--no-body-mask` disables) | Fix 1: apply outside-body silhouette mask |
| `--histogram-match` | ON (`--no-histogram-match` disables) | Fix 2: rank-based intensity match to source real subject |
| `--resample-to-source` | ON (`--no-resample-to-source` disables) | Fix 3: transfer synth to source real subject's spacing/origin/direction |
| `--ovary-target-intensity` | 0.26 | Path B (v3): additive offset to ovary region to hit RAovSeg enhancement window |
| `--iscs-alpha` | 0.8 | ISCS shared-noise weight |
| `--noise-seed` | 0 | Reproducibility |

To reproduce v1 (no fixes): `--no-body-mask --no-histogram-match
--no-resample-to-source --ovary-target-intensity 0` (0 disables the
rescale).

To reproduce v2 (3 fixes, no Path B): default flags + `--ovary-target-
intensity 0`.

To reproduce v3 (recommended baseline): default flags (all ON) +
`--ovary-target-intensity 0.26`.

## 7.8 Notes on external artefacts

**Datasets**: UT-EndoMRI is published by Liang et al. (2025). Access
governed by the paper's data-sharing terms. Not included in this
repository. Cite:

> Liang et al. (2025). *Deep learning for automated segmentation of the
> ovaries on endometriosis MRI: an open dataset and methodology*.
> Scientific Data, 12:XXX. [Full citation to be inserted from
> repository / bibtex when finalised.]

**RAovSeg external code**: `RAovSeg/RAovSeg_tools.py` is imported from
the paper's associated public repo. Unchanged from upstream. Provides:

- `postprocess_(mask, closing_iterations)` — morphological closing +
  largest connected component.
- `dsc_cal_np(pred, gt)` — DSC calculation.

**Compute**: Sheffield Stanage HPC ARC. A100 80GB GPUs, SLURM
scheduler. Not part of the repository but essential for reproduction
at the training step counts we use (~40 GPU-hours per Phase 1 variant).
Smaller machines (e.g. RTX 3090 with reduced batch) may reproduce with
adjusted hyperparameters (batch × step counts).

## 7.9 Known deviations from published RAovSeg

Our recreation reproduces the paper's DSC 0.290 baseline but with a few
implementation-side choices not fully specified in the paper:

| Aspect | Paper | Our recreation |
|---|---|---|
| ResClass inference threshold | Unspecified | 0.6 (val-tuned) |
| Morphological closing iterations | Unspecified | 10 |
| ResClass train/val split ratio | Unspecified | 60/40 subject-level |
| AttUSeg train/val split ratio | Unspecified | 80/20 subject-level |
| SPLIT_SEED for 30/8 test split | Unspecified | 42 (deterministic across all our runs) |
| Focal Tversky (α, β, γ) | (0.8, 0.2, 4/3) | (0.8, 0.2, 1.33) |

The DSC 0.290 real-only baseline reproduces despite these
under-specifications, confirming our recreation is faithful.

## 7.10 Quick reference — reproduce X

| To reproduce | Read | Then run |
|---|---|---|
| Phase 1 quality table (§5.2.2) | §7.4.3 + §7.4.4 | Four `train_*.sh` + four `quality_metrics.py` + `aggregate_metrics.py` |
| v3 SPADE downstream DSC 0.178 (§5.4.1) | §7.4.5 + §7.4.6 | `assemble_synth_1c_spade.sh` + eight `run_raovseg_aug_spade_seed*.sh` |
| Option B t=0.22 or t=0.28 | §7.4.5 + §7.4.6 | `assemble_synth_1c_spade_t022.sh` (or _t028) + three seeds |
| Phase 2 exp2 downstream DSC 0.020 (§5.5.2) | §7.4.7 | `train_exp2.sh` + `assemble_synth_exp2.sh` + three seeds |
| exp2_lam05 (pending) | §7.4.7 | `train_exp2_lam05.sh` + `assemble_synth_exp2_lam05.sh` + three seeds |
| Real-only baseline 0.290 | §7.4.2 | Three baseline seeds |

Any step above cross-references its dependencies. If a run fails,
`logs/{jobname}_{jobid}.out` has the SLURM stdout for the failing job.

## 7.11 Mechanism figures — ovary-voxel intensity table

Companion table to the three mechanism figures
(`figures/fig_mech_overlay.png`, `figures/fig_mech_body_hist.png`,
`figures/fig_mech_ovary_hist.png`). Renders the ovary-voxel intensity
distribution after RAovSeg's percentile-clip + minmax normalisation
(no enhancement), for three real D2 test subjects and three synthetic
subjects assembled with the Phase 1 v3 1c SPADE generator. RAovSeg's
enhancement window is `[0.22, 0.30]`; the Path B rescaling target used
at assembly time is `t = 0.26`.

Rendered by
`python -m src.RaovSeg_recreation.mechanism_figures --real-dir UT-EndoMRI/D2_TCPW --real-subjects D2-016 D2-017 D2-024 --synth-dir exp1c_spade_samples --synth-subjects D2-900 D2-901 D2-902 --real-label "Real D2" --synth-label "Synth (1c SPADE)" --out-dir figures`.
Raw values are in `figures/mech_ovary_intensity_table.csv`.

| Variant | n vox | mean | median | p10 | p90 | % in [0.22, 0.30] |
|---|---:|---:|---:|---:|---:|---:|
| **Real D2 (pooled)** | 37,505 | **0.499** | 0.462 | 0.269 | 0.802 | **10.1%** |
| real / D2-016 | 16,035 | 0.480 | 0.446 | 0.281 | 0.735 | 11.1% |
| real / D2-017 | 11,415 | 0.490 | 0.465 | 0.295 | 0.718 | 8.3% |
| real / D2-024 | 10,055 | 0.541 | 0.503 | 0.209 | 1.000 | 10.5% |
| **Synth 1c SPADE (pooled)** | 54,342 | **0.203** | 0.169 | 0.058 | 0.397 | **15.1%** |
| synth / D2-900 | 5,179 | 0.207 | 0.186 | 0.102 | 0.345 | 20.7% |
| synth / D2-901 | 15,884 | 0.195 | 0.161 | 0.077 | 0.363 | 14.5% |
| synth / D2-902 | 33,279 | 0.206 | 0.169 | 0.049 | 0.415 | 14.5% |

**Reading the table.** The enhancement window `[0.22, 0.30]` was designed
around the intensity band a T2FS ovary occupies after RAovSeg's
normalisation, and it is what triggers the segmenter's ovary-detection
stage. Two facts stand out:

1. **Real D2 ovaries mostly land *above* the window.** Pooled mean 0.499
   with only 10.1% of voxels inside `[0.22, 0.30]`. This is a property of
   the D2 cohort's intensity statistics after percentile-clip
   normalisation, not of the synth, and it means even the real-only
   RAovSeg baseline is operating with the enhancement stage under-firing.

2. **Synth ovaries land *below* the window.** Pooled mean 0.203, with
   15.1% in-window. The Path B rescale at assembly time targeted t = 0.26,
   but the additive per-volume offset did not close the gap: after the
   per-subject histogram-match into the raw real subject's intensity
   range, the ovary rescale was fighting the global distribution shape
   and lost.

The two distributions bracket the enhancement window from opposite sides
(see `fig_mech_ovary_hist.png`). This is the mechanism behind the
Phase 1 downstream degradation: at no point in the pipeline does the
labelled ovary tissue occupy the same intensity band as the real
ovaries, so the augmentation trains the segmenter on a fundamentally
different intensity prior than the one it will see at test time.
