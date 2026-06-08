# Experiment 1a — 2D DDPM with concat conditioning

The Phase 1 baseline. A 2D conditional DDPM where the 5-channel label map is
concatenated with the noisy image as the U-Net's input. No SPADE, no PatchGAN,
no adversarial loss. The number this produces is what 1b (SPADE) and 1c
(SPADE+PatchGAN) must beat to justify their components in the ablation.

## Layout (drops into the existing repo)

```
EndometriosisDataset/
    src/
        preprocess.py             # existing RAovSeg recreation (unchanged)
        train_attuseg.py          # existing
        ...
        Generator/                # NEW subpackage for synthetic-data work
            __init__.py
            build_generator_split.py
            preprocess_for_generator.py
            dataset.py
            model.py
            train.py
            smoke_test.py
    configs/
        exp1a.yaml                # NEW
    scripts/
        preprocess_generator.sh   # NEW — SLURM
        train_exp1a.sh            # NEW — SLURM
```

The `Generator/` namespace cleanly separates synthetic-data experiments from
the RAovSeg recreation. Imports are `python -m src.Generator.train`.

## Pipeline order

```
   UT-EndoMRI/D2_TCPW/D2-XXX/D2-XXX_T2FS.nii.gz, _ut, _ov, _em
            │
            ├──► src/preprocess.py (existing, unchanged)
            │    enhancement applied, single-channel ovary
            │    → data/processed/{train_val,test}/<subj>/{image,ov_label}.npy
            │    → data/processed/manifest.csv  ← SOURCE OF TRUTH for splits
            │
            └──► src/Generator/build_generator_split.py
                 reads manifest.csv
                 → data/splits/d2_generator_split.json   (RAovSeg train ∪ endo subjects)
                 │
                 └──► src/Generator/preprocess_for_generator.py
                      no enhancement, 5-channel label, L/R ovary split via CC + midline fallback
                      → data/processed_generator/D2/<subj>/image_T2FS.nii.gz
                      → data/processed_generator/D2/<subj>/label_T2FS.nii.gz   (4D vector for dataloader)
                      → data/processed_generator/D2/<subj>/label_T2FS_{bg,uterus,ov_L,ov_R,em}.nii.gz   (per-class for QA)
                      │
                      └──► src/Generator/train.py
                           Exp 1a: concat-conditioned DDPM
                           → runs/exp1a/{ckpt, samples, tb}/...
```

## Key design decisions (why things look the way they do)

**Spacing matches RAovSeg recreation:** `(0.35, 0.35, 6.0)` mm. Synthetic and
real volumes share spacing when RAovSeg later ingests synthetic data, so no
resampling step is needed at that handoff.

**No ovary intensity enhancement in this pipeline.** Enhancement is destructive
and RAovSeg-specific. Generators learn in plain [0,1] space; enhancement is
applied identically to real and synthetic at the moment RAovSeg consumes them
(downstream, in Week 5). This is enforced by construction: this pipeline never
applies it.

**5-channel labels even though RAovSeg only does binary ovary segmentation.**
The generator (especially 1b/1c with SPADE) needs spatial structure for all
anatomical classes. Using 5 channels here keeps the same training contract
across 1a/1b/1c so the ablation isolates conditioning mechanism alone.

**L/R ovary split via 3D connected components, with midline fallback.**
UT-EndoMRI ships a single combined `_ov.nii.gz`. We split it via 3D CCs:
sort by mean x-centroid, leftmost → L-channel, rightmost → R-channel. If
only one CC exists (one ovary absent, or both fused via thin connectivity),
fall back to pure x = W/2 midline split for that subject. Tiny extra
components (< 5% of largest) are merged into the nearest of the two by
3D centroid distance. Per-subject decisions are logged in
`preprocess_summary.json`. Look for `"lr_decision": "midline_fallback"` —
those subjects are worth eyeballing in Slicer to confirm the split is sane.

**Generator train set = RAovSeg's 30 ∪ all 11 endo subjects, minus test.**
RAovSeg's 30 train_val subjects all have `has_em=0`, so channel 4 of their
label tensor is empty. Adding the endo subjects (D2-008, 009, 040, 041, 054,
070, 071, 072, 077, 078, 079) gives the generator examples for the
endometrioma channel. The 8 RAovSeg test subjects are sacred — never trained
on, never overlapped with, even if some are in the endo list (they get
dropped from train if so, logged in the split JSON's `_meta`).

## Running

### 1. Preprocess (one-time)

```bash
ssh ijp25lg@stanage.shef.ac.uk
cd /mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset
mkdir -p logs

# Set your email in scripts/preprocess_generator.sh first, then:
sbatch scripts/preprocess_generator.sh
squeue --me
```

The job runs `build_generator_split.py` then `preprocess_for_generator.py`.
~20 minutes wall time. Outputs:

- `data/splits/d2_generator_split.json` — confirm `_meta.n_train` looks
  right (~36-37 expected: 30 RAovSeg + 11 endo, minus any in test set)
- `data/processed_generator/D2/<subj>/` — one folder per subject
- `data/processed_generator/D2/preprocess_summary.json` — per-subject decisions

**Sanity check after preprocessing.** Open one or two subjects in ITK-SNAP:

```bash
# (locally, after rsync)
rsync -avhP ijp25lg@stanage.shef.ac.uk:/mnt/parscratch/users/ijp25lg/synth_mri/EndometriosisDataset/data/processed_generator/D2/D2-001/ ./check/
```

Check:
- `image_T2FS.nii.gz` looks like a normal T2 fat-sat scan, NOT enhanced
- `label_T2FS_uterus.nii.gz` overlays correctly on the uterus
- `label_T2FS_ov_L.nii.gz` and `label_T2FS_ov_R.nii.gz` are spatially distinct
  and on the correct sides
- For subjects logged with `lr_decision: midline_fallback` in the summary,
  manually verify the split looks sensible

### 2. Smoke test (~5 min on interactive GPU)

```bash
srun --partition=gpu --qos=gpu --gres=gpu:1 \
     --mem=82G --cpus-per-task=4 --time=01:00:00 --pty bash
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1
set +u; source activate synth_mri; set -u
cd /mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset
python -m src.Generator.smoke_test --config configs/exp1a.yaml
```

Expected output:
- batch shapes: image (8, 1, 512, 512), label (8, 5, 512, 512)
- ~80M U-Net params
- L_diff dropping over 20 steps
- "OK — pipeline is wired correctly."

### 3. Full training

```bash
# Set email in scripts/train_exp1a.sh first
sbatch scripts/train_exp1a.sh
squeue --me
```

A100 80GB, ~12-18h to 80k steps at batch 8. 24h walltime gives headroom.

### 4. Monitoring

```bash
# locally
rsync -avhP ijp25lg@stanage.shef.ac.uk:/mnt/parscratch/users/ijp25lg/synth_mri/runs/exp1a/ ./local_runs/exp1a/
tensorboard --logdir ./local_runs/exp1a/tb
```

Sample grids at `runs/exp1a/samples/step_XXXXXX.png` are the fastest sanity
check. By ~30k steps recognisable pelvic anatomy. By 80k it should look like
real T2FS, with ovaries/uterus where the input label said.

## Things to watch

- **OOM at batch 8**: drop to batch 4 in `configs/exp1a.yaml`.
- **Loss plateaus high (>0.6)**: usually a label/image alignment bug. Open
  a sample grid early — if labels and image content are spatially divorced,
  the resampling reference frame got wrong. The preprocessor now uses
  `SetReferenceImage` for labels which should prevent this, but verify.
- **L/R channels mostly empty**: connected-components might be too
  conservative. Check `preprocess_summary.json` — if many subjects show
  `"lr_decision": "no_ovary"`, the resampled mask sums are zero, which
  means the label resampling went wrong. If many show `"midline_fallback"`,
  the structure-element choice in `_split_ovary_lr` may need to switch
  from 6- to 18-connectivity.
- **Mode collapse on uterus, missing ovaries**: increase
  `ovary_oversample_weight` from 3.0 to 5.0.
- **Loss fine but samples are noise**: scheduler mismatch between training
  (DDPM) and sampling (DDIM). Both must use identical beta_schedule,
  beta_start, beta_end, num_train_timesteps. They do here, but if you fork
  for 1b/1c, double-check.

## What's NOT in this drop (Week 2 follow-up)

- **Inference / synthetic NIfTI assembly script** — turning the trained
  checkpoint into a pool of synthetic volumes with augmentation (elastic
  deformation, rotation, flips with channel swap, ovary scaling).
- **Quality metrics** — FID, boundary DSC, NN-LPIPS, intensity histogram.

These come after the model is trained and you've eyeballed the sample grids.
Want to wire those up, ping me once the training run is in the queue.
