# Stanage Cheatsheet — Synth MRI Project
 
Username: `ijp25lg`
Project folder: `/mnt/parscratch/users/ijp25lg/synth_mri`
Dataset root: `/mnt/parscratch/users/ijp25lg/synth_mri/EndometriosisDataset`
Conda env: `synth_mri` (lives in `/mnt/parscratch/users/ijp25lg/anaconda/.envs/synth_mri`)
 
---
 
## 1. Connect
 
From your **local terminal**:
 
```bash
ssh ijp25lg@stanage.shef.ac.uk
```
 
Off-campus: connect to the University SSL VPN first. You'll get a Duo prompt — type `1` for push.
 
You land on `login1` or `login2`. **Login nodes are not for compute** — only editing, submitting jobs, light file ops.
 
---
 
## 2. Navigate to the project
 
```bash
cd /mnt/parscratch/users/$USER/synth_mri
```
 
---
 
## 3. Daily activation routine
 
Every time you start work, you need ALL THREE of these — the cuDNN module is critical, without it torch will fail with a `GLIBC_2.27 not found` error:
 
```bash
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1
source activate synth_mri
```
 
You'll see `(synth_mri)` in your prompt when active.
 
---
 
## 4. Get an interactive session
 
CPU-only (file ops, light tasks):
 
```bash
srun --pty --time=01:00:00 --mem=8G --cpus-per-task=2 bash
```
 
GPU (debugging, short tests — NOT for real training):
 
```bash
# A100 (80 GB) — default choice
srun --partition=gpu --qos=gpu --gres=gpu:1 \
     --mem=82G --cpus-per-task=4 --time=02:00:00 --pty bash
 
# H100 (80 GB) — newer, faster
srun --partition=gpu-h100 --qos=gpu --gres=gpu:1 \
     --mem=82G --cpus-per-task=4 --time=02:00:00 --pty bash
 
# H100 NVL (94 GB) — most GPU memory, Intel CPU host
srun --partition=gpu-h100-nvl --qos=gpu --gres=gpu:1 \
     --mem=96G --cpus-per-task=4 --time=02:00:00 --pty bash
```
 
Wait for the prompt to change from `login2` to a node name. Then run the activation routine in §3.
 
When done: `exit` to leave the worker node.
 
---
 
## 5. Full "I just logged in" workflow
 
```bash
ssh ijp25lg@stanage.shef.ac.uk
 
# Get a GPU
srun --partition=gpu --qos=gpu --gres=gpu:1 \
     --mem=82G --cpus-per-task=4 --time=02:00:00 --pty bash
 
# Activate environment (all three lines!)
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1
source activate synth_mri
 
# Sanity check the GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
 
# Go to project
cd /mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset
```
 
---
 
## 6. File transfer (run from LOCAL terminal, not Stanage)
 
Upload:
 
```bash
rsync -avhP /local/path/to/file.zip ijp25lg@stanage.shef.ac.uk:/mnt/parscratch/users/ijp25lg/synth_mri/
```
 
Download:
 
```bash
rsync -avhP ijp25lg@stanage.shef.ac.uk:/mnt/parscratch/users/ijp25lg/synth_mri/results/ ./local_results/
```
 
Re-run the same command to resume a partial transfer.
 
---
 
## 7. SLURM batch jobs (real training)
 
For anything longer than ~1 hour or unattended runs, use `sbatch`. Template:
 
```bash
#!/bin/bash
#SBATCH --job-name=synth_mri_train
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-user=YOUR_EMAIL@sheffield.ac.uk
#SBATCH --mail-type=END,FAIL
 
set -euo pipefail
 
module purge
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1
source activate synth_mri
 
cd /mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset
 
python src/train.py --config configs/exp1c.yaml
```
 
Submit and monitor:
 
```bash
mkdir -p logs                  # one-time: ensure log dir exists
sbatch train.sh                # submit → prints job ID
squeue --me                    # see your queued/running jobs
scancel <jobid>                # cancel
seff <jobid>                   # post-mortem efficiency report
sacct -u $USER --starttime=today
```
 
---
 
## 8. ONE-TIME setup steps (already done — for reference)
 
If you ever need to recreate the env from scratch:
 
### a) Configure conda to use parscratch
 
```bash
mkdir -p /mnt/parscratch/users/$USER/anaconda/.pkg-cache/
mkdir -p /mnt/parscratch/users/$USER/anaconda/.envs
 
cat > ~/.condarc << EOF
pkgs_dirs:
  - /mnt/parscratch/users/$USER/anaconda/.pkg-cache/
envs_dirs:
  - /mnt/parscratch/users/$USER/anaconda/.envs
EOF
```
 
### b) Create the env (in an interactive session)
 
```bash
srun --pty --time=01:00:00 --mem=8G --cpus-per-task=2 bash
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1
conda create -n synth_mri python=3.11 -y
source activate synth_mri
 
# Scientific stack from conda-forge (avoids GCC build errors)
conda install -c conda-forge "numpy>=1.24,<2.3" "scipy>=1.11" "matplotlib>=3.7" -y
 
# PyTorch via pip with cu121 wheel (NOT conda — conda CUDA libs need newer glibc)
pip install torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu121
 
# MONAI + dependencies
pip install "monai>=1.3" "SimpleITK>=2.3"
pip install monai-generative
 
exit  # leave the worker node
```
 
### c) Why pip not conda for torch
 
The Stanage worker nodes still run an older OS (CentOS 7, glibc 2.17). Conda's `pytorch-cuda` package ships CUDA 12 libraries that need glibc ≥ 2.27, so they fail to load with `GLIBC_2.27 not found`. The pip cu121 wheels bundle their own CUDA libraries that are compatible with the older system, AND the `cuDNN/8.9.2.26-CUDA-12.1.1` module provides the right system-level cuDNN. This is the pattern Sheffield's official PyTorch docs recommend.
 
---
 
## 9. Common gotchas (specific to this setup)
 
- **Forget `module load cuDNN/...` → torch fails to import** with `GLIBC_2.27 not found`. Always all three modules.
- **Login node vs worker node**: prompt shows `login2` = no compute. `node001`, `node101`, etc. = worker, do work here.
- **Two login nodes (login1, login2)**: connecting to `stanage.shef.ac.uk` lands you on either at random. For tmux, pick one and SSH directly: `ssh ijp25lg@stanage-login1.shef.ac.uk`.
- **Home dir is 50 GB** — keep big stuff on `/mnt/parscratch/users/$USER/`.
- **Nothing on Stanage is backed up** — push code to GitHub, push results to Google Drive / X-Drive when they matter.
- **Lustre hates millions of tiny files** — keep slice data in NIfTI volumes, not per-slice PNGs.
- **GPU interactive sessions are for debugging only** — real training goes through `sbatch`.
- **Don't `pip install` numpy/scipy directly on Stanage** — system GCC is too old (4.8.5) to build them. Use conda-forge.
---
 
## 10. Useful one-liners
 
```bash
quota -u -s                              # home directory usage
du -sh /mnt/parscratch/users/$USER/*     # parscratch usage breakdown
squeue --me                              # my running/queued jobs
sinfo -p gpu                             # A100 partition status
sinfo -p gpu-h100                        # H100 partition status
seff <jobid>                             # post-mortem CPU/RAM efficiency
sacct -u $USER --starttime=today         # today's job history
nvidia-smi                               # GPU status (only on GPU nodes)
module avail <name>                      # is software X available?
module list                              # what's currently loaded
```
 
---
 
## 11. Sanity-check commands
 
Run these any time you suspect something's broken:
 
```bash
# Activation worked?
which python              # → .../synth_mri/bin/python
python --version          # → 3.11.x
 
# Torch + CUDA on GPU node?
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
 
# MONAI Generative imports?
python -c "from generative.networks.nets import DiffusionModelUNet; print('OK')"
 
# SimpleITK?
python -c "import SimpleITK as sitk; print(sitk.Version_VersionString())"
```

---

## 12. Inference sweeps on a trained checkpoint

For tuning guidance scale and DDIM step count on an existing EMA checkpoint
without retraining. Uses `src/Generator/inference_validate.py`, which picks
high-foreground labels from the train split so the conditioning is stressed.

### Path conventions

```
runs/<exp_name>/
  ckpt/step_NNNNNN.pt        # checkpoint (EMA weights are inside)
  samples/                    # in-training periodic sample grids
  samples/<sweep_name>/       # ← put inference sweeps here
  config_used.yaml            # the config that produced the run
  tb/                         # TensorBoard logs
```

Run names currently on disk: `exp1a`, `exp1b` (body-centered v2),
`exp1a_v1_imgcentered`, `exp1b_v1_imgcentered`, `exp1b_bad_init`,
`exp1b_gamma_only`.

### Tier 1 sweep — guidance scale + DDIM steps (interactive GPU)

After §5's full "I just logged in" routine (GPU + activate + cd into the repo):

```bash
EXP=exp1b                                                   # or exp1a
CKPT=/mnt/parscratch/users/$USER/synth_mri/runs/$EXP/ckpt/step_080000.pt
OUT=/mnt/parscratch/users/$USER/synth_mri/runs/$EXP/samples/tier1_tuning
mkdir -p "$OUT"
ls -lh "$CKPT"

# Guidance sweep at 50 DDIM steps
# (note: as of June 2026 the YAML default is g=2.0 / s=100 — see exp1b.yaml.
#  passing --guidance-scale and --num-inference-steps explicitly overrides
#  the YAML so the sweep grid spans the same range regardless of default.)
for G in 1.5 2.0 5.0 7.5; do
  python -m src.Generator.inference_validate \
    --config src/Generator/${EXP}.yaml \
    --ckpt   "$CKPT" \
    --out    "$OUT/g${G}_s50.png" \
    --guidance-scale "$G" \
    --num-inference-steps 50
done

# DDIM-steps sweep at guidance 3.0
for S in 100 250; do
  python -m src.Generator.inference_validate \
    --config src/Generator/${EXP}.yaml \
    --ckpt   "$CKPT" \
    --out    "$OUT/g3.0_s${S}.png" \
    --guidance-scale 3.0 \
    --num-inference-steps "$S"
done

# Optional cross-cell at the lower-guidance + more-steps combo
python -m src.Generator.inference_validate \
  --config src/Generator/${EXP}.yaml \
  --ckpt   "$CKPT" \
  --out    "$OUT/g2.0_s100.png" \
  --guidance-scale 2.0 \
  --num-inference-steps 100

ls -lh "$OUT/"
```

Each grid takes ~1 min at `s50`, ~2 min at `s100`, ~5 min at `s250` on an
A100. Whole sweep fits inside a 1 h interactive session.

### Pull results to local

From your **local terminal**, not Stanage:

```bash
EXP=exp1b
rsync -avhzP \
  ijp25lg@stanage.shef.ac.uk:/mnt/parscratch/users/ijp25lg/synth_mri/runs/$EXP/samples/tier1_tuning/ \
  ./${EXP}_v2_bodycentered/tier1_tuning/
```

### Flags supported by `inference_validate.py`

| Flag | Default | Purpose |
|---|---|---|
| `--config` | (required) | Training YAML, e.g. `src/Generator/exp1b.yaml` |
| `--ckpt`   | (required) | Checkpoint .pt path |
| `--out`    | (required) | Output PNG path |
| `--n`      | 4 | Number of high-foreground labels to sample |
| `--guidance-scale` | from YAML | CFG scale; 1.0 disables, 3-5 typical |
| `--num-inference-steps` | from YAML | DDIM steps; more is smoother but slower |
| `--no-ema` | off | Use training weights instead of EMA |

### Notes

- `inference_validate.py` picks labels by highest target-organ voxel count,
  so the grids look anatomically richer than the random fixed-labels grids
  produced during training. The `g3.0_s50` cell of any sweep is the
  apples-to-apples comparison to the in-training final sample grid.
- The `samples/tier1_tuning/` subdirectory is a convention, not enforced.
  Use any name — the script just writes to whatever path `--out` specifies.
- For ablation parity, if a tuning combo materially helps one variant, run
  the same sweep on the other variant before drawing comparison conclusions.

---

## 13. Explainability runs (heatmaps per sample)

`src/Generator/explain.py` produces one multi-panel PNG per sample combining
five XAI views: deepest-layer activation, per-channel GradientSHAP,
counterfactual label ablation, per-timestep snapshots, and SPADE γ maps
(1b only — the panel is absent for 1a). Same CLI shape as
`inference_validate.py`.

### Cost per sample

| Step | Sampling runs | Wall-clock @ 100 DDIM steps |
|---|---|---|
| (1) attn + main synth | 1 | ~10 s |
| (2) GradientSHAP | 0 (single-t forwards) | ~2 s |
| (3) SPADE γ | 1 | ~10 s |
| (4) counterfactual (full + 4 ablated) | 5 | ~50 s |
| (5) per-timestep snapshots | 1 | ~10 s |
| **Total per sample** | **8** | **~80 s** |

For the default `--n 4`, budget ~6 min on an A100. Pass `--skip-counterfactual`
to drop the 5-run row if you're iterating.

### Push the new file (one-time)

`explain.py` is new — sync it (and the updated `inference_validate.py` /
YAMLs) from your **local terminal**:

```bash
rsync -avhzP --exclude '__pycache__' \
  src/Generator/ \
  ijp25lg@stanage.shef.ac.uk:/mnt/parscratch/users/ijp25lg/synth_mri/EndometriosisDataset/src/Generator/
```

### Run on Stanage (interactive)

After §5's full activation routine:

```bash
EXP=exp1b                                                       # or exp1a
CKPT=/mnt/parscratch/users/$USER/synth_mri/runs/$EXP/ckpt/step_080000.pt
OUT=/mnt/parscratch/users/$USER/synth_mri/runs/$EXP/explain
mkdir -p "$OUT"

python -m src.Generator.explain \
  --config src/Generator/${EXP}.yaml \
  --ckpt   "$CKPT" \
  --out_dir "$OUT"
```

Picks up `guidance_scale` and `num_inference_steps` from the YAML defaults
(now g=2.0 / s=100 after Tier 1). Override either with `--guidance-scale`
or `--num-inference-steps` if you want to inspect a different inference
configuration.

### Pull results to local

```bash
EXP=exp1b
rsync -avhzP \
  ijp25lg@stanage.shef.ac.uk:/mnt/parscratch/users/ijp25lg/synth_mri/runs/$EXP/explain/ \
  ./${EXP}_v2_bodycentered/explain/
```

### Flags supported by `explain.py`

| Flag | Default | Purpose |
|---|---|---|
| `--config` | (required) | Training YAML |
| `--ckpt` | (required) | Checkpoint .pt path |
| `--out_dir` | (required) | Directory for per-sample PNGs |
| `--n` | 4 | Number of high-foreground samples |
| `--guidance-scale` | from YAML | CFG scale at inference |
| `--num-inference-steps` | from YAML | DDIM steps |
| `--gradshap-t` | 500 | Mid-range timestep used for GradientSHAP target |
| `--gradshap-samples` | 10 | Interpolation samples for GradientSHAP |
| `--n-snapshots` | 6 | Per-timestep snapshots through the denoising chain |
| `--no-ema` | off | Use training weights instead of EMA |
| `--skip-counterfactual` | off | Skip the 4 ablation runs |

### How to read the figure

A five-row (1b) / four-row (1a) grid, 6 panels per row:

| Row | What it shows | How to read it |
|---|---|---|
| 1 | real, label, synth, overlay, deep-layer attention | First sanity row. Attention heatmap = where the bottleneck features have the most signal magnitude, averaged across timesteps. |
| 2 | GradientSHAP per label channel | Brighter = that label pixel had more influence on the predicted noise. Empty channels (e.g. em where no endometrioma exists) should be dark. |
| 3 | Counterfactual: full + each organ ablated + (−ut vs full) diff | Each ablated panel removes one organ from the conditioning. The diff (blue/red bwr) shows what changed when uterus was removed. Strong diff in organ region = model is actually using that channel. |
| 4 | Per-timestep snapshots from pure noise → clean image | The story of how the image is built: silhouette first, organs later, texture last. |
| 5 (1b only) | \|γ\| per SPADE module | Per-decoder-level modulation magnitude. Bright = SPADE thinks the label conditioning matters strongly at that pixel for that layer. Empty rows = SPADE has not learned to use that level. |

For comparing 1a vs 1b: line up `explain/sample_00.png` from each side by
side. Rows 1–4 are directly comparable. Row 5 exists for 1b only — that
asymmetry is itself a signal about what SPADE provides that concat does not.