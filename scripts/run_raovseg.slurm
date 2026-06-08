#!/bin/bash
#SBATCH --job-name=raovseg
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=82G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
# #SBATCH --partition=gpu-h100      # use this (and remove --partition=gpu above) for H100
# #SBATCH --partition=gpu-h100-nvl  # use this for H100 NVL (and bump --mem to 96G)
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=lorenzo@garduno.mx

# Run the full RAovSeg baseline pipeline (Exp 0a) on Sheffield Stanage:
#   preprocess -> train_resclass -> train_attuseg -> evaluate
# Submit from /mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset:
#   mkdir -p logs   # one-time, before first submission
#   sbatch scripts/run_raovseg.slurm

set -euo pipefail
export SLURM_EXPORT_ENV=ALL

# --- Module + conda env activation (Stanage pattern) ---
# Disable -u during activation: conda's MKL/CUDA activate.d scripts reference
# unset env vars (e.g. MKL_INTERFACE_LAYER) which would otherwise abort under
# `set -u`. Re-enable nounset once the env is live.
set +u
module purge
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1
source activate synth_mri
set -u

# --- Paths ---
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
mkdir -p logs models data/processed data/predictions

# --- Runtime info ---
echo "Job: $SLURM_JOB_ID  Host: $(hostname)  Date: $(date -Iseconds)"
echo "Project: $PROJECT_ROOT"
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"

# --- Stage 1: preprocess (CPU-bound) ---
echo ">>> [1/4] preprocess"
python src/preprocess.py
echo "=== Manifest summary ==="
column -t -s, data/processed/manifest.csv | head -50 || cat data/processed/manifest.csv | head -50

# --- Stage 2: ResClass slice classifier ---
echo ">>> [2/4] train_resclass"
python src/train_resclass.py

# --- Stage 3: AttUSeg segmentation ---
echo ">>> [3/4] train_attuseg"
python src/train_attuseg.py

# --- Stage 4: evaluation on test split ---
echo ">>> [4/4] evaluate"
python src/evaluate.py

echo "Done: $(date -Iseconds)"
