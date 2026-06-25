#!/bin/bash
# =============================================================================
# Exp 1b: 2D DDPM + SPADE conditioning
# Submit with:  sbatch scripts/train_exp1b.sh
# =============================================================================
#SBATCH --job-name=exp1b_spade
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/exp1b_%j.out
#SBATCH --error=logs/exp1b_%j.err
#SBATCH --mail-user=lgardunoroqueni1@sheffield.ac.uk
#SBATCH --mail-type=END,FAIL

set -euo pipefail
export SLURM_EXPORT_ENV=ALL

module purge
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Conda's MKL activation script reads unset vars; relax `set -u` for activation only
set +u
source activate synth_mri
set -u

cd /mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset

echo "=== JOB $SLURM_JOB_ID on $(hostname) ==="
echo "=== started: $(date) ==="
nvidia-smi
git rev-parse HEAD || echo "(not a git repo)"
echo "=== running ==="

python -m src.Generator.train --config src/Generator/exp1b.yaml

echo "=== finished: $(date) ==="
