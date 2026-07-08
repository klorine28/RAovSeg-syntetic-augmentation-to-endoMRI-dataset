#!/bin/bash
# =============================================================================
# Exp 2 (Phase 2): cross-domain conditional DDPM
#   Generator sees D1_MHS T2 (non-fat-suppressed, larger anatomy pool)
#   Discriminator sees D2_TCPW T2FS (fat-suppressed target domain)
#   D is unconditional to avoid label-distribution shortcut between cohorts.
# Submit with:  sbatch scripts/train_exp2.sh
# =============================================================================
#SBATCH --job-name=exp2_xdomain
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=4
#SBATCH --time=36:00:00
#SBATCH --output=logs/exp2_xdomain_%j.out
#SBATCH --error=logs/exp2_xdomain_%j.err
#SBATCH --mail-user=lgardunoroqueni1@sheffield.ac.uk
#SBATCH --mail-type=END,FAIL

set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export MKL_THREADING_LAYER=GNU

module purge
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1

set +u
source activate synth_mri
set -u

cd /mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset

echo "=== JOB $SLURM_JOB_ID on $(hostname) ==="
echo "=== started: $(date) ==="
nvidia-smi || true
git rev-parse HEAD || echo "(not a git repo)"
echo "=== running ==="

python -m src.Generator.train --config src/Generator/exp2.yaml

echo "=== finished: $(date) ==="
