#!/bin/bash
# =============================================================================
# lam05 synth + skip enhancement (Option C). Seed 1.
# =============================================================================
#SBATCH --job-name=raov_lam05C_s1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/raov_lam05C_s1_%j.out
#SBATCH --error=logs/raov_lam05C_s1_%j.err
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

VARIANT=exp2_lam05_pathC
SEED=1
OUT_BASE=/mnt/parscratch/users/$USER/synth_mri/runs/raovseg_aug_${VARIANT}_seed${SEED}
SYNTH_DIR=/mnt/parscratch/users/$USER/synth_mri/synth_volumes/exp2_lam05
DATA_DIR=/mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset/UT-EndoMRI/D2_TCPW

mkdir -p $OUT_BASE/processed $OUT_BASE/models $OUT_BASE/predictions logs

echo "=== JOB $SLURM_JOB_ID on $(hostname) ==="
echo "=== variant=${VARIANT}, seed=${SEED}, started: $(date) ==="
nvidia-smi || true

python src/RaovSeg_recreation/preprocess.py \
  --data-dir $DATA_DIR --extra-train-dir $SYNTH_DIR \
  --output-dir $OUT_BASE/processed \
  --skip-enhancement-for-prefix D2-9

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

echo "=== finished: $(date) ==="
