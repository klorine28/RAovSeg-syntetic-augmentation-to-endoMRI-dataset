#!/bin/bash
# =============================================================================
# RAovSeg augmented training: exp2_lam05 (D1→D2 λ=0.05) synth as aug, seed 2
# Submit with:  sbatch scripts/run_raovseg_aug_exp2_lam05_seed2.sh
# =============================================================================
#SBATCH --job-name=raov_aug_lam05_s2
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/raov_aug_lam05_s2_%j.out
#SBATCH --error=logs/raov_aug_lam05_s2_%j.err
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

VARIANT=exp2_lam05
SEED=2
OUT_BASE=/mnt/parscratch/users/$USER/synth_mri/runs/raovseg_aug_${VARIANT}_seed${SEED}
SYNTH_DIR=/mnt/parscratch/users/$USER/synth_mri/synth_volumes/${VARIANT}
DATA_DIR=/mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset/UT-EndoMRI/D2_TCPW

mkdir -p $OUT_BASE/processed $OUT_BASE/models $OUT_BASE/predictions logs

echo "=== JOB $SLURM_JOB_ID on $(hostname) ==="
echo "=== variant=${VARIANT}, seed=${SEED}, started: $(date) ==="
echo "OUT_BASE=$OUT_BASE  SYNTH_DIR=$SYNTH_DIR"
nvidia-smi || true

echo ">>> [1/4] preprocess"
python src/RaovSeg_recreation/preprocess.py \
  --data-dir $DATA_DIR \
  --extra-train-dir $SYNTH_DIR \
  --output-dir $OUT_BASE/processed

echo ">>> [2/4] train_resclass --seed $SEED"
python src/RaovSeg_recreation/train_resclass.py \
  --data-dir $OUT_BASE/processed/train_val \
  --output-dir $OUT_BASE/models \
  --seed $SEED

echo ">>> [3/4] train_attuseg --seed $SEED"
python src/RaovSeg_recreation/train_attuseg.py \
  --data-dir $OUT_BASE/processed/train_val \
  --output-dir $OUT_BASE/models \
  --seed $SEED

echo ">>> [4/4] evaluate"
python src/RaovSeg_recreation/evaluate.py \
  --test-dir $OUT_BASE/processed/test \
  --models-dir $OUT_BASE/models \
  --output-dir $OUT_BASE/predictions

echo "=== finished: $(date) ==="
