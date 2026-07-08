#!/bin/bash
# =============================================================================
# RAovSeg augmented training: 1c_spade (target=0.22) as augmentation, seed 1
# Submit with:  sbatch scripts/run_raovseg_aug_spade_seed0.sh
# Full pipeline: preprocess (with --extra-train-dir) → train_resclass → train_attuseg → evaluate
# =============================================================================
#SBATCH --job-name=raov_aug_spade_t022_s1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/raov_aug_spade_t022_s1_%j.out
#SBATCH --error=logs/raov_aug_spade_t022_s1_%j.err
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

# --- Per-run output dir so 6 SLURM jobs don't clobber each other ---
VARIANT=spade_t022
SEED=1
OUT_BASE=/mnt/parscratch/users/$USER/synth_mri/runs/raovseg_aug_${VARIANT}_seed${SEED}
SYNTH_DIR=/mnt/parscratch/users/$USER/synth_mri/synth_volumes/exp1c_spade_t022
DATA_DIR=/mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset/UT-EndoMRI/D2_TCPW

mkdir -p $OUT_BASE/processed $OUT_BASE/models $OUT_BASE/predictions logs

echo "=== JOB $SLURM_JOB_ID on $(hostname) ==="
echo "=== variant=${VARIANT}, seed=${SEED}, started: $(date) ==="
echo "OUT_BASE=$OUT_BASE  SYNTH_DIR=$SYNTH_DIR"
nvidia-smi || true

# --- Stage 1: preprocess (real D2 + synth from 1c_spade → train_val; 8 sacred D2 → test) ---
echo ">>> [1/4] preprocess"
python src/RaovSeg_recreation/preprocess.py \
  --data-dir $DATA_DIR \
  --extra-train-dir $SYNTH_DIR \
  --output-dir $OUT_BASE/processed

# --- Stage 2: train_resclass ---
echo ">>> [2/4] train_resclass --seed $SEED"
python src/RaovSeg_recreation/train_resclass.py \
  --data-dir $OUT_BASE/processed/train_val \
  --output-dir $OUT_BASE/models \
  --seed $SEED

# --- Stage 3: train_attuseg ---
echo ">>> [3/4] train_attuseg --seed $SEED"
python src/RaovSeg_recreation/train_attuseg.py \
  --data-dir $OUT_BASE/processed/train_val \
  --output-dir $OUT_BASE/models \
  --seed $SEED

# --- Stage 4: evaluate on the 8 sacred test subjects ---
echo ">>> [4/4] evaluate"
python src/RaovSeg_recreation/evaluate.py \
  --test-dir $OUT_BASE/processed/test \
  --models-dir $OUT_BASE/models \
  --output-dir $OUT_BASE/predictions

echo "=== finished: $(date) ==="
