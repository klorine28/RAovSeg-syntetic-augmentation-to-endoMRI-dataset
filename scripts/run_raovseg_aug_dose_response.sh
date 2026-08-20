#!/bin/bash
# =============================================================================
# Exp 1 — RAovSeg dose-response of synth-ovary target intensity.
#
# Parametric SLURM script. Consumes a pre-built retargeted synth volume dir
# (produced by scripts/retarget_ovary_intensity.py) and runs the full
# preprocess -> train_resclass -> train_attuseg -> evaluate pipeline against
# it with the standard 30-real + 30-synth augmentation.
#
# Submit with env vars:
#   sbatch --export=ALL,T_TAG=005,SEED=0 scripts/run_raovseg_aug_dose_response.sh
#
# Or via the fan-out helper (6 t-values x 3 seeds = 18 jobs):
#   python scripts/submit_dose_response.py
#
# Assumes the retargeted synth dir already exists at
#   /mnt/parscratch/users/$USER/synth_mri/synth_volumes/exp1c_spade_fixed_t${T_TAG}
# The submit helper (recommended) creates it before submitting.
# =============================================================================
#SBATCH --job-name=raov_dose
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/raov_dose_%x_%j.out
#SBATCH --error=logs/raov_dose_%x_%j.err
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

# --- Params (override via --export=ALL,T_TAG=...,SEED=...) ---
T_TAG="${T_TAG:?T_TAG env var required (e.g. 005 for t=0.05, 026 for t=0.26)}"
SEED="${SEED:-0}"

VARIANT="spade_dose_t${T_TAG}"
OUT_BASE=/mnt/parscratch/users/$USER/synth_mri/runs/raovseg_aug_${VARIANT}_seed${SEED}
SYNTH_DIR=/mnt/parscratch/users/$USER/synth_mri/synth_volumes/exp1c_spade_fixed_t${T_TAG}
DATA_DIR=/mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset/UT-EndoMRI/D2_TCPW

if [[ ! -d "$SYNTH_DIR" ]]; then
  echo "ERROR: retargeted synth dir not found: $SYNTH_DIR" >&2
  echo "Build it first, e.g.:" >&2
  echo "  python scripts/retarget_ovary_intensity.py \\" >&2
  echo "      --src-dir  /mnt/parscratch/users/\$USER/synth_mri/synth_volumes/exp1c_spade_fixed \\" >&2
  echo "      --out-dir  $SYNTH_DIR \\" >&2
  echo "      --target-normalized 0.${T_TAG}" >&2
  exit 1
fi

mkdir -p $OUT_BASE/processed $OUT_BASE/models $OUT_BASE/predictions logs

echo "=== JOB $SLURM_JOB_ID on $(hostname) ==="
echo "=== variant=${VARIANT}, T_TAG=${T_TAG}, seed=${SEED} ==="
echo "=== started: $(date) ==="
echo "OUT_BASE=$OUT_BASE  SYNTH_DIR=$SYNTH_DIR"
nvidia-smi || true

# --- Stage 1: preprocess (real D2 + retargeted synth → train_val; 8 sacred D2 → test) ---
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
