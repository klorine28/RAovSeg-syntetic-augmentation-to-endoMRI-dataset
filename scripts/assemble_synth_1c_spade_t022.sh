#!/bin/bash
# Option B: SPADE synth with ovary target intensity = 0.22 (lower edge of enhancement window)
#SBATCH --job-name=synth_spade_t022
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/synth_spade_t022_%j.out
#SBATCH --error=logs/synth_spade_t022_%j.err

set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export MKL_THREADING_LAYER=GNU

module purge
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1
set +u; source activate synth_mri; set -u

cd /mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset
echo "=== JOB $SLURM_JOB_ID on $(hostname) ==="
echo "=== started: $(date) ==="
nvidia-smi || true

RUNS=/mnt/parscratch/users/$USER/synth_mri/runs
GEN_ROOT=/mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset/data/processed_generator/D2
GEN_SPLIT=/mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset/data/splits/d2_generator_split.json
RAW_DATA=/mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset/UT-EndoMRI/D2_TCPW
SYNTH_DIR=/mnt/parscratch/users/$USER/synth_mri/synth_volumes/exp1c_spade_t022

rm -rf $SYNTH_DIR
mkdir -p $SYNTH_DIR

python -m src.Generator.assemble_synthetic_volumes \
  --config src/Generator/exp1c_spade.yaml \
  --ckpt   $RUNS/exp1c_spade/ckpt/step_100000.pt \
  --gen-preprocessed-root $GEN_ROOT \
  --gen-split-file $GEN_SPLIT \
  --raw-data-dir $RAW_DATA \
  --out-dir $SYNTH_DIR \
  --iscs-alpha 0.8 \
  --noise-seed 0 \
  --ovary-target-intensity 0.22

echo "=== finished: $(date) ==="
