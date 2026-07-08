#!/bin/bash
# =============================================================================
# Assemble synthetic NIfTI volumes from exp2 (cross-domain D1→D2).
# Conditioning: D1 preprocessed masks. Raw reference frame: D1_MHS raw T2.
# Output: named D2-9XX (RAovSeg-compatible naming). Style intended as T2FS
# but note exp2 synth quality was poor — see RAOVSEG_AUGMENTATION_EXPERIMENT.md
# §8h (to be added). This assembly is Track 1 of the recommendation: run it
# so we have the actual DSC number to cite.
# Submit with:  sbatch scripts/assemble_synth_exp2.sh
# =============================================================================
#SBATCH --job-name=synth_exp2
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/synth_exp2_%j.out
#SBATCH --error=logs/synth_exp2_%j.err
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

RUNS=/mnt/parscratch/users/$USER/synth_mri/runs
GEN_ROOT=/mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset/data/processed_generator/D1
GEN_SPLIT=/mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset/data/splits/d1_generator_split.json
RAW_DATA=/mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset/UT-EndoMRI/D1_MHS
SYNTH_DIR=/mnt/parscratch/users/$USER/synth_mri/synth_volumes/exp2

rm -rf $SYNTH_DIR
mkdir -p $SYNTH_DIR

python -m src.Generator.assemble_synthetic_volumes \
  --config src/Generator/exp2.yaml \
  --ckpt   $RUNS/exp2_d1_gen_d2_disc/ckpt/step_100000.pt \
  --gen-preprocessed-root $GEN_ROOT \
  --gen-split-file $GEN_SPLIT \
  --raw-data-dir $RAW_DATA \
  --out-dir $SYNTH_DIR \
  --iscs-alpha 0.8 \
  --noise-seed 0 \
  --ovary-target-intensity 0.26

echo "=== finished: $(date) ==="
ls -lh $SYNTH_DIR/ | head -20
