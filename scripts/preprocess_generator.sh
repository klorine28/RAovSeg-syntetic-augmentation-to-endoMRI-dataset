#!/bin/bash
# =============================================================================
# Generator data preprocessing job.
#
# Builds:
#   1. Generator split JSON  (data/splits/d2_generator_split.json)
#   2. Preprocessed D2 tree  (data/processed_generator/D2/)
#
# CPU-only, ~30s per subject × ~37 subjects ≈ 20 min wall time.
# Submit with:  sbatch scripts/preprocess_generator.sh
# =============================================================================
#SBATCH --job-name=gen_preprocess
#SBATCH --partition=sheffield
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=logs/gen_preprocess_%j.out
#SBATCH --error=logs/gen_preprocess_%j.err
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

PROJ_ROOT=/mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset
RAW_ROOT=$PROJ_ROOT/UT-EndoMRI/D2_TCPW
MANIFEST=$PROJ_ROOT/data/processed/manifest.csv
SPLIT_FILE=$PROJ_ROOT/data/splits/d2_generator_split.json
OUT_ROOT=$PROJ_ROOT/data/processed_generator/D2

echo "=== JOB $SLURM_JOB_ID on $(hostname) ==="
echo "=== started: $(date) ==="
echo "  raw_root: $RAW_ROOT"
echo "  manifest: $MANIFEST"
echo "  out:      $OUT_ROOT"

# Step 1: build generator split from RAovSeg manifest
mkdir -p "$(dirname "$SPLIT_FILE")"
python -m src.Generator.build_generator_split \
    --manifest "$MANIFEST" \
    --raw_root "$RAW_ROOT" \
    --out_file "$SPLIT_FILE"

# Step 2: preprocess all subjects in the split
python -m src.Generator.preprocess_for_generator \
    --raw_root "$RAW_ROOT" \
    --out_root "$OUT_ROOT" \
    --split_file "$SPLIT_FILE" \
    --manifest "$MANIFEST"

echo "=== finished: $(date) ==="
