#!/bin/bash
# =============================================================================
# Exp 2 — RAovSeg band sweep on REAL data only.
#
# Parametric SLURM script. Submit with env vars overriding O1/O2/SEED:
#   sbatch --export=ALL,O1=0.15,O2=0.25,SEED=0 scripts/run_raovseg_band_sweep.sh
#
# Or via the submit helper which fans out 5 bands x 3 seeds:
#   python scripts/submit_band_sweep.py
#
# Bands intended (see SIDE_EXPERIMENT_DESIGN.md §Exp 2):
#   [0.22, 0.30]  published control (identical to Exp 0a — you already have this)
#   [0.15, 0.25]  below the published band
#   [0.30, 0.42]  rising edge toward the ovary median (0.474)
#   [0.42, 0.56]  centred on the ovary median
#   none          enhancement disabled entirely (floor; use SKIP_ENH=1)
# =============================================================================
#SBATCH --job-name=raov_band_sweep
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/raov_band_%x_%j.out
#SBATCH --error=logs/raov_band_%x_%j.err
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

# --- Params (override via --export=ALL,O1=...,O2=...,SEED=...,SKIP_ENH=1) ---
O1="${O1:-0.22}"
O2="${O2:-0.30}"
SEED="${SEED:-0}"
SKIP_ENH="${SKIP_ENH:-0}"          # 1 = disable enhancement entirely

# Variant tag — encodes the condition in filesystem paths so 15 sweep runs
# never clobber each other.
if [[ "$SKIP_ENH" == "1" ]]; then
  VARIANT="band_noenh"
else
  # Rendered as e.g. band_o22_o30 -> band_o022_o030 so lexicographic sort works
  O1_TAG=$(printf "%03d" "$(python -c "print(int(round(${O1}*100)))")")
  O2_TAG=$(printf "%03d" "$(python -c "print(int(round(${O2}*100)))")")
  VARIANT="band_o${O1_TAG}_o${O2_TAG}"
fi

OUT_BASE=/mnt/parscratch/users/$USER/synth_mri/runs/raovseg_${VARIANT}_seed${SEED}
DATA_DIR=/mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset/UT-EndoMRI/D2_TCPW

mkdir -p $OUT_BASE/processed $OUT_BASE/models $OUT_BASE/predictions logs

echo "=== JOB $SLURM_JOB_ID on $(hostname) ==="
echo "=== variant=${VARIANT}, o1=${O1}, o2=${O2}, seed=${SEED}, skip_enh=${SKIP_ENH} ==="
echo "=== started: $(date) ==="
echo "OUT_BASE=$OUT_BASE"
nvidia-smi || true

# --- Stage 1: preprocess (real D2 only; band-swept enhancement) ---
echo ">>> [1/4] preprocess (o1=${O1}, o2=${O2}, skip_enh=${SKIP_ENH})"
PREPROC_ARGS=(--data-dir "$DATA_DIR" --output-dir "$OUT_BASE/processed")
if [[ "$SKIP_ENH" == "1" ]]; then
  # Skip enhancement for every subject (prefix "D2-" matches all real subject IDs).
  PREPROC_ARGS+=(--skip-enhancement-for-prefix "D2-")
else
  PREPROC_ARGS+=(--o1 "$O1" --o2 "$O2")
fi
python src/RaovSeg_recreation/preprocess.py "${PREPROC_ARGS[@]}"

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
