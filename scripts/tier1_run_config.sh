#!/bin/bash
# =============================================================================
# Tier 1 sweep — single (config, seed) run.
#
# Submitted by scripts/tier1_sweep_coordinator.py via sbatch with these env vars:
#   TRIAL_ID              — Optuna trial number
#   SEED                  — training seed (0/1/2)
#   OVARY_TARGET          — --ovary-target-intensity (float in [0.20, 0.32])
#   ISCS_ALPHA            — --iscs-alpha (float in [0.0, 1.0])
#   ISCS_LOWPASS_SIGMA    — --iscs-lowpass-sigma (float in [0.0, 8.0])
#   Z_SMOOTH_SIGMA        — --z-smooth-sigma (float in [0.0, 1.5])
#   BODY_MASK             — 1 to keep the body silhouette mask, 0 to disable
#   HIST_MATCH            — 1 to keep histogram matching, 0 to disable
#   SKIP_ENH              — 1 to skip RAovSeg enhancement for D2-9 synth, 0 to keep
#
# Everything else is hard-coded to the Tier 1 baseline (1c_spade checkpoint).
# Edit paths in the CONFIG block for your Stanage layout.
# =============================================================================
#SBATCH --job-name=tier1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --output=logs/tier1_%j.out
#SBATCH --error=logs/tier1_%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export MKL_THREADING_LAYER=GNU

# The coordinator submits with --export=ALL so SLURM inherits the login
# shell's env. That env has synth_mri already "active" (CONDA_PREFIX,
# CONDA_SHLVL, PATH). Conda's `source activate` then thinks the env is
# already active and no-ops, leaving `python` pointing at the wrong
# interpreter. Wipe the inherited conda state so activation below starts
# clean.
unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_PYTHON_EXE CONDA_SHLVL \
      CONDA_PROMPT_MODIFIER CONDA_EXE PYTHONHOME PYTHONPATH

module purge
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1

set +u
source activate synth_mri
set -u

cd /mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset

# ---------- CONFIG (edit for your layout) ----------
CKPT_PATH=/mnt/parscratch/users/$USER/synth_mri/runs/exp1c_spade/ckpt/step_100000.pt
GEN_CFG=src/Generator/exp1c_spade.yaml
GEN_ROOT=/mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset/data/processed_generator/D2
GEN_SPLIT=/mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset/data/splits/d2_generator_split.json
RAW_DATA=/mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset/UT-EndoMRI/D2_TCPW
DATA_DIR=$RAW_DATA
ISCS_PROFILE=/mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset/metrics/real_iscs_profile.json

SWEEP_ROOT=/mnt/parscratch/users/$USER/synth_mri/sweep/tier1
OUT_BASE=${SWEEP_ROOT}/trial_${TRIAL_ID}/seed_${SEED}
SYNTH_DIR=${OUT_BASE}/synth
mkdir -p $OUT_BASE/processed $OUT_BASE/models $OUT_BASE/predictions_ov $OUT_BASE/predictions_ut logs
mkdir -p $SYNTH_DIR

echo "=== JOB $SLURM_JOB_ID on $(hostname) ==="
echo "=== trial=$TRIAL_ID seed=$SEED  $(date) ==="
echo "params: ovary_target=$OVARY_TARGET iscs_alpha=$ISCS_ALPHA iscs_lowpass=$ISCS_LOWPASS_SIGMA z_smooth=$Z_SMOOTH_SIGMA"
echo "flags:  body_mask=$BODY_MASK hist_match=$HIST_MATCH skip_enh=$SKIP_ENH"
nvidia-smi || true

# ---------- 1) Assemble synth ----------
ASSEMBLE_FLAGS=(
  --config "$GEN_CFG"
  --ckpt   "$CKPT_PATH"
  --gen-preprocessed-root "$GEN_ROOT"
  --gen-split-file "$GEN_SPLIT"
  --raw-data-dir "$RAW_DATA"
  --out-dir "$SYNTH_DIR"
  --ovary-target-intensity "$OVARY_TARGET"
  --iscs-alpha "$ISCS_ALPHA"
  --iscs-lowpass-sigma "$ISCS_LOWPASS_SIGMA"
  --z-smooth-sigma "$Z_SMOOTH_SIGMA"
  --noise-seed "$SEED"
)
[ "$BODY_MASK" = "0" ] && ASSEMBLE_FLAGS+=( --no-body-mask )
[ "$HIST_MATCH" = "0" ] && ASSEMBLE_FLAGS+=( --no-histogram-match )

echo ">>> [1/6] assemble"
python -m src.Generator.assemble_synthetic_volumes "${ASSEMBLE_FLAGS[@]}"

# ---------- 2) Score synth against real ISCS profile (diagnostic) ----------
echo ">>> [2/6] score synth ISCS"
python -m src.analysis.score_synth_iscs \
  --synth-dir "$SYNTH_DIR" \
  --profile "$ISCS_PROFILE" \
  --out "$OUT_BASE/iscs_score.json" \
  --label "trial_${TRIAL_ID}_seed_${SEED}"

# ---------- 2b) Score synth vs real intensity histogram (cheap KL diagnostic) ----------
echo ">>> [2b/6] score synth vs real hist_kl"
python -m src.analysis.hist_kl_synth_vs_real \
  --synth-dir "$SYNTH_DIR" \
  --real-dir "$RAW_DATA" \
  --real-split-file "$GEN_SPLIT" \
  --real-split-key train \
  --sequence T2FS \
  --out-json "$OUT_BASE/hist_kl.json" \
  --label "trial_${TRIAL_ID}_seed_${SEED}" || \
    echo "  (hist_kl failed, continuing — non-fatal for the trial)"

# ---------- 3) Preprocess (real + synth into train_val, 8 sacred as test) ----------
PREPROCESS_FLAGS=(
  --data-dir "$DATA_DIR"
  --extra-train-dir "$SYNTH_DIR"
  --output-dir "$OUT_BASE/processed"
)
[ "$SKIP_ENH" = "1" ] && PREPROCESS_FLAGS+=( --skip-enhancement-for-prefix D2-9 )

echo ">>> [3/6] preprocess"
python src/RaovSeg_recreation/preprocess.py "${PREPROCESS_FLAGS[@]}"

# ---------- 4) Train both targets ----------
for TARGET in ov ut; do
  echo ">>> [4/6] train_resclass --target $TARGET seed $SEED"
  python src/RaovSeg_recreation/train_resclass.py \
    --data-dir "$OUT_BASE/processed/train_val" \
    --output-dir "$OUT_BASE/models" \
    --seed "$SEED" --target "$TARGET"

  echo ">>> [4/6] train_attuseg --target $TARGET seed $SEED"
  python src/RaovSeg_recreation/train_attuseg.py \
    --data-dir "$OUT_BASE/processed/train_val" \
    --output-dir "$OUT_BASE/models" \
    --seed "$SEED" --target "$TARGET"
done

# ---------- 5) Evaluate both targets ----------
for TARGET in ov ut; do
  echo ">>> [5/6] evaluate --target $TARGET"
  python src/RaovSeg_recreation/evaluate.py \
    --test-dir "$OUT_BASE/processed/test" \
    --models-dir "$OUT_BASE/models" \
    --output-dir "$OUT_BASE/predictions_${TARGET}" \
    --target "$TARGET" \
    --metrics-out "$OUT_BASE/metrics_${TARGET}.json"
done

# ---------- 6) Write done-marker file the coordinator polls ----------
echo ">>> [6/6] mark done"
touch "$OUT_BASE/DONE"

# ---------- 7) Cleanup bulky intermediates (~5 GB -> ~5 MB per trial-seed) ----------
# The coordinator only needs metrics_ov.json, metrics_ut.json, iscs_score.json,
# and the DONE marker to compute the objective. Everything else (models,
# predictions, preprocessed real+synth data, synth NIfTIs) is re-derivable
# from the trial params and takes ~5 GB per (trial, seed). At 120 jobs, that's
# ~600 GB. This step trims each job to the essential JSONs only.
echo ">>> [7/7] cleanup intermediates (keeping metrics + DONE)"
rm -rf "$OUT_BASE/synth" \
       "$OUT_BASE/processed" \
       "$OUT_BASE/models" \
       "$OUT_BASE/predictions_ov" \
       "$OUT_BASE/predictions_ut"
echo "kept: $(ls $OUT_BASE)"

echo "=== finished: $(date) ==="
