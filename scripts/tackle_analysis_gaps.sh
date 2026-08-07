#!/bin/bash
# =============================================================================
# tackle_analysis_gaps.sh
#
# Orchestrator for the "analysis gap" work-list. Generates SBATCH scripts
# for items 2–6 from the LAMBDA_ABLATION_COLLAPSE.md aftermath discussion.
#
# All modes default to DRY-RUN (write scripts, print submit commands).
# Pass --submit to actually queue.
#
# MODES
#
#   reassemble VARIANT [--submit]
#       Item 2: Re-assemble a synth-volumes variant with enough SLURM time
#       to cover all 32 train subjects. VARIANT ∈ {exp1c_concat, exp1c_spade,
#       exp2, exp2_lam05_fixed, exp1c_spade_t022, exp1c_spade_t028, ...}.
#       Writes scripts/reassemble_<VARIANT>.sh. Runs the same
#       assemble_synth CLI but with --time=06:00:00 and rm+mkdir the
#       output dir so half-populated leftovers don't bias the retry.
#
#   real-baseline SEEDS [--submit]
#       Item 3: Real-only baseline (no --extra-train-dir) at N seeds.
#       SEEDS is a comma-list, e.g. 0,1,2,3,4. Writes one SBATCH per seed;
#       each runs the full RAovSeg pipeline on D2 real train only, evals
#       on the 8 sacred test subjects. ~5 h per seed on one A100.
#
#   downstream-1a1b [--submit]
#       Item 4: Assemble synth for 1a and 1b (currently not assembled),
#       then run RAovSeg augmented training + eval per seed. This is 4
#       SBATCHes: 2 assembly (a, b) + 2 downstream (a, b) — a template
#       to hand-launch per seed by editing SEED at the top of each.
#
#   synth-aware-preproc VARIANT [--submit]
#       Item 5: Rerun downstream training for VARIANT with
#       --skip-enhancement-for-prefix D2-9 enabled in preprocess. This
#       tests whether a synth-aware preprocessing path (skip RAovSeg's
#       enhancement band for synth subjects only) helps downstream DSC.
#       Writes scripts/raov_synth_aware_<VARIANT>_seedX.sh for seed 0..2.
#
#   larger-n N [--submit]
#       Item 6: Build a larger D2 split (N train subjects vs the current
#       32), regenerate the D2 generator preprocessing, retrain the
#       generator with 1c_spade config, reassemble, downstream. This
#       writes the full chain but is EXPENSIVE — full generator retrain
#       is 30–36 h. Prints the submission order and dependency flags.
#
# All generated SBATCHes:
#   - use the standard synth_mri env activation
#   - export MKL_THREADING_LAYER=GNU
#   - write logs to logs/<jobname>_%j.{out,err}
# =============================================================================
set -euo pipefail

REPO_ROOT="/mnt/parscratch/users/${USER}/synth_mri/EndometriosisDataset"
SCRIPTS_DIR="${REPO_ROOT}/scripts"
LOGS_DIR="${REPO_ROOT}/logs"
RUNS_ROOT="/mnt/parscratch/users/${USER}/synth_mri/runs"
SYNTH_ROOT="/mnt/parscratch/users/${USER}/synth_mri/synth_volumes"
RAW_DATA="${REPO_ROOT}/UT-EndoMRI/D2_TCPW"
GEN_ROOT="${REPO_ROOT}/data/processed_generator/D2"
GEN_SPLIT="${REPO_ROOT}/data/splits/d2_generator_split.json"

usage() { grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
require_hpc() {
    [[ -d "$REPO_ROOT" ]] || { echo "ERROR: not on HPC ($REPO_ROOT missing)"; exit 2; }
}

_sbatch_header() {
    # $1 = jobname, $2 = time_h (HH:MM:SS)
    cat <<EOF
#!/bin/bash
#SBATCH --job-name=$1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=4
#SBATCH --time=$2
#SBATCH --output=logs/$1_%j.out
#SBATCH --error=logs/$1_%j.err

set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export MKL_THREADING_LAYER=GNU

module purge
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1
set +u; source activate synth_mri; set -u

cd /mnt/parscratch/users/\${USER}/synth_mri/EndometriosisDataset
echo "=== JOB \$SLURM_JOB_ID on \$(hostname) ==="
echo "=== started: \$(date) ==="
nvidia-smi || true
EOF
}

# ---------- item 2: reassemble ----------

_get_variant_config() {
    # Map VARIANT → (config path, ckpt path, extra --ovary-target-intensity)
    # Emitted as: CONFIG|CKPT|OVARY_TARGET   (last field may be empty)
    case "$1" in
        exp1c_concat)         echo "src/Generator/exp1c_concat.yaml|${RUNS_ROOT}/exp1c_concat/ckpt/step_100000.pt|0.26" ;;
        exp1c_spade)          echo "src/Generator/exp1c_spade.yaml|${RUNS_ROOT}/exp1c_spade/ckpt/step_100000.pt|0.26" ;;
        exp1c_spade_t022)     echo "src/Generator/exp1c_spade.yaml|${RUNS_ROOT}/exp1c_spade/ckpt/step_100000.pt|0.22" ;;
        exp1c_spade_t028)     echo "src/Generator/exp1c_spade.yaml|${RUNS_ROOT}/exp1c_spade/ckpt/step_100000.pt|0.28" ;;
        exp2)                 echo "src/Generator/exp2.yaml|${RUNS_ROOT}/exp2_d1_gen_d2_disc/ckpt/step_100000.pt|0.26" ;;
        exp2_lam05_fixed)     echo "src/Generator/exp2_lam05_fixed.yaml|${RUNS_ROOT}/exp2_lam05_fixed/ckpt/step_100000.pt|0.26" ;;
        exp2_fixed)           echo "src/Generator/exp2_fixed.yaml|${RUNS_ROOT}/exp2_d1_gen_d2_disc_fixed/ckpt/step_100000.pt|0.26" ;;
        exp2_lam50_fixed)     echo "src/Generator/exp2_lam50_fixed.yaml|${RUNS_ROOT}/exp2_lam50_fixed/ckpt/step_100000.pt|0.26" ;;
        *) echo ""; return 1 ;;
    esac
}

cmd_reassemble() {
    require_hpc
    local variant="${1:-}"
    [[ -n "$variant" ]] || { echo "ERROR: reassemble needs VARIANT"; exit 3; }
    local spec
    spec=$(_get_variant_config "$variant") || {
        echo "ERROR: unknown variant '$variant'. Add it to _get_variant_config()."; exit 3; }
    local cfg="${spec%%|*}"; local rest="${spec#*|}"
    local ckpt="${rest%%|*}"; local ovt="${rest#*|}"

    local synth_dir="${SYNTH_ROOT}/${variant}"
    local out="${SCRIPTS_DIR}/reassemble_${variant}.sh"

    {
        _sbatch_header "reassemble_${variant}" "06:00:00"
        echo ""
        echo "SYNTH_DIR=${synth_dir}"
        echo "rm -rf \$SYNTH_DIR && mkdir -p \$SYNTH_DIR"
        echo ""
        echo "python -m src.Generator.assemble_synthetic_volumes \\"
        echo "  --config ${cfg} \\"
        echo "  --ckpt   ${ckpt} \\"
        echo "  --gen-preprocessed-root ${GEN_ROOT} \\"
        echo "  --gen-split-file ${GEN_SPLIT} \\"
        echo "  --raw-data-dir ${RAW_DATA} \\"
        echo "  --out-dir \$SYNTH_DIR \\"
        echo "  --iscs-alpha 0.8 \\"
        echo "  --noise-seed 0 \\"
        echo "  --ovary-target-intensity ${ovt}"
        echo ""
        echo "echo \"=== finished: \$(date) ===\""
        echo "echo \"=== subject count: \$(ls \$SYNTH_DIR | wc -l) ===\""
    } > "$out"
    chmod +x "$out"

    echo ""
    echo "  Item 2 — Reassemble $variant"
    echo "  Wrote:  $out"
    echo "  Submit:  sbatch $out"
    [[ "${SUBMIT:-0}" == "1" ]] && sbatch "$out"
}

# ---------- item 3: real-only baseline seeds ----------

cmd_real_baseline() {
    require_hpc
    local seeds_csv="${1:-}"
    local targets_csv="${2:-ov,ut}"
    [[ -n "$seeds_csv" ]] || { echo "ERROR: real-baseline needs SEEDS (e.g. 0,1,2,3,4)"; exit 3; }
    IFS=',' read -r -a seeds <<<"$seeds_csv"
    IFS=',' read -r -a targets <<<"$targets_csv"

    # Turn `("ov" "ut")` into a shell literal like `ov ut` that we embed
    # in the generated SBATCHes so the compute node loops correctly.
    local targets_list="${targets[*]}"

    for seed in "${seeds[@]}"; do
        local out="${SCRIPTS_DIR}/train_raovseg_real_only_seed${seed}.sh"
        local base="${RUNS_ROOT}/raovseg_real_only_seed${seed}"
        {
            _sbatch_header "raov_real_s${seed}" "16:00:00"
            echo ""
            echo "OUT_BASE=${base}"
            echo "mkdir -p \$OUT_BASE/processed \$OUT_BASE/models"
            echo ""
            echo ">>> [1] preprocess (real only, no --extra-train-dir)"
            echo "python src/RaovSeg_recreation/preprocess.py \\"
            echo "  --data-dir ${RAW_DATA} \\"
            echo "  --output-dir \$OUT_BASE/processed"
            echo ""
            echo ">>> [2] train + evaluate BOTH targets (matches tier1_run_config.sh)"
            echo "for TARGET in ${targets_list}; do"
            echo "  echo \"--- train_resclass --target \$TARGET seed ${seed} ---\""
            echo "  python src/RaovSeg_recreation/train_resclass.py \\"
            echo "    --data-dir \$OUT_BASE/processed/train_val \\"
            echo "    --output-dir \$OUT_BASE/models \\"
            echo "    --seed ${seed} --target \$TARGET"
            echo ""
            echo "  echo \"--- train_attuseg --target \$TARGET seed ${seed} ---\""
            echo "  python src/RaovSeg_recreation/train_attuseg.py \\"
            echo "    --data-dir \$OUT_BASE/processed/train_val \\"
            echo "    --output-dir \$OUT_BASE/models \\"
            echo "    --seed ${seed} --target \$TARGET"
            echo ""
            echo "  echo \"--- evaluate --target \$TARGET ---\""
            echo "  mkdir -p \$OUT_BASE/predictions_\${TARGET}"
            echo "  python src/RaovSeg_recreation/evaluate.py \\"
            echo "    --test-dir \$OUT_BASE/processed/test \\"
            echo "    --models-dir \$OUT_BASE/models \\"
            echo "    --output-dir \$OUT_BASE/predictions_\${TARGET} \\"
            echo "    --target \$TARGET \\"
            echo "    --metrics-out \$OUT_BASE/metrics_\${TARGET}.json"
            echo "done"
            echo ""
            echo "echo \"=== finished: \$(date) ===\""
        } > "$out"
        chmod +x "$out"
        echo "  wrote $out"
        [[ "${SUBMIT:-0}" == "1" ]] && sbatch "$out"
    done

    echo ""
    echo "  Item 3 — Real-only baseline (${#seeds[@]} seeds × ${#targets[@]} targets: ${targets_list})"
    echo "  NOTE: existing augmented run_raovseg_aug_*.sh templates only train"
    echo "        the OVARY head (default --target). If you want a strict"
    echo "        paired comparison on uterus too, the augmented runs also"
    echo "        need to be re-done with the ut loop — the current ones"
    echo "        have no uterus data. Ovary paired-Wilcoxon is unaffected."
    echo ""
    echo "  Submit all:"
    for seed in "${seeds[@]}"; do
        echo "    sbatch scripts/train_raovseg_real_only_seed${seed}.sh"
    done
    echo ""
    echo "  After all finish, run paired-Wilcoxon per target:"
    for tgt in "${targets[@]}"; do
        echo "    python -m src.analysis.paired_wilcoxon_from_metrics \\"
        echo "      --baseline-glob 'runs/raovseg_real_only_seed*/metrics_${tgt}.json' \\"
        echo "      --metrics-glob  'runs/raovseg_aug_*/metrics_${tgt}.json' \\"
        echo "      --out-json metrics/paired_wilcoxon_${tgt}.json \\"
        echo "      --out-png  figures/fig_paired_wilcoxon_${tgt}.png"
    done
}

# ---------- item 4: downstream DSC for 1a and 1b ----------

cmd_downstream_1a1b() {
    require_hpc

    # 1a and 1b: assemble first if not present, then downstream per seed.
    # NOTE: 1a and 1b were trained WITHOUT PatchGAN, so assembly config
    # is just the base 1a.yaml / 1b.yaml (no --ovary-target-intensity
    # constraint changes; use default 0.26).
    for variant in exp1a exp1b; do
        local ckpt="${RUNS_ROOT}/${variant}/ckpt/step_080000.pt"
        local synth_dir="${SYNTH_ROOT}/${variant}"
        local out="${SCRIPTS_DIR}/reassemble_${variant}.sh"
        {
            _sbatch_header "assemble_${variant}" "04:00:00"
            echo ""
            echo "SYNTH_DIR=${synth_dir}"
            echo "rm -rf \$SYNTH_DIR && mkdir -p \$SYNTH_DIR"
            echo ""
            echo "python -m src.Generator.assemble_synthetic_volumes \\"
            echo "  --config src/Generator/${variant}.yaml \\"
            echo "  --ckpt   ${ckpt} \\"
            echo "  --gen-preprocessed-root ${GEN_ROOT} \\"
            echo "  --gen-split-file ${GEN_SPLIT} \\"
            echo "  --raw-data-dir ${RAW_DATA} \\"
            echo "  --out-dir \$SYNTH_DIR \\"
            echo "  --iscs-alpha 0.8 --noise-seed 0 \\"
            echo "  --ovary-target-intensity 0.26"
        } > "$out"
        chmod +x "$out"
        echo "  wrote assembly: $out"

        for seed in 0 1 2; do
            local dout="${SCRIPTS_DIR}/run_raovseg_aug_${variant}_seed${seed}.sh"
            local base="${RUNS_ROOT}/raovseg_aug_${variant}_seed${seed}"
            {
                _sbatch_header "raov_aug_${variant}_s${seed}" "08:00:00"
                echo ""
                echo "OUT_BASE=${base}"
                echo "SYNTH_DIR=${synth_dir}"
                echo "mkdir -p \$OUT_BASE/processed \$OUT_BASE/models \$OUT_BASE/predictions"
                echo ""
                echo "python src/RaovSeg_recreation/preprocess.py \\"
                echo "  --data-dir ${RAW_DATA} \\"
                echo "  --extra-train-dir \$SYNTH_DIR \\"
                echo "  --output-dir \$OUT_BASE/processed"
                echo ""
                echo "python src/RaovSeg_recreation/train_resclass.py \\"
                echo "  --data-dir \$OUT_BASE/processed/train_val \\"
                echo "  --output-dir \$OUT_BASE/models --seed ${seed}"
                echo ""
                echo "python src/RaovSeg_recreation/train_attuseg.py \\"
                echo "  --data-dir \$OUT_BASE/processed/train_val \\"
                echo "  --output-dir \$OUT_BASE/models --seed ${seed}"
                echo ""
                echo "python src/RaovSeg_recreation/evaluate.py \\"
                echo "  --test-dir \$OUT_BASE/processed/test \\"
                echo "  --models-dir \$OUT_BASE/models \\"
                echo "  --output-dir \$OUT_BASE/predictions \\"
                echo "  --metrics-out \$OUT_BASE/metrics_ov.json"
            } > "$dout"
            chmod +x "$dout"
            echo "  wrote downstream: $dout"
        done
    done

    echo ""
    echo "  Item 4 — Downstream DSC for 1a and 1b (3 seeds each = 8 SBATCHes)"
    echo "  Submit order (assembly first, downstream depends on it):"
    for variant in exp1a exp1b; do
        echo "    A=\$(sbatch --parsable scripts/reassemble_${variant}.sh)"
        for seed in 0 1 2; do
            echo "    sbatch --dependency=afterok:\$A scripts/run_raovseg_aug_${variant}_seed${seed}.sh"
        done
    done
}

# ---------- item 5: synth-aware preprocessing branch ----------

cmd_synth_aware_preproc() {
    require_hpc
    local variant="${1:-}"
    [[ -n "$variant" ]] || { echo "ERROR: synth-aware-preproc needs VARIANT"; exit 3; }
    local synth_dir="${SYNTH_ROOT}/${variant}"

    for seed in 0 1 2; do
        local out="${SCRIPTS_DIR}/run_raovseg_synth_aware_${variant}_seed${seed}.sh"
        local base="${RUNS_ROOT}/raovseg_synth_aware_${variant}_seed${seed}"
        {
            _sbatch_header "raov_saw_${variant}_s${seed}" "08:00:00"
            echo ""
            echo "OUT_BASE=${base}"
            echo "SYNTH_DIR=${synth_dir}"
            echo "mkdir -p \$OUT_BASE/processed \$OUT_BASE/models \$OUT_BASE/predictions"
            echo ""
            echo ">>> preprocess WITH --skip-enhancement-for-prefix D2-9"
            echo "python src/RaovSeg_recreation/preprocess.py \\"
            echo "  --data-dir ${RAW_DATA} \\"
            echo "  --extra-train-dir \$SYNTH_DIR \\"
            echo "  --output-dir \$OUT_BASE/processed \\"
            echo "  --skip-enhancement-for-prefix D2-9"
            echo ""
            echo ">>> train_resclass --seed ${seed}"
            echo "python src/RaovSeg_recreation/train_resclass.py \\"
            echo "  --data-dir \$OUT_BASE/processed/train_val \\"
            echo "  --output-dir \$OUT_BASE/models --seed ${seed}"
            echo ""
            echo ">>> train_attuseg --seed ${seed}"
            echo "python src/RaovSeg_recreation/train_attuseg.py \\"
            echo "  --data-dir \$OUT_BASE/processed/train_val \\"
            echo "  --output-dir \$OUT_BASE/models --seed ${seed}"
            echo ""
            echo ">>> evaluate"
            echo "python src/RaovSeg_recreation/evaluate.py \\"
            echo "  --test-dir \$OUT_BASE/processed/test \\"
            echo "  --models-dir \$OUT_BASE/models \\"
            echo "  --output-dir \$OUT_BASE/predictions \\"
            echo "  --metrics-out \$OUT_BASE/metrics_ov.json"
        } > "$out"
        chmod +x "$out"
        echo "  wrote $out"
        [[ "${SUBMIT:-0}" == "1" ]] && sbatch "$out"
    done

    echo ""
    echo "  Item 5 — Synth-aware preprocessing for $variant (3 seeds)"
    echo "  Difference from standard aug: --skip-enhancement-for-prefix D2-9"
    echo "  passed to preprocess.py, so synth D2-9XX subjects skip the"
    echo "  [0.22, 0.30] enhancement window while real D2-0XX get it."
}

# ---------- item 6: larger-n split ----------

cmd_larger_n() {
    require_hpc
    local n="${1:-}"
    [[ -n "$n" ]] || { echo "ERROR: larger-n needs N (e.g. 60)"; exit 3; }

    # 1) Build the new split
    local split_out="${REPO_ROOT}/data/splits/d2_generator_split_n${n}.json"
    local build_script="${SCRIPTS_DIR}/build_d2_split_n${n}.py"
    cat > "$build_script" <<PY
#!/usr/bin/env python3
"""One-off: build a larger D2 generator split with N training subjects.

Keeps the 8 sacred test subjects fixed. Draws train from the remaining
available D2-XXX subject dirs, preserving any current train subjects as
a prefix so partial models can be resumed.
"""
import json, sys
from pathlib import Path

N = ${n}
SACRED_TEST = ["D2-005", "D2-015", "D2-016", "D2-017",
               "D2-023", "D2-024", "D2-026", "D2-038"]

raw = Path("${RAW_DATA}")
all_subj = sorted(p.name for p in raw.iterdir() if p.is_dir() and p.name.startswith("D2-"))
pool = [s for s in all_subj if s not in SACRED_TEST]

# Seed with current split's train (so if we ever want to resume from a
# smaller checkpoint, ordering is stable)
current = Path("${GEN_SPLIT}")
if current.exists():
    curr_train = json.load(current.open()).get("train", [])
    curr_train = [s for s in curr_train if s in pool]
    remaining = [s for s in pool if s not in curr_train]
    train = curr_train + remaining
else:
    train = pool

if len(train) < N:
    sys.exit(f"only {len(train)} subjects available; requested N={N}")

train = train[:N]
out = {"train": train, "test": SACRED_TEST}
with open("${split_out}", "w") as f:
    json.dump(out, f, indent=2)
print(f"wrote {'${split_out}'} with {len(train)} train + {len(SACRED_TEST)} test")
PY
    chmod +x "$build_script"
    echo "  wrote split builder: $build_script"

    # 2) Preprocess for larger n
    local prep_script="${SCRIPTS_DIR}/gen_preprocess_n${n}.sh"
    {
        _sbatch_header "gen_preprocess_n${n}" "04:00:00"
        echo ""
        echo "python -m src.Generator.build_generator_split \\"
        echo "  --split-file ${split_out} \\"
        echo "  --raw-data ${RAW_DATA} \\"
        echo "  --out-dir ${REPO_ROOT}/data/processed_generator/D2_n${n} || true"
        echo ""
        echo "# NOTE: if build_generator_split's flags differ, adjust above."
        echo "# You may need to call preprocess_for_generator.py directly."
    } > "$prep_script"
    chmod +x "$prep_script"
    echo "  wrote preprocess: $prep_script"

    # 3) Train generator with 1c_spade config on the larger split
    local train_yaml="${REPO_ROOT}/src/Generator/exp1c_spade_n${n}.yaml"
    local train_script="${SCRIPTS_DIR}/train_exp1c_spade_n${n}.sh"
    {
        _sbatch_header "train_1c_spade_n${n}" "36:00:00"
        echo ""
        echo "# NOTE: the YAML ${train_yaml} must be created first — copy"
        echo "# src/Generator/exp1c_spade.yaml and update:"
        echo "#   experiment.output_dir → runs/exp1c_spade_n${n}"
        echo "#   data.preprocessed_root → data/processed_generator/D2_n${n}"
        echo "#   data.split_file → ${split_out}"
        echo "python -m src.Generator.train --config ${train_yaml}"
    } > "$train_script"
    chmod +x "$train_script"
    echo "  wrote train: $train_script"

    # 4) Assembly + downstream (per seed) — reuse the reassemble machinery
    local assemble_script="${SCRIPTS_DIR}/reassemble_exp1c_spade_n${n}.sh"
    {
        _sbatch_header "assemble_1c_spade_n${n}" "06:00:00"
        echo ""
        echo "SYNTH_DIR=${SYNTH_ROOT}/exp1c_spade_n${n}"
        echo "rm -rf \$SYNTH_DIR && mkdir -p \$SYNTH_DIR"
        echo ""
        echo "python -m src.Generator.assemble_synthetic_volumes \\"
        echo "  --config ${train_yaml} \\"
        echo "  --ckpt   ${RUNS_ROOT}/exp1c_spade_n${n}/ckpt/step_100000.pt \\"
        echo "  --gen-preprocessed-root ${REPO_ROOT}/data/processed_generator/D2_n${n} \\"
        echo "  --gen-split-file ${split_out} \\"
        echo "  --raw-data-dir ${RAW_DATA} \\"
        echo "  --out-dir \$SYNTH_DIR \\"
        echo "  --iscs-alpha 0.8 --noise-seed 0 \\"
        echo "  --ovary-target-intensity 0.26"
    } > "$assemble_script"
    chmod +x "$assemble_script"
    echo "  wrote assembly: $assemble_script"

    echo ""
    echo "  Item 6 — Larger-n pipeline (N=${n})"
    echo ""
    echo "  Manual steps (unavoidable — this is a bulk pipeline):"
    echo "    1) python3 ${build_script}"
    echo "    2) cp src/Generator/exp1c_spade.yaml ${train_yaml}"
    echo "       # then edit output_dir, preprocessed_root, split_file"
    echo "    3) sbatch ${prep_script}"
    echo "    4) A=\$(sbatch --parsable --dependency=afterok:\$3 ${train_script})"
    echo "    5) B=\$(sbatch --parsable --dependency=afterok:\$A ${assemble_script})"
    echo "    6) ./scripts/tackle_analysis_gaps.sh real-baseline 0,1,2   # after new split preprocess"
    echo ""
    echo "  Full pipeline is ~3 days of wall-clock. Only pursue if the"
    echo "  minimum retrain set (§11.1 of LAMBDA_ABLATION_COLLAPSE.md)"
    echo "  didn't answer the question."
}

# ------------------------------------------------------------- main

if [[ $# -lt 1 ]]; then usage; fi
MODE="$1"; shift

SUBMIT=0
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --submit) SUBMIT=1; shift ;;
        --help|-h) usage ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done
set -- "${POSITIONAL[@]:+${POSITIONAL[@]}}"
export SUBMIT

case "$MODE" in
    reassemble)          cmd_reassemble "${1:-}" ;;
    real-baseline)       cmd_real_baseline "${1:-}" ;;
    downstream-1a1b)     cmd_downstream_1a1b ;;
    synth-aware-preproc) cmd_synth_aware_preproc "${1:-}" ;;
    larger-n)            cmd_larger_n "${1:-}" ;;
    -h|--help)           usage ;;
    *) echo "ERROR: unknown mode '$MODE'"; usage ;;
esac
