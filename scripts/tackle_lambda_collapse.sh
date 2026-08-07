#!/bin/bash
# =============================================================================
# tackle_lambda_collapse.sh
#
# Orchestrator for the three levels of investigating / fixing the exp2 λ
# ablation collapse. Full backstory in LAMBDA_ABLATION_COLLAPSE.md.
#
# All three levels are HPC-only. This script writes SBATCH scripts and
# optionally submits them. Run from an interactive session in the project
# root:
#
#     cd /mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset
#     ./scripts/tackle_lambda_collapse.sh <mode> [args] [--submit]
#
# MODES
#
#   diag [--submit]
#       Level 1: instrument train.py to log per-step gradient norms of
#       |∇L_diff| vs |∇(λ·L_adv)|. Trains exp2_lam05 from scratch for 15 000
#       steps (past λ warmup at 10 000, into ramp). ~60–80 min on one A100.
#       Answers hypothesis 5.1 (saturated-D adversarial gradient underflow)
#       vs 5.2 (determinism) from LAMBDA_ABLATION_COLLAPSE.md.
#
#   retrain-one FIX_TYPE [--submit]
#       Level 2: writes a fixed variant config + SBATCH. Trains ONE variant
#       (exp2_lam05_fixed) with the chosen fix. ~24–36 h. Confirms whether
#       the fix produces weights that differ from exp2 at the tensor level.
#
#   retrain-all FIX_TYPE [--submit]
#       Level 3: same as retrain-one but for all three variants
#       (exp2_fixed, exp2_lam05_fixed, exp2_lam50_fixed). ~24–36 h × 3 in
#       parallel if the cluster has queue room.
#
# FIX_TYPE (for retrain-one / retrain-all):
#
#   seed        — vary experiment.seed per variant (42, 43, 44)
#                 Attacks hypothesis 5.2 only. Cheapest fix; no code change.
#   no-resume   — set experiment.resume: false
#                 Attacks hypothesis 5.2 only. Guarantees fresh start.
#   both        — combine seed + no-resume
#                 Belt-and-suspenders for 5.2. Recommended for retrain-*.
#   hinge       — NOT IMPLEMENTED. Would require training-loop code change to
#                 replace BCE-based generator_adv_loss with hinge form.
#                 Attacks hypothesis 5.1. See LAMBDA_ABLATION_COLLAPSE.md §5.
#
# Without --submit, the script writes the SBATCH files and prints the
# submit commands but does not queue anything. This is the default; you
# review before submitting.
#
# =============================================================================
set -euo pipefail

REPO_ROOT="/mnt/parscratch/users/${USER}/synth_mri/EndometriosisDataset"
CONFIGS_DIR="${REPO_ROOT}/src/Generator"
SCRIPTS_DIR="${REPO_ROOT}/scripts"
LOGS_DIR="${REPO_ROOT}/logs"
RUNS_ROOT="/mnt/parscratch/users/${USER}/synth_mri/runs"

# The seed values used when FIX_TYPE includes `seed`. Keep them distinct.
SEEDS_PER_VARIANT=("42" "43" "44")

# The lambda_peak values that define the three variants. Order aligned with
# SEEDS_PER_VARIANT so index 0 = exp2, 1 = exp2_lam05, 2 = exp2_lam50.
VARIANT_NAMES=("exp2" "exp2_lam05" "exp2_lam50")
VARIANT_LAMBDA=("0.01" "0.05" "0.5")

usage() {
    grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

require_hpc() {
    if [[ ! -d "$REPO_ROOT" ]]; then
        echo "ERROR: expected repo at $REPO_ROOT — are you on HPC?" >&2
        exit 2
    fi
    if [[ "$(hostname)" == login* ]]; then
        echo "WARNING: on a login node. Compute-heavy submissions are fine, but" >&2
        echo "         don't run diag as an interactive foreground here."       >&2
    fi
}

# ------------------------------------------------------------------- diag

write_diag_sbatch() {
    # Fully self-contained SBATCH. Does its own truncated-config generation
    # and its own GRAD_DIAG toggle on the compute node — no cross-node file
    # dependencies. train.py already carries the GRAD_DIAG-guarded logging
    # block (added Jul 2026), so we don't patch anything.
    local out="${SCRIPTS_DIR}/_diag_lambda_grads.sbatch"
    cat >"$out" <<'EOF'
#!/bin/bash
#SBATCH --job-name=diag_lam_grads
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/diag_lambda_grads_%j.out
#SBATCH --error=logs/diag_lambda_grads_%j.err

set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export MKL_THREADING_LAYER=GNU

module purge
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1
set +u; source activate synth_mri; set -u

cd /mnt/parscratch/users/${USER}/synth_mri/EndometriosisDataset

echo "=== JOB $SLURM_JOB_ID on $(hostname) ==="
echo "=== started: $(date) ==="
nvidia-smi || true

# Write truncated diag config to this compute node's /tmp — no cross-node
# file dependency, unlike the earlier version. Also throw the checkpoints
# / TB dir into /tmp so they get wiped on job end.
python3 - <<'PY'
import yaml
with open("src/Generator/exp2_lam05.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["experiment"]["name"] = "exp2_lam05_diag"
cfg["experiment"]["output_dir"] = "/tmp/runs_diag/exp2_lam05_diag"
cfg["experiment"]["resume"] = False          # fresh warmup ramp
cfg["training"]["total_steps"] = 15000       # past warmup (10k), into ramp
cfg["training"]["log_every"] = 25            # finer resolution near λ ramp
cfg["training"]["sample_every"] = 999999
cfg["training"]["ckpt_every"] = 999999
with open("/tmp/exp2_lam05_diag.yaml", "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("[diag] wrote /tmp/exp2_lam05_diag.yaml")
PY
mkdir -p /tmp/runs_diag

# GRAD_DIAG=1 activates the gradient-split logging block in train.py.
export GRAD_DIAG=1
python -m src.Generator.train --config /tmp/exp2_lam05_diag.yaml

echo "=== finished: $(date) ==="
echo ""
echo "=== [GRAD_DIAG] summary ==="
grep '\[GRAD_DIAG\]' logs/diag_lambda_grads_${SLURM_JOB_ID}.out | tail -40 || true
EOF
    chmod +x "$out"
    echo "[diag] wrote SBATCH → $out"
}

cmd_diag() {
    require_hpc
    mkdir -p "$LOGS_DIR"
    write_diag_sbatch

    echo ""
    echo "=========================================================="
    echo "  Level 1 — Diagnostic ready"
    echo "=========================================================="
    echo "  SBATCH:   scripts/_diag_lambda_grads.sbatch"
    echo "  (Config and env vars are generated on the compute node — no"
    echo "   cross-node file dependencies. Requires train.py that includes"
    echo "   the GRAD_DIAG-guarded block; pushed Jul 2026.)"
    echo ""
    echo "  Submit with:"
    echo "    sbatch scripts/_diag_lambda_grads.sbatch"
    echo ""
    echo "  Watch progress after submission:"
    echo "    tail -f logs/diag_lambda_grads_*.out | grep -E 'GRAD_DIAG|step '"
    echo ""
    echo "  Interpretation of the [GRAD_DIAG] lines:"
    echo "    ratio ≤ 1e-6 : adversarial gradient is being underflowed."
    echo "                   Fix = code change (hinge loss / non-saturating)."
    echo "                   Config-only fixes (seed/no-resume) will NOT help."
    echo "    ratio ≥ 1e-3 : adversarial gradient is meaningful."
    echo "                   Fix = config change (seed / no-resume)."
    echo "                   The collapse is a determinism artefact."
    echo "=========================================================="

    if [[ "${SUBMIT:-0}" == "1" ]]; then
        sbatch "${SCRIPTS_DIR}/_diag_lambda_grads.sbatch"
    fi
}

# ---------------------------------------------------------- retrain

write_variant_config() {
    # $1 = variant index (0..2 into VARIANT_NAMES / VARIANT_LAMBDA / SEEDS)
    # $2 = fix type: seed | no-resume | both
    # Writes {variant}_fixed.yaml under src/Generator/, deriving from the
    # unfixed variant's config and applying seed and/or resume: false.
    local idx=$1
    local fix=$2
    local name="${VARIANT_NAMES[$idx]}"
    local lam="${VARIANT_LAMBDA[$idx]}"
    local seed="${SEEDS_PER_VARIANT[$idx]}"

    # exp2 uses runs/exp2_d1_gen_d2_disc as its output dir, others match name
    local out_subdir
    if [[ "$name" == "exp2" ]]; then
        out_subdir="exp2_d1_gen_d2_disc_fixed"
    else
        out_subdir="${name}_fixed"
    fi

    local src_cfg="${CONFIGS_DIR}/${name}.yaml"
    local dst_cfg="${CONFIGS_DIR}/${name}_fixed.yaml"

    python3 - "$src_cfg" "$dst_cfg" "$name" "$lam" "$seed" "$fix" "$out_subdir" <<'PY'
import sys, yaml
src, dst, name, lam, seed, fix, out_subdir = sys.argv[1:]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["experiment"]["name"] = f"{name}_fixed"
cfg["experiment"]["output_dir"] = f"/mnt/parscratch/users/ijp25lg/synth_mri/runs/{out_subdir}"
if fix in ("seed", "both"):
    cfg["experiment"]["seed"] = int(seed)
if fix in ("no-resume", "both"):
    cfg["experiment"]["resume"] = False
cfg["discriminator"]["lambda_peak"] = float(lam)
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"[retrain] wrote config → {dst}  (seed={seed}, lambda_peak={lam}, resume={cfg['experiment']['resume']})")
PY
}

write_variant_sbatch() {
    # $1 = variant name
    local name=$1
    local out="${SCRIPTS_DIR}/train_${name}_fixed.sh"
    cat >"$out" <<EOF
#!/bin/bash
#SBATCH --job-name=${name}_fixed
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=4
#SBATCH --time=36:00:00
#SBATCH --output=logs/${name}_fixed_%j.out
#SBATCH --error=logs/${name}_fixed_%j.err

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
git rev-parse HEAD || echo "(not a git repo)"
echo "=== running ==="

python -m src.Generator.train --config src/Generator/${name}_fixed.yaml

echo "=== finished: \$(date) ==="
EOF
    chmod +x "$out"
    echo "[retrain] wrote SBATCH → $out"
}

cmd_retrain_one() {
    require_hpc
    local fix="${1:-}"
    if [[ -z "$fix" ]]; then
        echo "ERROR: retrain-one requires FIX_TYPE (seed | no-resume | both | hinge)" >&2
        exit 3
    fi
    if [[ "$fix" == "hinge" ]]; then
        echo "ERROR: hinge fix requires training-loop code changes and is not"  >&2
        echo "       automated by this script. See LAMBDA_ABLATION_COLLAPSE.md §7." >&2
        exit 3
    fi
    if [[ "$fix" != "seed" && "$fix" != "no-resume" && "$fix" != "both" ]]; then
        echo "ERROR: unknown FIX_TYPE '$fix'. Use seed | no-resume | both."      >&2
        exit 3
    fi

    # Retrain exp2_lam05 only (index 1) as the canonical test.
    write_variant_config 1 "$fix"
    write_variant_sbatch "exp2_lam05"

    echo ""
    echo "=========================================================="
    echo "  Level 2 — Single-variant retrain ready"
    echo "=========================================================="
    echo "  Variant:  exp2_lam05 (λ_peak=0.05)"
    echo "  Fix:      $fix"
    echo "  Config:   src/Generator/exp2_lam05_fixed.yaml"
    echo "  SBATCH:   scripts/train_exp2_lam05_fixed.sh"
    echo ""
    echo "  Submit with:"
    echo "    sbatch scripts/train_exp2_lam05_fixed.sh"
    echo ""
    echo "  After it finishes (~36 h), verify divergence from exp2:"
    echo "    md5sum runs/exp2_d1_gen_d2_disc/ckpt/step_100000.pt \\"
    echo "           runs/exp2_lam05_fixed/ckpt/step_100000.pt"
    echo "  Then compare the EMA tensor hashes with the Python snippet in"
    echo "  LAMBDA_ABLATION_COLLAPSE.md §3.5. Different tensor hashes → fix"
    echo "  works. Same tensor hashes → hypothesis 5.1 is the culprit and"
    echo "  a code-level (hinge) fix is required."
    echo "=========================================================="

    if [[ "${SUBMIT:-0}" == "1" ]]; then
        sbatch "${SCRIPTS_DIR}/train_exp2_lam05_fixed.sh"
    fi
}

cmd_retrain_all() {
    require_hpc
    local fix="${1:-}"
    if [[ -z "$fix" ]]; then
        echo "ERROR: retrain-all requires FIX_TYPE (seed | no-resume | both | hinge)" >&2
        exit 3
    fi
    if [[ "$fix" == "hinge" ]]; then
        echo "ERROR: hinge fix requires training-loop code changes and is not"  >&2
        echo "       automated by this script. See LAMBDA_ABLATION_COLLAPSE.md §7." >&2
        exit 3
    fi
    if [[ "$fix" != "seed" && "$fix" != "no-resume" && "$fix" != "both" ]]; then
        echo "ERROR: unknown FIX_TYPE '$fix'. Use seed | no-resume | both."      >&2
        exit 3
    fi

    for i in 0 1 2; do
        write_variant_config "$i" "$fix"
        write_variant_sbatch "${VARIANT_NAMES[$i]}"
    done

    echo ""
    echo "=========================================================="
    echo "  Level 3 — Full ablation retrain ready"
    echo "=========================================================="
    echo "  Fix applied: $fix"
    echo ""
    for i in 0 1 2; do
        printf "  Variant %d: %-13s  seed=%s  lambda_peak=%s\n" \
            "$i" "${VARIANT_NAMES[$i]}" "${SEEDS_PER_VARIANT[$i]}" "${VARIANT_LAMBDA[$i]}"
    done
    echo ""
    echo "  Submit all three (parallel if queue allows):"
    for name in "${VARIANT_NAMES[@]}"; do
        echo "    sbatch scripts/train_${name}_fixed.sh"
    done
    echo ""
    echo "  After all three finish, verify divergence:"
    echo "    python3 <<'PY'"
    echo "    import torch, hashlib"
    echo "    paths = ["
    echo "        'runs/exp2_d1_gen_d2_disc_fixed/ckpt/step_100000.pt',"
    echo "        'runs/exp2_lam05_fixed/ckpt/step_100000.pt',"
    echo "        'runs/exp2_lam50_fixed/ckpt/step_100000.pt',"
    echo "    ]"
    echo "    def h(sd):"
    echo "        hh = hashlib.md5()"
    echo "        for k in sorted(sd.keys()):"
    echo "            hh.update(k.encode()); hh.update(sd[k].detach().cpu().numpy().tobytes())"
    echo "        return hh.hexdigest()"
    echo "    for p in paths:"
    echo "        c = torch.load(p, map_location='cpu', weights_only=False)"
    echo "        print(p, 'EMA:', h(c['ema']))"
    echo "    PY"
    echo ""
    echo "  Three distinct EMA hashes → collapse fixed."
    echo "  If any two are still equal → hypothesis 5.1 still dominates;"
    echo "  the config-level fix isn't enough. Move to a hinge-loss patch."
    echo "=========================================================="

    if [[ "${SUBMIT:-0}" == "1" ]]; then
        for name in "${VARIANT_NAMES[@]}"; do
            sbatch "${SCRIPTS_DIR}/train_${name}_fixed.sh"
        done
    fi
}

# ------------------------------------------------------------- main

if [[ $# -lt 1 ]]; then
    usage
fi

MODE="$1"
shift

# Parse --submit anywhere in the remaining args.
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
    diag)          cmd_diag ;;
    retrain-one)   cmd_retrain_one "${1:-}" ;;
    retrain-all)   cmd_retrain_all "${1:-}" ;;
    -h|--help)     usage ;;
    *) echo "ERROR: unknown mode '$MODE'" >&2; usage ;;
esac
