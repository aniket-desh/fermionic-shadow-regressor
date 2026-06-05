#!/bin/bash
# =======================================================================
# Regression v18: chemical-input advantage (orb vs R) on the BEST STABLE model.
#
# WHY THIS EXISTS
#   Manuscript pillar 4 ("R-input is the baseline control; chemically-informed
#   orbital-energy inputs improve data efficiency/robustness") is a load-bearing
#   claim, but v17 ran the orb-vs-R ablation on the v16/v17 residual-trunk +
#   decoupled-clipping architecture, which is UNSTABLE (train_mse → 1e6, seed
#   |Δ| on borderline = 0.78, R arm flatlined to pearson≈0.00 / crashed). That
#   conflated two unrelated changes and produced no usable baseline.
#
#   v18 strips all of that back to the BEST + ONLY STABLE architecture we have:
#   v15_explicit (explicit-amplitude composition, amp_rank 16, NO residual trunk,
#   NO gradient clipping — v15 trained stably unclipped precisely because it has
#   no trunk). The ONLY variable is the input: orbital energies vs scalar R.
#   This gives a clean, credible pillar-4 comparison: orb should beat R, and R
#   should still TRAIN (be "works but worse", not "zero correlation").
#
# MATRIX: {orb, R} × {seed 42, 1729} = 4 train + 4 eval jobs.
# DATASET: reuses h4_regress_v13 (dt=0.05). NO datagen stage.
#
# Architecture = v15_explicit exactly (see slurm/regression_v15.sh):
#   --adaptive_bandwidth --omega_op_floor 8.0 --soft_omega_floor
#   --explicit_amplitude --amp_rank 16 --grad_clip 1.0
#   (NO --with_residual; --grad_clip 1.0 ADDED after the original unclipped v18
#    orb arm collapsed (0.323). v18b proved the clip restores it (0.922). The
#    "stable unclipped" premise was wrong — see the COMMON_ARCH_FLAGS note below.)
#   v18_orb : + --use_orb_features --standardize_orb_energies   (chemically informed)
#   v18_R   : (neither)                                         (geometry only; scalar R)
#
# Decision rule (per-bin Pearson seed-pooled n=100):
#   - v18_orb beats v18_R by a clear per-bin margin (esp. short/borderline R)
#     AND v18_R still trains (per-bin pearson well above 0; not a flatline)
#       → supports the chemically-informed-features claim. Ship it.
#   - v18_R ≈ v18_orb across bins
#       → geometry alone suffices; report honestly, drop the orb-feature claim.
#   - v18_R flatlines to ≈0 again
#       → the R pipeline is broken (not an honest baseline); debug the R input
#         path before drawing any orb-vs-R conclusion. Do NOT report it as a win.
#
# Stages (composable; default = train + eval):
#   --train  submit training jobs (4 = 2 input-modes × 2 seeds)
#   --eval   submit eval+plot+composition jobs (one per checkpoint)
#
# Usage:
#   bash slurm/regression_v18.sh                          # train + eval, all 4
#   bash slurm/regression_v18.sh --eval                   # eval-only on existing ckpts
#   bash slurm/regression_v18.sh --train --eval --exclude trig0019,trig0034
# =======================================================================
set -euo pipefail

TAG="h4_regress_v18"
DATA_TAG="h4_regress_v13"   # reuse the dt=0.05 canonical dataset
OMEGA_OP_FLOOR="8.0"
DO_TRAIN=false; DO_EVAL=false
ANY_STAGE=false
EXCLUDE_NODES=""

usage() {
  cat << EOF
Usage: $0 [--train] [--eval] [--all] [--tag NAME] [--data_tag NAME] [--omega_op_floor F] [--exclude NODELIST]

Stages (composable; default = train + eval):
  --train     submit training jobs ({orb,R} × 2 seeds = 4)
  --eval      submit eval+plot+composition jobs (one per checkpoint)
  --all       explicit form of "train + eval"

Overrides:
  --tag NAME             model tag prefix (default: h4_regress_v18)
  --data_tag NAME        upstream dataset tag (default: h4_regress_v13)
  --omega_op_floor F     ω_floor in E_h (default: 8.0)
  --exclude NODELIST     SLURM --exclude for all jobs (e.g. trig0019,trig0034 —
                         the flaky-GPU trig nodes that crashed v16/v17 evals).
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --train)            DO_TRAIN=true; ANY_STAGE=true ;;
    --eval)             DO_EVAL=true; ANY_STAGE=true ;;
    --all)              DO_TRAIN=true; DO_EVAL=true; ANY_STAGE=true ;;
    --tag)              TAG="$2"; shift ;;
    --data_tag)         DATA_TAG="$2"; shift ;;
    --omega_op_floor)   OMEGA_OP_FLOOR="$2"; shift ;;
    --exclude)          EXCLUDE_NODES="$2"; shift ;;
    -h|--help)          usage; exit 0 ;;
    *)                  echo "unknown flag: $1"; usage; exit 1 ;;
  esac
  shift
done

if [ "$ANY_STAGE" = "false" ]; then
  DO_TRAIN=true; DO_EVAL=true
fi

DATA_DIR="results/fermionic_pipeline/regression/${DATA_TAG}"
DATA_PATH="${DATA_DIR}/regression_targets.h5"
SEEDS=(42 1729)

PARTITION="compute_full_node"
ACCOUNT="rrg-aspuru"

mkdir -p logs

echo "=== Regression v18 (orb-vs-R on v15_explicit arch; dataset=${DATA_TAG}, ω_floor=${OMEGA_OP_FLOOR}) ==="
echo "  tag:      ${TAG}"
echo "  stages:   train=${DO_TRAIN} eval=${DO_EVAL}"
echo "  data:     ${DATA_PATH}"
echo "  arch:     v15_explicit (explicit amplitude, amp_rank 16, NO residual trunk, NO clipping)"
[ -n "$EXCLUDE_NODES" ] && echo "  exclude:  ${EXCLUDE_NODES}"
echo ""

if [ ! -f "${DATA_PATH}" ]; then
  echo "ERROR: dataset missing at ${DATA_PATH}"
  echo "v18 reuses the v13 dt=0.05 dataset; run slurm/regression_v13.sh --data first."
  exit 1
fi

SUBMITTED=()

# v15_explicit architecture, input-mode-agnostic. NO --with_residual (no trunk).
# --grad_clip 1.0 is REQUIRED: the original v18 ran unclipped on the (now
# FALSIFIED) premise that the trunk-free explicit branch was stable unclipped.
# It is not — train_mse explodes either way (the orb arm collapsed: 0.323).
# v18b (orb_clip s42) proved grad_clip 1.0 restores the composition: envelope
# ~0.998, phase ~0.02 rad, overall Pearson 0.922 (≈ v15's 0.915). Note: clipping
# does NOT tame the train_mse magnitude (~1e4) — it bounds the per-step gradient,
# which is what actually protects the amp/freq composition. See 6/04 log.
COMMON_ARCH_FLAGS="--adaptive_bandwidth --omega_op_floor ${OMEGA_OP_FLOOR} --soft_omega_floor --explicit_amplitude --amp_rank 16 --grad_clip 1.0"

# Input-mode ablation. orb = chemically informed; R = geometry only.
declare -a VARIANTS=(
  "v18_orb:${COMMON_ARCH_FLAGS} --use_orb_features --standardize_orb_energies"
  "v18_R:${COMMON_ARCH_FLAGS}"
)

for VARIANT_SPEC in "${VARIANTS[@]}"; do
  VARIANT_NAME="${VARIANT_SPEC%%:*}"
  VARIANT_FLAGS="${VARIANT_SPEC#*:}"

  for SEED in "${SEEDS[@]}"; do
    SEED_TAG="${TAG}_${VARIANT_NAME}_s${SEED}"
    MODEL_DIR="results/fermionic_pipeline/regression/${SEED_TAG}_model"
    CKPT="${MODEL_DIR}/regressor.pt"
    JOB_TRAIN=""

    if [ "$DO_TRAIN" = "true" ]; then
      cat > "slurm/_train_${SEED_TAG}.sh" << 'EOF'
#!/bin/bash
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2
source "$HOME/envs/gqs/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
cd "$SCRATCH/generative-quantum-states"
export PYTHONUNBUFFERED=1
EOF
      cat >> "slurm/_train_${SEED_TAG}.sh" << EOF
python3 -m fermionic_pipeline.training.regressor_trainer \\
  --data_path ${DATA_PATH} \\
  --save_dir ${MODEL_DIR} \\
  --device cuda \\
  --seed ${SEED} \\
  --steps 150000 \\
  --batch_size 256 \\
  --lr 1e-3 \\
  --final_lr 1e-7 \\
  --warmup_frac 0.05 \\
  --weight_decay 5e-4 \\
  --d_hidden 768 \\
  --n_layers 6 \\
  --n_fourier 256 \\
  --fourier_scale 20.0 \\
  --conditioned_frequencies \\
  --freq_net_hidden 128 \\
  --freq_net_layers 3 \\
  ${VARIANT_FLAGS} \\
  --alpha_corr 1.0 \\
  --eval_every 2000
EOF
      chmod +x "slurm/_train_${SEED_TAG}.sh"

      SBATCH_ARGS_T=(
        --parsable
        --partition=${PARTITION}
        --job-name="reg-train-${SEED_TAG}"
        --output="logs/reg_train_${SEED_TAG}_%j.out"
        --error="logs/reg_train_${SEED_TAG}_%j.err"
        --time=01:30:00
        --gpus-per-node=4
        --cpus-per-task=4
        --account=${ACCOUNT}
      )
      [ -n "$EXCLUDE_NODES" ] && SBATCH_ARGS_T+=(--exclude="${EXCLUDE_NODES}")
      JOB_TRAIN=$(sbatch "${SBATCH_ARGS_T[@]}" "slurm/_train_${SEED_TAG}.sh")
      echo "[submitted] train ${VARIANT_NAME} seed=${SEED}: job ${JOB_TRAIN}"
      SUBMITTED+=("${VARIANT_NAME} seed=${SEED}: train=${JOB_TRAIN}")
    fi

    if [ "$DO_EVAL" = "true" ]; then
      if [ -z "$JOB_TRAIN" ] && [ ! -f "${CKPT}" ]; then
        echo "ERROR: checkpoint missing at ${CKPT}"
        echo "Train hasn't completed for ${SEED_TAG}; pass --train and --eval together."
        exit 1
      fi

      cat > "slurm/_eval_${SEED_TAG}.sh" << 'EOF'
#!/bin/bash
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2
source "$HOME/envs/gqs/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
cd "$SCRATCH/generative-quantum-states"
export PYTHONUNBUFFERED=1
EOF
      cat >> "slurm/_eval_${SEED_TAG}.sh" << EOF
python3 -m fermionic_pipeline.eval.regressor_eval \\
  --data_path ${DATA_PATH} \\
  --checkpoint ${CKPT} \\
  --save_dir ${MODEL_DIR}/eval \\
  --device cuda --ljung_box_p 0.06
python3 -m fermionic_pipeline.eval.plot_regression \\
  --data_path ${DATA_PATH} \\
  --checkpoint ${CKPT} \\
  --save_dir ${MODEL_DIR}/plots \\
  --device cuda --ljung_box_p 0.06
python3 -m fermionic_pipeline.eval.composition_diagnostic \\
  --data_path ${DATA_PATH} \\
  --checkpoint ${CKPT} \\
  --save_dir ${MODEL_DIR}/eval \\
  --device cuda
EOF
      chmod +x "slurm/_eval_${SEED_TAG}.sh"

      SBATCH_ARGS_E=(
        --parsable
        --partition=${PARTITION}
        --job-name="reg-eval-${SEED_TAG}"
        --output="logs/reg_eval_${SEED_TAG}_%j.out"
        --error="logs/reg_eval_${SEED_TAG}_%j.err"
        --time=04:00:00
        --gpus-per-node=4
        --cpus-per-task=4
        --account=${ACCOUNT}
      )
      [ -n "$EXCLUDE_NODES" ] && SBATCH_ARGS_E+=(--exclude="${EXCLUDE_NODES}")
      if [ -n "$JOB_TRAIN" ]; then
        SBATCH_ARGS_E+=(--dependency=afterok:${JOB_TRAIN})
      fi
      JOB_EVAL=$(sbatch "${SBATCH_ARGS_E[@]}" "slurm/_eval_${SEED_TAG}.sh")
      DEP_NOTE=""
      [ -n "$JOB_TRAIN" ] && DEP_NOTE=" (afterok ${JOB_TRAIN})"
      echo "[submitted] eval  ${VARIANT_NAME} seed=${SEED}: job ${JOB_EVAL}${DEP_NOTE}"
      SUBMITTED+=("${VARIANT_NAME} seed=${SEED}: eval=${JOB_EVAL}")
    fi
  done
done

if [ "${#SUBMITTED[@]}" -gt 0 ]; then
  echo ""
  echo "=== Submitted ==="
  for line in "${SUBMITTED[@]}"; do
    echo "  ${line}"
  done
fi
