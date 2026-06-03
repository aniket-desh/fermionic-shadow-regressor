#!/bin/bash
# =======================================================================
# Regression v14: target the [0.74, 1.0) composition failure on the
# existing dt=0.05 dataset (h4_regress_v13). Two variants × two seeds.
#
# v13 readout (5/07): v12f8 phase err 1.41 rad in [0.74, 1.0), v11 0.66 rad.
# Composition_diagnostic flagged this as the dominant failure mode for both
# architectures. Bandwidth fixes (R-conditioned floor, queued) cannot move
# phase composition directly. v14 addresses two upstream issues:
#
#   1. The training Pearson loss (--alpha_corr) operates on the cross-
#      observable axis at fixed (R, t), not the eval-time per-observable
#      temporal axis. With v13 using --alpha_corr 1.0, the optimizer was
#      pushing on the wrong objective. New --alpha_temporal_corr matches
#      the eval metric.
#
#   2. The trunk MLP must internally synthesize per-observable, R-dependent
#      linear combinations of {sin(ωt), cos(ωt)} — multiplicative
#      interactions are hard for additive GELU layers. New
#      --explicit_amplitude makes the composition explicit:
#        y_μ(R,t) = Σ_k a_kμ(R) cos(ω_k(R) t) + b_kμ(R) sin(ω_k(R) t) + dc_μ(R)
#      with low-rank a_kμ = U_kr · V_rμ.
#
# Variants:
#   v14_tcorr     — v12f8 + loss-axis fix only (--alpha_corr 0
#                   --alpha_temporal_corr 1.0). Tests whether the wrong
#                   training objective alone caused the composition failure.
#                   Minimal change.
#   v14_explicit  — v14_tcorr + full architectural stack
#                   (--explicit_amplitude --amp_rank 16
#                    --soft_omega_floor --standardize_orb_energies).
#                   Tests the explicit linear-in-amplitude factorization.
#
# Reading v14 results (per-bin Pearson + phase_err_mean, n=100 seed-pooled):
#   - v14_tcorr lifts [0.74, 1.0) ≥ 0.7 P, ≤ 0.5 rad phase err:
#       loss-axis was the root cause; ship --alpha_temporal_corr 1.0.
#   - v14_tcorr flat, v14_explicit lifts: composition is the root cause;
#       ship explicit-amplitude factorization.
#   - Both flat: neither loss nor representation; pivot to amplitude-branch
#       / CIS-CASCI feature directions (4/27 fallback).
#   - v14_explicit regresses long R below v12f8 0.97: trunk had useful
#       extra-Fourier capacity; revisit factorization with a FAN-style
#       residual branch.
#
# Stages (composable; default = train + eval):
#   --train  submit training jobs (4 = 2 variants × 2 seeds)
#   --eval   submit eval+plot+composition jobs (1 per checkpoint)
#
# Usage:
#   bash slurm/regression_v14.sh                # train + eval
#   bash slurm/regression_v14.sh --eval         # eval-only (existing ckpts)
#   bash slurm/regression_v14.sh --tag h4_regress_v14_alt
# =======================================================================
set -euo pipefail

TAG="h4_regress_v14"
DATA_TAG="h4_regress_v13"   # reuse the dt=0.05 canonical dataset
OMEGA_OP_FLOOR="8.0"
DO_TRAIN=false; DO_EVAL=false
ANY_STAGE=false

usage() {
  cat << EOF
Usage: $0 [--train] [--eval] [--all] [--tag NAME] [--data_tag NAME] [--omega_op_floor F]

Stages (composable; default = train + eval):
  --train     submit training jobs (v14_tcorr × 2 seeds + v14_explicit × 2 seeds)
  --eval      submit eval+plot+composition jobs (one per checkpoint)
  --all       explicit form of "train + eval"

Tag overrides:
  --tag NAME             model tag prefix (default: h4_regress_v14)
  --data_tag NAME        upstream dataset tag (default: h4_regress_v13)
  --omega_op_floor F     ω_floor in E_h for the v12 architecture (default: 8.0)
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

echo "=== Regression v14 (composition fix; dataset=${DATA_TAG}, ω_floor=${OMEGA_OP_FLOOR}) ==="
echo "  tag:      ${TAG}"
echo "  stages:   train=${DO_TRAIN} eval=${DO_EVAL}"
echo "  data:     ${DATA_PATH}"
echo ""

if [ ! -f "${DATA_PATH}" ]; then
  echo "ERROR: dataset missing at ${DATA_PATH}"
  echo "v14 reuses the v13 dt=0.05 dataset; run slurm/regression_v13.sh --data first."
  exit 1
fi

SUBMITTED=()

# Two model variants on the same dataset:
#   v14_tcorr:    v12f8 + loss-axis fix (alpha_corr → alpha_temporal_corr)
#   v14_explicit: v14_tcorr + explicit-amplitude factorization + soft floor + orb std
#
# Both variants disable --alpha_corr (cross-obs Pearson) and enable
# --alpha_temporal_corr (per-obs temporal Pearson). Architecture-only
# differences live in VARIANT_FLAGS.
COMMON_LOSS_FLAGS="--alpha_corr 0 --alpha_temporal_corr 1.0 --temporal_corr_every 10 --temporal_corr_n_geom 2"
COMMON_ARCH_FLAGS="--adaptive_bandwidth --omega_op_floor ${OMEGA_OP_FLOOR}"

declare -a VARIANTS=(
  "v14_tcorr:${COMMON_ARCH_FLAGS}"
  "v14_explicit:${COMMON_ARCH_FLAGS} --soft_omega_floor --standardize_orb_energies --explicit_amplitude --amp_rank 16"
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
  --use_orb_features \\
  ${VARIANT_FLAGS} \\
  ${COMMON_LOSS_FLAGS} \\
  --eval_every 2000
EOF
      chmod +x "slurm/_train_${SEED_TAG}.sh"

      SBATCH_ARGS_T=(
        --parsable
        --partition=${PARTITION}
        --job-name="reg-train-${SEED_TAG}"
        --output="logs/reg_train_${SEED_TAG}_%j.out"
        --error="logs/reg_train_${SEED_TAG}_%j.err"
        --time=24:00:00
        --gpus-per-node=4
        --cpus-per-task=4
        --account=${ACCOUNT}
      )
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
