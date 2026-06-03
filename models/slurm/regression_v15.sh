#!/bin/bash
# =======================================================================
# Regression v15: isolate the architectural changes from v14's loss-axis
# regression. Two variants × two seeds. Reuses the dt=0.05 dataset
# (h4_regress_v13). 1h walltime (v14 ran in 17-22 min).
#
# Why v15 exists (5/10 log entry):
#   v14 swapped --alpha_corr 1.0 → --alpha_temporal_corr 1.0 with sparse
#   2-R-point sampling, expecting the eval-aligned loss to fix [0.74, 1.0)
#   composition. Result: catastrophic regression. Overall Pearson dropped
#   0.87 → 0.16; range_ratio collapsed to 0.0-0.5; v14_tcorr_s1729 hit
#   tcorr=1.0000 on training R-points but produced CONSTANT predictions on
#   test R-points. Three compounding causes:
#     (a) _temporal_corr_loss is scale-invariant (Pearson divides amplitude),
#         doesn't penalize collapse to small-magnitude prediction
#     (b) --temporal_corr_n_geom 2 → sparse spatial coverage, model
#         overfits the few sampled R-points
#     (c) Removing --alpha_corr 1.0 removed the load-bearing uniform per-batch
#         regularizer. v13's cross-obs Pearson at fixed (R,t) was on the
#         "wrong axis" relative to the eval metric but kept the model
#         honest about per-(R,t) structure on every R-point in every batch.
#
#   Conclusion: the architectural changes (--soft_omega_floor,
#   --standardize_orb_energies, --explicit_amplitude) got an unfair test in
#   v14 because they were entangled with the loss-axis regression. v15 keeps
#   v13's loss exactly and tests architecture/normalization in isolation.
#
# Variants:
#   v15_v12f8_plus  v13 v12f8 + --soft_omega_floor --standardize_orb_energies.
#                   Cheap small-fix stack. Tests whether the clamp kink
#                   (R* where ω_op crosses 8, geometrically inside [0.74, 1.0))
#                   plus orb-energy normalization is enough.
#   v15_explicit    v15_v12f8_plus + --explicit_amplitude --amp_rank 16.
#                   Tests the linear-in-amplitude factorization on top.
#
# Decision rules (per-bin Pearson + composition_diagnostic phase_err, n=100):
#   - v15_v12f8_plus lifts [0.74, 1.0) ≥ 0.7 P, ≤ 0.5 rad phase err:
#       small fixes alone suffice; ship as new baseline.
#   - v15_v12f8_plus flat, v15_explicit lifts: composition is the bottleneck;
#       ship explicit factorization.
#   - v15_explicit regresses long R below 0.97: GELU trunk was carrying
#       useful extra-Fourier capacity; revisit with FAN-style residual branch.
#   - Both flat: architecture isn't the bottleneck. Pivot to amplitude-branch
#       / CIS-CASCI feature directions (4/27 fallback note).
#
# Stages (composable; default = train + eval):
#   --train  submit training jobs (4 = 2 variants × 2 seeds)
#   --eval   submit eval+plot+composition jobs (1 per checkpoint)
#
# Usage:
#   bash slurm/regression_v15.sh                # train + eval
#   bash slurm/regression_v15.sh --eval         # eval-only on existing ckpts
#   bash slurm/regression_v15.sh --tag h4_regress_v15_alt
# =======================================================================
set -euo pipefail

TAG="h4_regress_v15"
DATA_TAG="h4_regress_v13"   # reuse the dt=0.05 canonical dataset
OMEGA_OP_FLOOR="8.0"
DO_TRAIN=false; DO_EVAL=false
ANY_STAGE=false

usage() {
  cat << EOF
Usage: $0 [--train] [--eval] [--all] [--tag NAME] [--data_tag NAME] [--omega_op_floor F]

Stages (composable; default = train + eval):
  --train     submit training jobs (v15_v12f8_plus × 2 seeds + v15_explicit × 2 seeds)
  --eval      submit eval+plot+composition jobs (one per checkpoint)
  --all       explicit form of "train + eval"

Tag overrides:
  --tag NAME             model tag prefix (default: h4_regress_v15)
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

echo "=== Regression v15 (architecture-only fix; dataset=${DATA_TAG}, ω_floor=${OMEGA_OP_FLOOR}) ==="
echo "  tag:      ${TAG}"
echo "  stages:   train=${DO_TRAIN} eval=${DO_EVAL}"
echo "  data:     ${DATA_PATH}"
echo ""

if [ ! -f "${DATA_PATH}" ]; then
  echo "ERROR: dataset missing at ${DATA_PATH}"
  echo "v15 reuses the v13 dt=0.05 dataset; run slurm/regression_v13.sh --data first."
  exit 1
fi

SUBMITTED=()

# Two architecture variants on v13's loss (--alpha_corr 1.0, NO temporal_corr).
# v15_v12f8_plus  : v13 v12f8 + soft floor + orb standardization
# v15_explicit    : v15_v12f8_plus + explicit-amplitude factorization (rank 16)
COMMON_ARCH_FLAGS="--adaptive_bandwidth --omega_op_floor ${OMEGA_OP_FLOOR} --soft_omega_floor --standardize_orb_energies"

declare -a VARIANTS=(
  "v15_v12f8_plus:${COMMON_ARCH_FLAGS}"
  "v15_explicit:${COMMON_ARCH_FLAGS} --explicit_amplitude --amp_rank 16"
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
        --time=01:00:00
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
