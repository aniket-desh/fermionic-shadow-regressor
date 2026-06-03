#!/bin/bash
# =======================================================================
# Regression v12: adaptive bandwidth + ω_op floor (long-R fix on top of v11).
#
# v11 catastrophically regressed long-R (pearson 0.99 → 0.62 in [1.5, 2.0],
# 0.99 → 0.67 in [2.0, 3.0]) because ω_op(R) at long R is ~0.86 E_h —
# packing all 64 Fourier slots into [0, 0.86] left no basis to represent
# the residual 1–5% of signal mass above that boundary, which v10's static
# basis was successfully fitting. v11 borderline [0.74, 1.0) zone still
# improved as predicted (0.39 → 0.72 / 0.60), so the W₁ capacity diagnosis
# was correct for short R and wrong for long R.
#
# v12 fix: floor the operational band so the slot-spread is always at
# least ω_floor wide regardless of geometry.
#   ω_max(R) = max(ω_op(R), ω_floor)
#   ω_k(R)   = ω_max(R) · sigmoid(freq_net(ε(R)))_k
# At short R, ω_op ≈ 7 dominates the floor → v11 short-R win preserved.
# At long R, ω_floor dominates → slots spread across [0, floor], can
# represent the residual high-freq peaks v10 was getting right.
#
# Default ω_floor = 5.0 E_h. Sweepable via --omega_op_floor.
#
# Stages (composable; default = all three):
#   --data     copy v11 dataset (already has omega_op) into v12 dir
#   --train    submit training jobs (one per seed)
#   --eval     submit eval+plot jobs (one per seed)
#
# Usage:
#   bash slurm/regression_v12.sh                            # full pipeline
#   bash slurm/regression_v12.sh --eval                     # eval-only on existing checkpoints
#   bash slurm/regression_v12.sh --omega_op_floor 3.0 --train --eval
#   bash slurm/regression_v12.sh --tag h4_regress_v12_f3 --omega_op_floor 3.0
# =======================================================================
set -euo pipefail

TAG="h4_regress_v12"
V11_TAG="h4_regress_v11"
OMEGA_OP_FLOOR="5.0"
DO_DATA=false; DO_TRAIN=false; DO_EVAL=false
ANY_STAGE=false

usage() {
  cat << EOF
Usage: $0 [--data] [--train] [--eval] [--all] [--tag NAME] [--v11-tag NAME] [--omega_op_floor F]

Stages (composable; default = all three):
  --data     copy v11 dataset (with omega_op) into v12 dir
  --train    submit training jobs (one per seed)
  --eval     submit eval+plot jobs (one per seed)
  --all      explicit form of "all three"

Tag overrides:
  --tag NAME             v12 dataset/model tag prefix (default: h4_regress_v12)
  --v11-tag NAME         v11 dataset to copy from (default: h4_regress_v11)
  --omega_op_floor F     ω_floor in E_h (default: 5.0). Set to 0 for v11 behavior.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --data)             DO_DATA=true; ANY_STAGE=true ;;
    --train)            DO_TRAIN=true; ANY_STAGE=true ;;
    --eval)             DO_EVAL=true; ANY_STAGE=true ;;
    --all)              DO_DATA=true; DO_TRAIN=true; DO_EVAL=true; ANY_STAGE=true ;;
    --tag)              TAG="$2"; shift ;;
    --v11-tag)          V11_TAG="$2"; shift ;;
    --omega_op_floor)   OMEGA_OP_FLOOR="$2"; shift ;;
    -h|--help)          usage; exit 0 ;;
    *)                  echo "unknown flag: $1"; usage; exit 1 ;;
  esac
  shift
done

if [ "$ANY_STAGE" = "false" ]; then
  DO_DATA=true; DO_TRAIN=true; DO_EVAL=true
fi

V11_DATA="results/fermionic_pipeline/regression/${V11_TAG}/regression_targets.h5"
DATA_DIR="results/fermionic_pipeline/regression/${TAG}"
DATA_PATH="${DATA_DIR}/regression_targets.h5"
SEEDS=(42 1729)

PARTITION="compute_full_node"
ACCOUNT="rrg-aspuru"

mkdir -p logs

echo "=== Regression v12 (adaptive bandwidth + ω_op floor=${OMEGA_OP_FLOOR}) ==="
echo "  tag:      ${TAG}"
echo "  stages:   data=${DO_DATA} train=${DO_TRAIN} eval=${DO_EVAL}"
echo "  data:     ${DATA_PATH}"
echo "  ω_floor:  ${OMEGA_OP_FLOOR} E_h"
echo ""

# ── Stage: data (copy v11 dataset; omega_op already precomputed there) ──
if [ "$DO_DATA" = "true" ]; then
  if [ ! -f "${V11_DATA}" ]; then
    echo "ERROR: v11 dataset missing at ${V11_DATA}"
    echo "Run slurm/regression_v11.sh --data first to build v11 dataset (with omega_op)."
    exit 1
  fi
  mkdir -p "${DATA_DIR}"
  if [ ! -f "${DATA_PATH}" ]; then
    echo "[copy] v11 -> v12 dataset"
    cp "${V11_DATA}" "${DATA_PATH}"
  else
    echo "[skip] v12 dataset already exists at ${DATA_PATH}"
  fi
  echo ""
fi

if [ "$DO_TRAIN" = "true" ] || [ "$DO_EVAL" = "true" ]; then
  if [ ! -f "${DATA_PATH}" ]; then
    echo "ERROR: v12 dataset missing at ${DATA_PATH}"
    echo "Pass --data first."
    exit 1
  fi
fi

SUBMITTED=()
for SEED in "${SEEDS[@]}"; do
  SEED_TAG="${TAG}_s${SEED}"
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
  --adaptive_bandwidth \\
  --omega_op_floor ${OMEGA_OP_FLOOR} \\
  --alpha_corr 1.0 \\
  --eval_every 2000
EOF
    chmod +x "slurm/_train_${SEED_TAG}.sh"

    JOB_TRAIN=$(sbatch --parsable \
      --partition=${PARTITION} \
      --job-name="reg-train-${SEED_TAG}" \
      --output="logs/reg_train_${SEED_TAG}_%j.out" \
      --error="logs/reg_train_${SEED_TAG}_%j.err" \
      --time=24:00:00 \
      --gpus-per-node=4 \
      --cpus-per-task=4 \
      --account=${ACCOUNT} \
      "slurm/_train_${SEED_TAG}.sh")
    echo "[submitted] train (seed=${SEED}): job ${JOB_TRAIN}"
    SUBMITTED+=("seed=${SEED}: train=${JOB_TRAIN}")
  fi

  if [ "$DO_EVAL" = "true" ]; then
    if [ -z "$JOB_TRAIN" ] && [ ! -f "${CKPT}" ]; then
      echo "ERROR: checkpoint missing at ${CKPT}"
      echo "Train hasn't completed for seed=${SEED}; pass --train (or --train --eval together)."
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

    SBATCH_ARGS=(
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
      SBATCH_ARGS+=(--dependency=afterok:${JOB_TRAIN})
    fi

    JOB_EVAL=$(sbatch "${SBATCH_ARGS[@]}" "slurm/_eval_${SEED_TAG}.sh")
    if [ -n "$JOB_TRAIN" ]; then
      echo "[submitted] eval  (seed=${SEED}): job ${JOB_EVAL} (after ${JOB_TRAIN})"
      SUBMITTED+=("seed=${SEED}: eval=${JOB_EVAL} (afterok ${JOB_TRAIN})")
    else
      echo "[submitted] eval  (seed=${SEED}): job ${JOB_EVAL}"
      SUBMITTED+=("seed=${SEED}: eval=${JOB_EVAL}")
    fi
  fi
done

if [ "${#SUBMITTED[@]}" -gt 0 ]; then
  echo ""
  echo "=== Submitted ==="
  for line in "${SUBMITTED[@]}"; do
    echo "  ${line}"
  done
fi
