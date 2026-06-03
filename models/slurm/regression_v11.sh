#!/bin/bash
# =======================================================================
# Regression v11: adaptive bandwidth, 2-seed protocol.
#
# Stages (composable; default = all three with dependency chain):
#   --data     build v11 dataset (copy v10 + add omega_op field)
#   --train    submit training jobs (one per seed, 24h)
#   --eval     submit eval+plot jobs (one per seed, ~minutes)
#   --all      explicit alias for all three
#
# When --train and --eval are both passed, eval has afterok dependency
# on train. With --eval alone, no dependency — runs against the existing
# checkpoints in place (use this after a code-side eval fix without
# overwriting prior training).
#
# Architecture change from v10:
#   freq_net output → sigmoid(...) → multiply by omega_op(R)
#   ω_k(R) = ω_op(R) · sigmoid(freq_net(ε(R)))_k    ∈ [0, ω_op(R)]
# Forces all K Fourier slots into the operational band per R; replaces the
# v10 static log-uniform [0.05, 20] init that wasted ~80/120 slots above
# 7 E_h at short R (W₁ wasted-capacity diagnosis from 4/25).
#
# Usage (on Trillium):
#   cd $SCRATCH/generative-quantum-states
#   bash slurm/regression_v11.sh                   # full pipeline (data → train → eval)
#   bash slurm/regression_v11.sh --eval            # eval only on existing checkpoints
#   bash slurm/regression_v11.sh --train --eval    # train + eval, skip data prep
#   bash slurm/regression_v11.sh --data            # only build dataset
#   bash slurm/regression_v11.sh --tag h4_regress_v11_v2 --train --eval
# =======================================================================
set -euo pipefail

TAG="h4_regress_v11"
V10_TAG="h4_regress_v10"
DO_DATA=false; DO_TRAIN=false; DO_EVAL=false
ANY_STAGE=false

usage() {
  cat << EOF
Usage: $0 [--data] [--train] [--eval] [--all] [--tag NAME] [--v10-tag NAME]

Stages (composable; default = all three):
  --data     build v11 dataset (copy v10 + add omega_op field)
  --train    submit training jobs (one per seed)
  --eval     submit eval+plot jobs (one per seed)
  --all      explicit form of "all three"

Tag overrides:
  --tag NAME       v11 dataset/model tag prefix (default: h4_regress_v11)
  --v10-tag NAME   v10 dataset tag to copy from (default: h4_regress_v10)

When --train and --eval are both submitted in one run, eval has an
afterok dependency on train. With --eval alone, no dependency — uses
the existing checkpoints in place.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --data)    DO_DATA=true; ANY_STAGE=true ;;
    --train)   DO_TRAIN=true; ANY_STAGE=true ;;
    --eval)    DO_EVAL=true; ANY_STAGE=true ;;
    --all)     DO_DATA=true; DO_TRAIN=true; DO_EVAL=true; ANY_STAGE=true ;;
    --tag)     TAG="$2"; shift ;;
    --v10-tag) V10_TAG="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *)         echo "unknown flag: $1"; usage; exit 1 ;;
  esac
  shift
done

if [ "$ANY_STAGE" = "false" ]; then
  DO_DATA=true; DO_TRAIN=true; DO_EVAL=true
fi

V10_DATA="results/fermionic_pipeline/regression/${V10_TAG}/regression_targets.h5"
DATA_DIR="results/fermionic_pipeline/regression/${TAG}"
DATA_PATH="${DATA_DIR}/regression_targets.h5"
SEEDS=(42 1729)

PARTITION="compute_full_node"
ACCOUNT="rrg-aspuru"

mkdir -p logs

echo "=== Regression v11 (adaptive bandwidth, 2-seed) ==="
echo "  tag:      ${TAG}"
echo "  stages:   data=${DO_DATA} train=${DO_TRAIN} eval=${DO_EVAL}"
echo "  v11 data: ${DATA_PATH}"
echo ""

# ── Stage: data ────────────────────────────────────────────────────
if [ "$DO_DATA" = "true" ]; then
  if [ ! -f "${V10_DATA}" ]; then
    echo "ERROR: v10 dataset missing at ${V10_DATA}"
    echo "Run slurm/regression_v10.sh first."
    exit 1
  fi
  mkdir -p "${DATA_DIR}"
  if [ ! -f "${DATA_PATH}" ]; then
    echo "[copy] v10 -> v11 dataset"
    cp "${V10_DATA}" "${DATA_PATH}"
  fi
  # omega_op precompute is cheap (~10 s for 251 R × 1500 t × 120 obs); run
  # in the login shell rather than queuing a separate slurm step.
  module load StdEnv/2023 python/3.11 cuda/12.2 || true
  source "$HOME/envs/gqs/bin/activate"
  export PYTHONNOUSERSITE=1
  unset PYTHONPATH
  echo "[precompute] omega_op(R) field"
  python3 -m fermionic_pipeline.data.compute_omega_op \
    --data_path "${DATA_PATH}" --frac 0.99
  echo ""
fi

# ── Pre-flight for train/eval ──────────────────────────────────────
if [ "$DO_TRAIN" = "true" ] || [ "$DO_EVAL" = "true" ]; then
  if [ ! -f "${DATA_PATH}" ]; then
    echo "ERROR: v11 dataset missing at ${DATA_PATH}"
    echo "Pass --data first (or pull from a prior datagen)."
    exit 1
  fi
fi

# ── Stage: train and/or eval, per seed ─────────────────────────────
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
