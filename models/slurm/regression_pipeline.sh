#!/bin/bash
# =======================================================================
# Direct observable regression pipeline: datagen → train → eval
#
# Usage (from Trillium login node):
#   bash slurm/regression_pipeline.sh [TAG]
#
# If the exact conditional HDF5 already exists (from the classifier
# pipeline), pass its path via EXACT_COND_PATH to skip matchgate
# recomputation and reuse the Q library + stored probabilities.
#
# Default TAG: h4_regress_v1
# =======================================================================
set -euo pipefail

TAG="${1:-h4_regress_v1}"

# ── Config ───────────────────────────────────────────────────────
CONFIG="fermionic_pipeline/configs/h4_fast.yaml"
DATA_DIR="results/fermionic_pipeline/regression/${TAG}"
DATA_PATH="${DATA_DIR}/regression_targets.h5"
MODEL_DIR="results/fermionic_pipeline/regression/${TAG}_model"

# If the exact conditional HDF5 exists, reuse it (much faster datagen)
EXACT_COND_PATH="${EXACT_COND_PATH:-results/fermionic_pipeline/exact_conditional/h4_exact_v1/exact_conditionals.h5}"

# Time grid overrides (same as classifier pipeline for comparability)
N_TIMES=500
T_MAX=100.0
N_Q=1000  # more Q's for better coverage (no sampling cost, just marginals)

PARTITION="compute_full_node"

PREAMBLE='
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2 scipy-stack/2024a
source "$HOME/envs/gqs/bin/activate"
cd "$SCRATCH/generative-quantum-states"
export PYTHONUNBUFFERED=1
'

echo "=== Observable Regression Pipeline ==="
echo "  tag:            ${TAG}"
echo "  config:         ${CONFIG}"
echo "  n_times:        ${N_TIMES}"
echo "  t_max:          ${T_MAX}"
echo "  n_q:            ${N_Q}"
echo "  exact_cond:     ${EXACT_COND_PATH}"
echo ""

# ── Job 1: Generate regression targets ───────────────────────────
# If exact conditional HDF5 exists, pass --exact_conditional to reuse
# stored p(b|Q,R,t) and skip matchgate computation entirely.
DATAGEN_EXTRA=""
if [ -f "$SCRATCH/generative-quantum-states/${EXACT_COND_PATH}" ]; then
    DATAGEN_EXTRA="--exact_conditional ${EXACT_COND_PATH}"
    echo "[info] will reuse exact conditional HDF5"
else
    echo "[info] will generate from scratch (no exact conditional found)"
fi

JOB_DATA=$(sbatch --parsable \
  --partition=${PARTITION} \
  --job-name="reg-gen-${TAG}" \
  --output="logs/reg_gen_${TAG}_%j.out" \
  --error="logs/reg_gen_${TAG}_%j.err" \
  --time=16:00:00 \
  --gpus-per-node=4 \
  --cpus-per-task=8 \
  --account=rrg-aspuru \
  --wrap="${PREAMBLE}
mkdir -p ${DATA_DIR}
python3 -m fermionic_pipeline.data.regression_dataset \\
  --config ${CONFIG} \\
  --output ${DATA_PATH} \\
  --n_times ${N_TIMES} \\
  --t_max ${T_MAX} \\
  --n_q ${N_Q} \\
  ${DATAGEN_EXTRA}
")
echo "[submitted] datagen:  job ${JOB_DATA}"

# ── Job 2: Train regressor ───────────────────────────────────────
JOB_TRAIN=$(sbatch --parsable \
  --dependency=afterok:${JOB_DATA} \
  --partition=${PARTITION} \
  --job-name="reg-train-${TAG}" \
  --output="logs/reg_train_${TAG}_%j.out" \
  --error="logs/reg_train_${TAG}_%j.err" \
  --time=16:00:00 \
  --gpus-per-node=4 \
  --cpus-per-task=4 \
  --account=rrg-aspuru \
  --wrap="${PREAMBLE}
python3 -m fermionic_pipeline.training.regressor_trainer \\
  --data_path ${DATA_PATH} \\
  --save_dir ${MODEL_DIR} \\
  --device cuda \\
  --steps 50000 \\
  --batch_size 256 \\
  --lr 1e-3 \\
  --final_lr 1e-7 \\
  --warmup_frac 0.05 \\
  --weight_decay 1e-4 \\
  --n_fourier 64 \\
  --fourier_scale 10.0
")
echo "[submitted] train:    job ${JOB_TRAIN} (after ${JOB_DATA})"

# ── Job 3: Evaluate ──────────────────────────────────────────────
JOB_EVAL=$(sbatch --parsable \
  --dependency=afterok:${JOB_TRAIN} \
  --partition=${PARTITION} \
  --job-name="reg-eval-${TAG}" \
  --output="logs/reg_eval_${TAG}_%j.out" \
  --error="logs/reg_eval_${TAG}_%j.err" \
  --time=16:00:00 \
  --gpus-per-node=4 \
  --cpus-per-task=4 \
  --account=rrg-aspuru \
  --wrap="${PREAMBLE}
python3 -m fermionic_pipeline.eval.regressor_eval \\
  --data_path ${DATA_PATH} \\
  --checkpoint ${MODEL_DIR}/regressor.pt \\
  --save_dir ${MODEL_DIR}/eval \\
  --device cuda \\
  --ljung_box_p 0.06
")
echo "[submitted] eval:     job ${JOB_EVAL} (after ${JOB_TRAIN})"

echo ""
echo "=== Pipeline submitted ==="
echo "  datagen: ${JOB_DATA}"
echo "  train:   ${JOB_TRAIN} → eval: ${JOB_EVAL}"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: \$SCRATCH/generative-quantum-states/${MODEL_DIR}/eval/"
