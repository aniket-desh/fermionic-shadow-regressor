#!/bin/bash
# =======================================================================
# Regression pipeline v2: scaled up with 51 geometries (h4.yaml grid)
#
# Usage (on Trillium):
#   cd $SCRATCH/generative-quantum-states
#   bash slurm/regression_v2.sh [TAG]
#
# Default TAG: h4_regress_v2
# =======================================================================
set -euo pipefail

TAG="${1:-h4_regress_v2}"

CONFIG="fermionic_pipeline/configs/h4.yaml"
DATA_DIR="results/fermionic_pipeline/regression/${TAG}"
DATA_PATH="${DATA_DIR}/regression_targets.h5"
MODEL_DIR="results/fermionic_pipeline/regression/${TAG}_model"

# 51 geometries (r_step=0.05), 500 time points, 1000 Q's
# With precomputed decompositions + 8 workers: datagen ~30-60 min
N_TIMES=500
T_MAX=100.0
N_Q=1000
N_WORKERS=8

PARTITION="compute_full_node"
ACCOUNT="rrg-aspuru"

PREAMBLE='
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2 scipy-stack/2024a
source "$HOME/envs/gqs/bin/activate"
cd "$SCRATCH/generative-quantum-states"
export PYTHONUNBUFFERED=1
'

echo "=== Regression Pipeline v2 (scaled) ==="
echo "  tag:       ${TAG}"
echo "  config:    ${CONFIG} (51 geometries)"
echo "  n_times:   ${N_TIMES}"
echo "  t_max:     ${T_MAX}"
echo "  n_q:       ${N_Q}"
echo "  n_workers: ${N_WORKERS}"
echo ""

# ── Job 1: Generate regression targets ───────────────────────────
JOB_DATA=$(sbatch --parsable \
  --partition=${PARTITION} \
  --job-name="reg-gen-${TAG}" \
  --output="logs/reg_gen_${TAG}_%j.out" \
  --error="logs/reg_gen_${TAG}_%j.err" \
  --time=24:00:00 \
  --gpus-per-node=4 \
  --cpus-per-task=8 \
  --account=${ACCOUNT} \
  --wrap="${PREAMBLE}
mkdir -p ${DATA_DIR}
python3 -m fermionic_pipeline.data.regression_dataset \\
  --config ${CONFIG} \\
  --output ${DATA_PATH} \\
  --n_times ${N_TIMES} \\
  --t_max ${T_MAX} \\
  --n_q ${N_Q} \\
  --n_workers ${N_WORKERS}
")
echo "[submitted] datagen: job ${JOB_DATA}"

# ── Job 2: Train regressor ───────────────────────────────────────
JOB_TRAIN=$(sbatch --parsable \
  --dependency=afterok:${JOB_DATA} \
  --partition=${PARTITION} \
  --job-name="reg-train-${TAG}" \
  --output="logs/reg_train_${TAG}_%j.out" \
  --error="logs/reg_train_${TAG}_%j.err" \
  --time=24:00:00 \
  --gpus-per-node=4 \
  --cpus-per-task=4 \
  --account=${ACCOUNT} \
  --wrap="${PREAMBLE}
python3 -m fermionic_pipeline.training.regressor_trainer \\
  --data_path ${DATA_PATH} \\
  --save_dir ${MODEL_DIR} \\
  --device cuda \\
  --steps 100000 \\
  --batch_size 256 \\
  --lr 1e-3 \\
  --final_lr 1e-7 \\
  --warmup_frac 0.05 \\
  --weight_decay 1e-4 \\
  --n_fourier 64 \\
  --fourier_scale 10.0 \\
  --eval_every 2000
")
echo "[submitted] train:   job ${JOB_TRAIN} (after ${JOB_DATA})"

# ── Job 3: Evaluate ──────────────────────────────────────────────
JOB_EVAL=$(sbatch --parsable \
  --dependency=afterok:${JOB_TRAIN} \
  --partition=${PARTITION} \
  --job-name="reg-eval-${TAG}" \
  --output="logs/reg_eval_${TAG}_%j.out" \
  --error="logs/reg_eval_${TAG}_%j.err" \
  --time=24:00:00 \
  --gpus-per-node=4 \
  --cpus-per-task=4 \
  --account=${ACCOUNT} \
  --wrap="${PREAMBLE}
python3 -m fermionic_pipeline.eval.regressor_eval \\
  --data_path ${DATA_PATH} \\
  --checkpoint ${MODEL_DIR}/regressor.pt \\
  --save_dir ${MODEL_DIR}/eval \\
  --device cuda \\
  --ljung_box_p 0.06
")
echo "[submitted] eval:    job ${JOB_EVAL} (after ${JOB_TRAIN})"

echo ""
echo "=== Pipeline submitted ==="
echo "  datagen: ${JOB_DATA} → train: ${JOB_TRAIN} → eval: ${JOB_EVAL}"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: \$SCRATCH/generative-quantum-states/${MODEL_DIR}/eval/"
