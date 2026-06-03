#!/bin/bash
# =======================================================================
# Regression pipeline v3: bigger model + more Fourier features
#
# Changes from v2:
#   - n_fourier: 64 → 128 (better high-frequency coverage)
#   - fourier_scale: 10 → 15 (wider frequency range)
#   - d_hidden: 256 → 512 (more capacity)
#   - n_layers: 3 → 4
#   - alpha_corr: 0 → 1.0 (Pearson correlation loss for sign structure)
#   - steps: 100k → 200k
#   - Reuses v2 dataset (same targets, just bigger model)
#
# Usage (on Trillium):
#   cd $SCRATCH/generative-quantum-states
#   bash slurm/regression_v3.sh [TAG]
# =======================================================================
set -euo pipefail

TAG="${1:-h4_regress_v3}"

# Reuse v2 dataset — no need to regenerate
DATA_PATH="results/fermionic_pipeline/regression/h4_regress_v2/regression_targets.h5"
MODEL_DIR="results/fermionic_pipeline/regression/${TAG}_model"

PARTITION="compute_full_node"
ACCOUNT="rrg-aspuru"

PREAMBLE='
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2 scipy-stack/2024a
source "$HOME/envs/gqs/bin/activate"
cd "$SCRATCH/generative-quantum-states"
export PYTHONUNBUFFERED=1
'

echo "=== Regression Pipeline v3 (bigger model) ==="
echo "  tag:      ${TAG}"
echo "  dataset:  ${DATA_PATH} (reusing v2)"
echo ""

# ── Job 1: Train ─────────────────────────────────────────────────
JOB_TRAIN=$(sbatch --parsable \
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
  --steps 200000 \\
  --batch_size 256 \\
  --lr 1e-3 \\
  --final_lr 1e-7 \\
  --warmup_frac 0.05 \\
  --weight_decay 1e-4 \\
  --d_hidden 512 \\
  --n_layers 4 \\
  --n_fourier 128 \\
  --fourier_scale 15.0 \\
  --alpha_corr 1.0 \\
  --eval_every 2000
")
echo "[submitted] train: job ${JOB_TRAIN}"

# ── Job 2: Evaluate + Plot ────────────────────────────────────────
# Write eval script to file (--wrap has a length limit on Trillium)
EVAL_SCRIPT="slurm/_eval_${TAG}.sh"
cat > "${EVAL_SCRIPT}" << 'EVALEOF'
#!/bin/bash
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2 scipy-stack/2024a
source "$HOME/envs/gqs/bin/activate"
cd "$SCRATCH/generative-quantum-states"
export PYTHONUNBUFFERED=1
EVALEOF
cat >> "${EVAL_SCRIPT}" << EOF
python3 -m fermionic_pipeline.eval.regressor_eval \\
  --data_path ${DATA_PATH} \\
  --checkpoint ${MODEL_DIR}/regressor.pt \\
  --save_dir ${MODEL_DIR}/eval \\
  --device cuda --ljung_box_p 0.06
python3 -m fermionic_pipeline.eval.plot_regression \\
  --data_path ${DATA_PATH} \\
  --checkpoint ${MODEL_DIR}/regressor.pt \\
  --save_dir ${MODEL_DIR}/plots \\
  --device cuda --ljung_box_p 0.06
EOF
chmod +x "${EVAL_SCRIPT}"

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
  "${EVAL_SCRIPT}")
echo "[submitted] eval:  job ${JOB_EVAL} (after ${JOB_TRAIN})"

echo ""
echo "=== Pipeline submitted ==="
echo "  train: ${JOB_TRAIN} → eval+plot: ${JOB_EVAL}"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: \$SCRATCH/generative-quantum-states/${MODEL_DIR}/"
