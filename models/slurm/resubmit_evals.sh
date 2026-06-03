#!/bin/bash
# =======================================================================
# Resubmit classifier evals (data + training already done)
# Also submit the full regression pipeline
#
# Usage (on Trillium):
#   cd $SCRATCH/generative-quantum-states
#   bash slurm/resubmit_evals.sh
# =======================================================================
set -euo pipefail

PARTITION="compute_full_node"
ACCOUNT="rrg-aspuru"

PREAMBLE='
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2 scipy-stack/2024a
source "$HOME/envs/gqs/bin/activate"
cd "$SCRATCH/generative-quantum-states"
export PYTHONUNBUFFERED=1
'

echo "=== Resubmitting classifier evals + regression pipeline ==="
echo ""

# ── Classifier eval: base ─────────────────────────────────────────
JOB_EVAL_BASE=$(sbatch --parsable \
  --partition=${PARTITION} \
  --job-name="eval-base-h4_exact_v1" \
  --output="logs/eval_base_h4_exact_v1_%j.out" \
  --error="logs/eval_base_h4_exact_v1_%j.err" \
  --time=04:00:00 \
  --gpus-per-node=4 \
  --cpus-per-task=4 \
  --account=${ACCOUNT} \
  --wrap="${PREAMBLE}
python3 -m fermionic_pipeline.eval.spectral_eval \\
  --data_path results/fermionic_pipeline/exact_conditional/h4_exact_v1/exact_conditionals.h5 \\
  --checkpoint results/fermionic_pipeline/conditional_classifier/h4_exact_v1_base/classifier.pt \\
  --save_dir results/fermionic_pipeline/conditional_classifier/h4_exact_v1_base/eval \\
  --device cuda --samples_per_q 50 --ljung_box_p 0.06
")
echo "[submitted] clf eval base: job ${JOB_EVAL_BASE}"

# ── Classifier eval: obs ──────────────────────────────────────────
JOB_EVAL_OBS=$(sbatch --parsable \
  --partition=${PARTITION} \
  --job-name="eval-obs-h4_exact_v1" \
  --output="logs/eval_obs_h4_exact_v1_%j.out" \
  --error="logs/eval_obs_h4_exact_v1_%j.err" \
  --time=04:00:00 \
  --gpus-per-node=4 \
  --cpus-per-task=4 \
  --account=${ACCOUNT} \
  --wrap="${PREAMBLE}
python3 -m fermionic_pipeline.eval.spectral_eval \\
  --data_path results/fermionic_pipeline/exact_conditional/h4_exact_v1/exact_conditionals.h5 \\
  --checkpoint results/fermionic_pipeline/conditional_classifier/h4_exact_v1_obs/classifier.pt \\
  --save_dir results/fermionic_pipeline/conditional_classifier/h4_exact_v1_obs/eval \\
  --device cuda --samples_per_q 50 --ljung_box_p 0.06
")
echo "[submitted] clf eval obs:  job ${JOB_EVAL_OBS}"

# ── Regression pipeline: datagen ──────────────────────────────────
# Reuses exact conditional HDF5 if it exists (no matchgate needed)
EXACT_COND="results/fermionic_pipeline/exact_conditional/h4_exact_v1/exact_conditionals.h5"
REG_DATA="results/fermionic_pipeline/regression/h4_regress_v1/regression_targets.h5"
REG_MODEL="results/fermionic_pipeline/regression/h4_regress_v1_model"

DATAGEN_EXTRA=""
if [ -f "${EXACT_COND}" ]; then
    DATAGEN_EXTRA="--exact_conditional ${EXACT_COND}"
    echo "[info] regression will reuse exact conditional HDF5"
fi

JOB_REG_DATA=$(sbatch --parsable \
  --partition=${PARTITION} \
  --job-name="reg-gen-h4_regress_v1" \
  --output="logs/reg_gen_h4_regress_v1_%j.out" \
  --error="logs/reg_gen_h4_regress_v1_%j.err" \
  --time=16:00:00 \
  --gpus-per-node=4 \
  --cpus-per-task=8 \
  --account=${ACCOUNT} \
  --wrap="${PREAMBLE}
mkdir -p results/fermionic_pipeline/regression/h4_regress_v1
python3 -m fermionic_pipeline.data.regression_dataset \\
  --config fermionic_pipeline/configs/h4_fast.yaml \\
  --output ${REG_DATA} \\
  --n_times 500 \\
  --t_max 100.0 \\
  --n_q 1000 \\
  ${DATAGEN_EXTRA}
")
echo "[submitted] reg datagen:   job ${JOB_REG_DATA}"

# ── Regression pipeline: train ────────────────────────────────────
JOB_REG_TRAIN=$(sbatch --parsable \
  --dependency=afterok:${JOB_REG_DATA} \
  --partition=${PARTITION} \
  --job-name="reg-train-h4_regress_v1" \
  --output="logs/reg_train_h4_regress_v1_%j.out" \
  --error="logs/reg_train_h4_regress_v1_%j.err" \
  --time=16:00:00 \
  --gpus-per-node=4 \
  --cpus-per-task=4 \
  --account=${ACCOUNT} \
  --wrap="${PREAMBLE}
python3 -m fermionic_pipeline.training.regressor_trainer \\
  --data_path ${REG_DATA} \\
  --save_dir ${REG_MODEL} \\
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
echo "[submitted] reg train:     job ${JOB_REG_TRAIN} (after ${JOB_REG_DATA})"

# ── Regression pipeline: eval ─────────────────────────────────────
JOB_REG_EVAL=$(sbatch --parsable \
  --dependency=afterok:${JOB_REG_TRAIN} \
  --partition=${PARTITION} \
  --job-name="reg-eval-h4_regress_v1" \
  --output="logs/reg_eval_h4_regress_v1_%j.out" \
  --error="logs/reg_eval_h4_regress_v1_%j.err" \
  --time=16:00:00 \
  --gpus-per-node=4 \
  --cpus-per-task=4 \
  --account=${ACCOUNT} \
  --wrap="${PREAMBLE}
python3 -m fermionic_pipeline.eval.regressor_eval \\
  --data_path ${REG_DATA} \\
  --checkpoint ${REG_MODEL}/regressor.pt \\
  --save_dir ${REG_MODEL}/eval \\
  --device cuda \\
  --ljung_box_p 0.06
")
echo "[submitted] reg eval:      job ${JOB_REG_EVAL} (after ${JOB_REG_TRAIN})"

echo ""
echo "=== All jobs submitted ==="
echo "  Classifier evals (immediate): ${JOB_EVAL_BASE}, ${JOB_EVAL_OBS}"
echo "  Regression pipeline:          ${JOB_REG_DATA} → ${JOB_REG_TRAIN} → ${JOB_REG_EVAL}"
echo ""
echo "Monitor: squeue -u \$USER"
