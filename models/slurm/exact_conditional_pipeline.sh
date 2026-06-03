#!/bin/bash
# =======================================================================
# Exact conditional classifier pipeline: datagen → train (base + obs) → eval
#
# Usage (from Trillium login node):
#   bash slurm/exact_conditional_pipeline.sh [TAG]
#
# Submits 5 jobs with dependency chains:
#   datagen → train_base → eval_base
#           → train_obs  → eval_obs
#
# Default TAG: h4_exact_v1
# =======================================================================
set -euo pipefail

TAG="${1:-h4_exact_v1}"

# ── Paths (relative to $SCRATCH/generative-quantum-states) ─────────
CONFIG="fermionic_pipeline/configs/h4_fast.yaml"
DATA_DIR="results/fermionic_pipeline/exact_conditional/${TAG}"
DATA_PATH="${DATA_DIR}/exact_conditionals.h5"
BASE_DIR="results/fermionic_pipeline/conditional_classifier/${TAG}_base"
OBS_DIR="results/fermionic_pipeline/conditional_classifier/${TAG}_obs"

# ── Exact conditional data generation parameters ───────────────────
# Override h4_fast defaults for spectral-quality time grid:
#   500 time points over t_max=100 → Δω = 2π/100 ≈ 0.063 Eₕ
#   Resolves H4 gaps down to ~0.13 Eₕ
N_Q=100
N_TIMES=500
T_MAX=100.0

# ── Common preamble for --wrap commands ────────────────────────────
PREAMBLE='
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2 scipy-stack/2024a
source "$HOME/envs/gqs/bin/activate"
cd "$SCRATCH/generative-quantum-states"
export PYTHONUNBUFFERED=1
'

# ── Trillium SBATCH defaults ──────────────────────────────────────
# Trillium requires --partition=compute_full_node and --gpus-per-node
# for all jobs. No --mem allowed (186 GiB/GPU fixed).
PARTITION="compute_full_node"

echo "=== Exact Conditional Pipeline ==="
echo "  tag:     ${TAG}"
echo "  config:  ${CONFIG}"
echo "  n_q:     ${N_Q}"
echo "  n_times: ${N_TIMES}"
echo "  t_max:   ${T_MAX}"
echo ""

# ── Job 1: Generate exact conditional dataset ────────────────────
JOB_DATA=$(sbatch --parsable \
  --partition=${PARTITION} \
  --job-name="exact-gen-${TAG}" \
  --output="logs/exact_gen_${TAG}_%j.out" \
  --error="logs/exact_gen_${TAG}_%j.err" \
  --time=04:00:00 \
  --gpus-per-node=4 \
  --cpus-per-task=8 \
  --account=rrg-aspuru \
  --wrap="${PREAMBLE}
mkdir -p ${DATA_DIR}
python3 -m fermionic_pipeline.data.exact_conditional_dataset \\
  --config ${CONFIG} \\
  --output ${DATA_PATH} \\
  --n_q ${N_Q} \\
  --n_times ${N_TIMES} \\
  --t_max ${T_MAX} \\
  --storage_dtype float32
")
echo "[submitted] datagen:    job ${JOB_DATA}"

# ── Job 2a: Train base classifier (KL only) ──────────────────────
JOB_TRAIN_BASE=$(sbatch --parsable \
  --dependency=afterok:${JOB_DATA} \
  --partition=${PARTITION} \
  --job-name="clf-base-${TAG}" \
  --output="logs/clf_base_${TAG}_%j.out" \
  --error="logs/clf_base_${TAG}_%j.err" \
  --time=04:00:00 \
  --gpus-per-node=4 \
  --cpus-per-task=4 \
  --account=rrg-aspuru \
  --wrap="${PREAMBLE}
python3 -m fermionic_pipeline.training.classifier_trainer \\
  --data_path ${DATA_PATH} \\
  --save_dir ${BASE_DIR} \\
  --device cuda \\
  --steps 50000 \\
  --batch_size 256 \\
  --lr 1e-3 \\
  --final_lr 1e-7 \\
  --warmup_frac 0.05 \\
  --weight_decay 1e-4
")
echo "[submitted] train_base: job ${JOB_TRAIN_BASE} (after ${JOB_DATA})"

# ── Job 2b: Train obs classifier (KL + observable head) ──────────
JOB_TRAIN_OBS=$(sbatch --parsable \
  --dependency=afterok:${JOB_DATA} \
  --partition=${PARTITION} \
  --job-name="clf-obs-${TAG}" \
  --output="logs/clf_obs_${TAG}_%j.out" \
  --error="logs/clf_obs_${TAG}_%j.err" \
  --time=04:00:00 \
  --gpus-per-node=4 \
  --cpus-per-task=4 \
  --account=rrg-aspuru \
  --wrap="${PREAMBLE}
python3 -m fermionic_pipeline.training.classifier_trainer \\
  --data_path ${DATA_PATH} \\
  --save_dir ${OBS_DIR} \\
  --device cuda \\
  --steps 50000 \\
  --batch_size 256 \\
  --lr 1e-3 \\
  --final_lr 1e-7 \\
  --warmup_frac 0.05 \\
  --weight_decay 1e-4 \\
  --observable_head \\
  --lambda_obs 1.0 \\
  --alpha_corr 1.0
")
echo "[submitted] train_obs:  job ${JOB_TRAIN_OBS} (after ${JOB_DATA})"

# ── Job 3a: Eval base classifier ─────────────────────────────────
JOB_EVAL_BASE=$(sbatch --parsable \
  --dependency=afterok:${JOB_TRAIN_BASE} \
  --partition=${PARTITION} \
  --job-name="eval-base-${TAG}" \
  --output="logs/eval_base_${TAG}_%j.out" \
  --error="logs/eval_base_${TAG}_%j.err" \
  --time=04:00:00 \
  --gpus-per-node=4 \
  --cpus-per-task=4 \
  --account=rrg-aspuru \
  --wrap="${PREAMBLE}
python3 -m fermionic_pipeline.eval.spectral_eval \\
  --data_path ${DATA_PATH} \\
  --checkpoint ${BASE_DIR}/classifier.pt \\
  --save_dir ${BASE_DIR}/eval \\
  --device cuda \\
  --samples_per_q 50 \\
  --ljung_box_p 0.06
")
echo "[submitted] eval_base:  job ${JOB_EVAL_BASE} (after ${JOB_TRAIN_BASE})"

# ── Job 3b: Eval obs classifier ──────────────────────────────────
JOB_EVAL_OBS=$(sbatch --parsable \
  --dependency=afterok:${JOB_TRAIN_OBS} \
  --partition=${PARTITION} \
  --job-name="eval-obs-${TAG}" \
  --output="logs/eval_obs_${TAG}_%j.out" \
  --error="logs/eval_obs_${TAG}_%j.err" \
  --time=04:00:00 \
  --gpus-per-node=4 \
  --cpus-per-task=4 \
  --account=rrg-aspuru \
  --wrap="${PREAMBLE}
python3 -m fermionic_pipeline.eval.spectral_eval \\
  --data_path ${DATA_PATH} \\
  --checkpoint ${OBS_DIR}/classifier.pt \\
  --save_dir ${OBS_DIR}/eval \\
  --device cuda \\
  --samples_per_q 50 \\
  --ljung_box_p 0.06
")
echo "[submitted] eval_obs:   job ${JOB_EVAL_OBS} (after ${JOB_TRAIN_OBS})"

echo ""
echo "=== Pipeline submitted ==="
echo "  datagen:    ${JOB_DATA}"
echo "  train_base: ${JOB_TRAIN_BASE} → eval_base: ${JOB_EVAL_BASE}"
echo "  train_obs:  ${JOB_TRAIN_OBS}  → eval_obs:  ${JOB_EVAL_OBS}"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: \$SCRATCH/generative-quantum-states/results/fermionic_pipeline/conditional_classifier/${TAG}_*/"
