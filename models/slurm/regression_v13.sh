#!/bin/bash
# =======================================================================
# Regression v13: dt=0.05 dataset + v12_f8 architecture + v11 baseline.
#
# Theory: Prop 2 of paper draft requires Δt ≤ π / (2‖H(R)‖). The H4 spectral
# norm sweep (scripts/diagnostics/h4_norm_vs_R.py, 5/03) gives ‖H(0.5)‖ = 24
# E_h, spread = 22.96 E_h. dt=0.05 admits Ω_max = π/0.05 = 62.8 E_h, a 2.7×
# margin against the worst-case Bohr support — comfortably safe.
#
# v10 dt=0.20 broke at R<0.74 because ω_op_999 ≈ 15 E_h there exceeds the
# Nyquist limit Ω = 15.7 E_h. dt=0.05 should remove the aliasing confound,
# enabling a clean test of v12_f8's [0.74, 1.0) regression vs v11.
#
# Reading the v13 results:
#   - If v12_f8 + dt=0.05 lifts [0.74, 1.0) past v11's old 0.66:
#       declare residual short-R failure data-only; skip v13 architectural fix.
#   - If v12_f8 + dt=0.05 leaves [0.74, 1.0) below mid/long bins:
#       constant floor is the architectural bottleneck; pivot to R-conditioned
#       floor (true v13 architecture).
#
# Stages (composable; default = all four):
#   --data     submit dt=0.05 datagen (~7-8h wall under batched primitives) + ω_op
#   --train    submit v12_f8 training (one per seed) + v11 baseline (one per seed)
#   --eval     submit eval+plot+composition_diagnostic (one per checkpoint)
#
# Walltime note: Trillium hard-caps at 24h. v10 N_T=1500 took 10h43m wall on
# Trillium. v13 N_T=6001 is 4× per-geom work and would take ~43h with the
# original per-t hot loop — over the wall. The 5/03 batching of the inner
# Q-loop (regression_dataset.py:_compute_signal_block_fast) measured 7× speedup
# locally at T=200, projecting v13 to ~7.5h wall. 24h walltime is now budget,
# not target.
#
# Usage:
#   bash slurm/regression_v13.sh                    # full pipeline
#   bash slurm/regression_v13.sh --eval             # eval-only on existing checkpoints
#   bash slurm/regression_v13.sh --train --eval     # skip datagen
#   bash slurm/regression_v13.sh --omega_op_floor 8 # override floor (default 8)
# =======================================================================
set -euo pipefail

TAG="h4_regress_v13"
OMEGA_OP_FLOOR="8.0"
DT="0.05"
N_TIMES="6001"
DO_DATA=false; DO_TRAIN=false; DO_EVAL=false
ANY_STAGE=false

usage() {
  cat << EOF
Usage: $0 [--data] [--train] [--eval] [--all] [--tag NAME] [--omega_op_floor F]

Stages (composable; default = all three):
  --data     build dt=0.05 dataset + omega_op field
  --train    submit training jobs (v12_f8 × 2 seeds + v11 baseline × 2 seeds)
  --eval     submit eval+plot+composition jobs (one per checkpoint)
  --all      explicit form of "all three"

Tag overrides:
  --tag NAME             dataset/model tag prefix (default: h4_regress_v13)
  --omega_op_floor F     ω_floor in E_h for v12 architecture (default: 8.0)
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --data)             DO_DATA=true; ANY_STAGE=true ;;
    --train)            DO_TRAIN=true; ANY_STAGE=true ;;
    --eval)             DO_EVAL=true; ANY_STAGE=true ;;
    --all)              DO_DATA=true; DO_TRAIN=true; DO_EVAL=true; ANY_STAGE=true ;;
    --tag)              TAG="$2"; shift ;;
    --omega_op_floor)   OMEGA_OP_FLOOR="$2"; shift ;;
    -h|--help)          usage; exit 0 ;;
    *)                  echo "unknown flag: $1"; usage; exit 1 ;;
  esac
  shift
done

if [ "$ANY_STAGE" = "false" ]; then
  DO_DATA=true; DO_TRAIN=true; DO_EVAL=true
fi

DATA_DIR="results/fermionic_pipeline/regression/${TAG}"
DATA_PATH="${DATA_DIR}/regression_targets.h5"
SEEDS=(42 1729)

PARTITION="compute_full_node"
ACCOUNT="rrg-aspuru"

mkdir -p logs

echo "=== Regression v13 (dt=${DT}, n_times=${N_TIMES}, ω_floor=${OMEGA_OP_FLOOR}) ==="
echo "  tag:      ${TAG}"
echo "  stages:   data=${DO_DATA} train=${DO_TRAIN} eval=${DO_EVAL}"
echo "  data:     ${DATA_PATH}"
echo ""

# ── Stage: datagen + omega_op ─────────────────────────────────────
JOB_DATA=""
if [ "$DO_DATA" = "true" ]; then
  cat > "slurm/_datagen_${TAG}.sh" << 'EOF'
#!/bin/bash
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2
source "$HOME/envs/gqs/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
cd "$SCRATCH/generative-quantum-states"
export PYTHONUNBUFFERED=1
# Pin BLAS threads to 1 per worker (NumPy/OpenBLAS otherwise spawns ~96
# threads per fork and thrashes the node).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
EOF
  cat >> "slurm/_datagen_${TAG}.sh" << EOF
python3 -m fermionic_pipeline.data.regression_dataset \\
  --output ${DATA_PATH} \\
  --n_atoms 4 \\
  --r_start 0.5 --r_end 3.0 \\
  --r_step 0.01 \\
  --t_max 300.0 --n_times ${N_TIMES} --n_q 500 \\
  --n_workers 16
python3 -m fermionic_pipeline.data.compute_omega_op \\
  --data_path ${DATA_PATH}
EOF
  chmod +x "slurm/_datagen_${TAG}.sh"

  JOB_DATA=$(sbatch --parsable \
    --partition=${PARTITION} \
    --job-name="reg-data-${TAG}" \
    --output="logs/reg_data_${TAG}_%j.out" \
    --error="logs/reg_data_${TAG}_%j.err" \
    --time=24:00:00 \
    --gpus-per-node=4 \
    --cpus-per-task=16 \
    --account=${ACCOUNT} \
    "slurm/_datagen_${TAG}.sh")
  echo "[submitted] datagen+ω_op: job ${JOB_DATA} (24h)"
fi

if [ "$DO_TRAIN" = "true" ] || [ "$DO_EVAL" = "true" ]; then
  if [ -z "$JOB_DATA" ] && [ ! -f "${DATA_PATH}" ]; then
    echo "ERROR: v13 dataset missing at ${DATA_PATH}"
    echo "Pass --data first, or wait for an in-flight datagen to complete."
    exit 1
  fi
fi

SUBMITTED=()

# Two model variants on the same dataset:
#   v12f8: --adaptive_bandwidth --omega_op_floor ${OMEGA_OP_FLOOR}
#   v11:   --adaptive_bandwidth --omega_op_floor 0    (no floor; v11 architecture)
declare -a VARIANTS=("v12f8:--adaptive_bandwidth --omega_op_floor ${OMEGA_OP_FLOOR}"
                     "v11:--adaptive_bandwidth --omega_op_floor 0")

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
        --time=24:00:00
        --gpus-per-node=4
        --cpus-per-task=4
        --account=${ACCOUNT}
      )
      if [ -n "$JOB_DATA" ]; then
        SBATCH_ARGS_T+=(--dependency=afterok:${JOB_DATA})
      fi
      JOB_TRAIN=$(sbatch "${SBATCH_ARGS_T[@]}" "slurm/_train_${SEED_TAG}.sh")
      DEP_NOTE=""
      [ -n "$JOB_DATA" ] && DEP_NOTE=" (afterok ${JOB_DATA})"
      echo "[submitted] train ${VARIANT_NAME} seed=${SEED}: job ${JOB_TRAIN}${DEP_NOTE}"
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
