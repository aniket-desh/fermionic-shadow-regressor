#!/bin/bash
# =======================================================================
# Regression v16: FAN-style residual on top of v15_explicit.
#
#   y_μ(R, t) = explicit_μ(R, t) + residual_μ(R, t)
#   explicit_μ = Σ_k a_kμ(x) cos(ω_k(R) t) + b_kμ(x) sin(ω_k(R) t) + dc_μ(x)
#                  (= v15_explicit, amp_rank=16)
#   residual_μ = MLP([R, sin(ω_k(R) t), cos(ω_k(R) t)])_μ
#                  (= v12f8-style trunk, shared ω, output zero-initialized)
#
# Motivation (5/20 log entry — v15 readout):
#   v15_explicit closed the [0.74, 1.0) composition gap (0.51 → 0.95) and lifted
#   <0.74 (0.66 → 0.94), but regressed long R (≥2.0: 0.994 → 0.873; [1.5, 2.0):
#   0.971 → 0.930). Interpretation: the linear-in-amplitude factorization is
#   the right function class at borderline R (composition error fixed), but
#   the GELU trunk in v12f8 was carrying useful extra-Fourier capacity at
#   long R — likely DC drift / nonsinusoidal residuals that the K=256
#   Fourier basis doesn't span.
#
#   v16 adds a v12f8-style MLP residual on top of v15_explicit. Residual output
#   layer is zero-initialized so training starts at v15_explicit and additively
#   learns long-R residual nonlinear structure. ω is shared between branches
#   so the residual is forced to encode genuinely non-Fourier structure rather
#   than patching amplitude errors via alternate frequencies.
#
# 5/24 update — grad clipping:
#   The first v16 run (jobs 528764/528766) suffered a loss explosion at step
#   ~18000 (warmup end / peak LR 1e-3): train_mse spiked to 3.4e6, val_corr
#   briefly → 0.0008, then cosine-annealed back. Root cause: the unbounded GELU
#   residual trunk + no gradient clipping. Test metrics still came out best-in-
#   class on s1729, but the trajectory through a 1e6 spike is not shippable.
#   This run now passes --grad_clip 1.0 to guard the residual trunk.
#
# Dataset: reuses h4_regress_v13 (dt=0.05). NO datagen stage — this script
# only knows --train and --eval.
#
# Variant: v16_residual × 2 seeds (42, 1729).
#
# Decision rules (per-bin Pearson, seed-pooled, n=100):
#   - Long R (≥2.0, [1.5, 2.0)) recovers to within 0.02 of v13_v12f8 AND
#     borderline [0.74, 1.0) stays ≥ 0.90  →  v16 unifies; ship as new baseline.
#   - Long R recovers but borderline regresses below 0.90  →  residual is leaking
#     into borderline. Tighten with --residual_amp_penalty (future flag) and rerun.
#   - Long R doesn't recover  →  the issue isn't extra-Fourier capacity. Punt to
#     R-conditioned gating between explicit and trunk branches.
#
# Stages (composable; default = train + eval):
#   --train  submit training jobs (2 = 1 variant × 2 seeds)
#   --eval   submit eval+plot+composition jobs (one per checkpoint)
#
# Usage:
#   bash slurm/regression_v16.sh                # train + eval
#   bash slurm/regression_v16.sh --eval         # eval-only on existing ckpts
#   bash slurm/regression_v16.sh --tag h4_regress_v16_alt
# =======================================================================
set -euo pipefail

TAG="h4_regress_v16"
DATA_TAG="h4_regress_v13"   # reuse the dt=0.05 canonical dataset
OMEGA_OP_FLOOR="8.0"
DO_TRAIN=false; DO_EVAL=false
ANY_STAGE=false

usage() {
  cat << EOF
Usage: $0 [--train] [--eval] [--all] [--tag NAME] [--data_tag NAME] [--omega_op_floor F]

Stages (composable; default = train + eval):
  --train     submit training jobs (v16_residual × 2 seeds)
  --eval      submit eval+plot+composition jobs (one per checkpoint)
  --all       explicit form of "train + eval"

Tag overrides:
  --tag NAME             model tag prefix (default: h4_regress_v16)
  --data_tag NAME        upstream dataset tag (default: h4_regress_v13)
  --omega_op_floor F     ω_floor in E_h for the v12 architecture (default: 8.0)
  --exclude NODELIST     SLURM --exclude passed to all jobs (e.g. trig0019).
                         Use to steer off a node with broken GPUs. The 5/22 and
                         5/24 v16 runs all crashed on trig0019 with
                         "No CUDA GPUs are available" despite gres/gpu=4 allocated.
EOF
}

EXCLUDE_NODES=""

while [ $# -gt 0 ]; do
  case "$1" in
    --train)            DO_TRAIN=true; ANY_STAGE=true ;;
    --eval)             DO_EVAL=true; ANY_STAGE=true ;;
    --all)              DO_TRAIN=true; DO_EVAL=true; ANY_STAGE=true ;;
    --tag)              TAG="$2"; shift ;;
    --data_tag)         DATA_TAG="$2"; shift ;;
    --omega_op_floor)   OMEGA_OP_FLOOR="$2"; shift ;;
    --exclude)          EXCLUDE_NODES="$2"; shift ;;
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

echo "=== Regression v16 (FAN-style residual on v15_explicit; dataset=${DATA_TAG}, ω_floor=${OMEGA_OP_FLOOR}) ==="
echo "  tag:      ${TAG}"
echo "  stages:   train=${DO_TRAIN} eval=${DO_EVAL}"
echo "  data:     ${DATA_PATH}"
echo ""

if [ ! -f "${DATA_PATH}" ]; then
  echo "ERROR: dataset missing at ${DATA_PATH}"
  echo "v16 reuses the v13 dt=0.05 dataset; run slurm/regression_v13.sh --data first."
  exit 1
fi

SUBMITTED=()

# Single variant: v15_explicit flags + --with_residual.
# Loss is v13's exactly: --alpha_corr 1.0, NO temporal_corr.
COMMON_ARCH_FLAGS="--adaptive_bandwidth --omega_op_floor ${OMEGA_OP_FLOOR} --soft_omega_floor --standardize_orb_energies --explicit_amplitude --amp_rank 16 --with_residual"

declare -a VARIANTS=(
  "v16_residual:${COMMON_ARCH_FLAGS}"
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
  --grad_clip 1.0 \\
  --eval_every 2000
EOF
      chmod +x "slurm/_train_${SEED_TAG}.sh"

      SBATCH_ARGS_T=(
        --parsable
        --partition=${PARTITION}
        --job-name="reg-train-${SEED_TAG}"
        --output="logs/reg_train_${SEED_TAG}_%j.out"
        --error="logs/reg_train_${SEED_TAG}_%j.err"
        --time=01:30:00
        --gpus-per-node=4
        --cpus-per-task=4
        --account=${ACCOUNT}
      )
      [ -n "$EXCLUDE_NODES" ] && SBATCH_ARGS_T+=(--exclude="${EXCLUDE_NODES}")
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
      [ -n "$EXCLUDE_NODES" ] && SBATCH_ARGS_E+=(--exclude="${EXCLUDE_NODES}")
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
