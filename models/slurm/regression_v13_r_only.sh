#!/bin/bash
# =======================================================================
# Regression v13 — R-only baseline (chemically-informed-features ablation).
#
# Paired control against the in-flight v13 v12f8 run (orbital-energy input).
# Same dataset (dt=0.05, h4_regress_v13), same architecture (v12f8 =
# --adaptive_bandwidth + --omega_op_floor 8), same hyperparams, same seeds.
# Single bit flipped: --use_orb_features dropped, so freq_net consumes
# scalar R instead of HF orbital energies (observable_regressor.py:128).
#
# Paper claim under test (QTML/journal item 4 in Aniket↔Luis plan):
#   "chemically informed features (orbital-energy inputs) improve data
#   efficiency/robustness; R-only is the controlled baseline."
#
# Pairing on disk:
#   h4_regress_v13_v12f8_s{42,1729}     ← in-flight, orb-energy input
#   h4_regress_v13_v12f8_R_s{42,1729}   ← this script, scalar R input
#
# Decision criteria after eval (composition_diagnostic + per-bin Pearson):
#   - if R-only ≈ orb-energy across all R bins:
#       chemically-informed-features claim does not hold for H4; drop or
#       reframe as "feature choice doesn't matter at H4 scale, but expect
#       gain at heavier systems where R is a weaker spectral coordinate."
#   - if R-only underperforms in [0.74, 1.0) but matches mid/long bins:
#       supports claim; orb-energy specifically resolves the short-R
#       electronic-structure regime where ε(R) gradient carries info R
#       cannot. Strongest paper story.
#   - if R-only collapses everywhere:
#       freq_net is load-bearing on orb-energy specifically; expected but
#       still useful as a clean ablation figure.
#
# Stages (composable; default = train + eval):
#   --train    submit v12f8_R training (one per seed)
#   --eval     submit eval+plot+composition_diagnostic (one per checkpoint)
#
# Usage:
#   bash slurm/regression_v13_r_only.sh                    # train + eval
#   bash slurm/regression_v13_r_only.sh --eval             # eval-only on existing ckpts
#   bash slurm/regression_v13_r_only.sh --after JOB_ID     # chain on v13 datagen
# =======================================================================
set -euo pipefail

TAG="h4_regress_v13"
OMEGA_OP_FLOOR="8.0"
DEP_AFTER=""
DO_TRAIN=false; DO_EVAL=false
ANY_STAGE=false

usage() {
  cat << EOF
Usage: $0 [--train] [--eval] [--all] [--after JOB_ID] [--tag NAME] [--omega_op_floor F]

Stages (composable; default = both):
  --train    submit R-only training (v12f8 architecture × 2 seeds)
  --eval     submit eval+plot+composition (one per checkpoint)
  --all      explicit form of "both"

Tag overrides:
  --tag NAME             dataset tag prefix (default: h4_regress_v13; reuses v13 dataset)
  --omega_op_floor F     ω_floor for v12 architecture (default: 8.0; matches v13 v12f8)
  --after JOB_ID         chain training afterok on JOB_ID (e.g. v13 datagen still in flight)
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --train)            DO_TRAIN=true; ANY_STAGE=true ;;
    --eval)             DO_EVAL=true; ANY_STAGE=true ;;
    --all)              DO_TRAIN=true; DO_EVAL=true; ANY_STAGE=true ;;
    --tag)              TAG="$2"; shift ;;
    --omega_op_floor)   OMEGA_OP_FLOOR="$2"; shift ;;
    --after)            DEP_AFTER="$2"; shift ;;
    -h|--help)          usage; exit 0 ;;
    *)                  echo "unknown flag: $1"; usage; exit 1 ;;
  esac
  shift
done

if [ "$ANY_STAGE" = "false" ]; then
  DO_TRAIN=true; DO_EVAL=true
fi

DATA_DIR="results/fermionic_pipeline/regression/${TAG}"
DATA_PATH="${DATA_DIR}/regression_targets.h5"
SEEDS=(42 1729)

PARTITION="compute_full_node"
ACCOUNT="rrg-aspuru"

mkdir -p logs

if [ -z "$DEP_AFTER" ] && [ ! -f "${DATA_PATH}" ]; then
  echo "ERROR: dataset missing at ${DATA_PATH}"
  echo "v13 datagen must complete first, or pass --after JOB_ID to chain on it."
  exit 1
fi

echo "=== Regression v13 R-only baseline (v12f8 arch, scalar R input) ==="
echo "  tag:      ${TAG}"
echo "  data:     ${DATA_PATH}"
echo "  stages:   train=${DO_TRAIN} eval=${DO_EVAL}"
[ -n "$DEP_AFTER" ] && echo "  after:    ${DEP_AFTER}"
echo ""

VARIANT_NAME="v12f8_R"
VARIANT_FLAGS="--adaptive_bandwidth --omega_op_floor ${OMEGA_OP_FLOOR}"
SUBMITTED=()

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
    if [ -n "$DEP_AFTER" ]; then
      SBATCH_ARGS_T+=(--dependency=afterok:${DEP_AFTER})
    fi
    JOB_TRAIN=$(sbatch "${SBATCH_ARGS_T[@]}" "slurm/_train_${SEED_TAG}.sh")
    DEP_NOTE=""
    [ -n "$DEP_AFTER" ] && DEP_NOTE=" (afterok ${DEP_AFTER})"
    echo "[submitted] train ${VARIANT_NAME} seed=${SEED}: job ${JOB_TRAIN}${DEP_NOTE}"
    SUBMITTED+=("${VARIANT_NAME} seed=${SEED}: train=${JOB_TRAIN}")
  fi

  if [ "$DO_EVAL" = "true" ]; then
    if [ -z "$JOB_TRAIN" ] && [ ! -f "${CKPT}" ]; then
      echo "ERROR: checkpoint missing at ${CKPT}"
      echo "Pass --train and --eval together, or wait for training to finish."
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

if [ "${#SUBMITTED[@]}" -gt 0 ]; then
  echo ""
  echo "=== Submitted ==="
  for line in "${SUBMITTED[@]}"; do
    echo "  ${line}"
  done
fi
