#!/bin/bash
# =======================================================================
# Regression v17: decoupled gradient clipping + chemically-informed-input ablation.
#
# Two independent changes vs v16:
#
# (A) DECOUPLED CLIPPING (fixes the v16 5/27 regression)
#     v16 used joint clipping: clip_grad_norm_(model.parameters(), 1.0). It killed
#     the loss explosion but ALSO broke the borderline composition win — global
#     clipping rescales the whole gradient when the residual trunk's large grads
#     trip the threshold, throttling the explicit branch (which learns the delicate
#     [0.74,1.0) phase/sign composition). Result: [0.74,1.0) Pearson 0.95 (v15) →
#     0.72 (v16 pooled), seed-fragile (s42 phase_err 1.10 rad, s1729 0.12 rad).
#     v17 clips the residual trunk ALONE (--residual_grad_clip 1.0) and leaves the
#     explicit branch unthrottled (--grad_clip 0). Hypothesis: borderline recovers
#     to v15 levels while long R stays solved (v16 ≥2.0 pooled 0.988).
#
# (B) CHEMICALLY-INFORMED-INPUT ABLATION (orb energies vs pure geometry R)
#     Paper claim target: HF orbital-energy inputs beat pure-R geometry inputs.
#     v17 runs BOTH input modes under an otherwise-identical architecture so the
#     comparison is clean. This ablation is now a STANDARD fixture — every future
#     regression run should carry the orb-vs-R pair.
#       v17_orb : --use_orb_features --standardize_orb_energies   (chemically informed)
#       v17_R   : (neither)                                       (geometry only; freq/amp
#                  nets take scalar R). NOTE: R is fed raw (one feature, range ~0.5-3.0);
#                  orb energies are standardized (4 features, disparate scales). The
#                  standardization asymmetry is inherent to the two pipelines, not a
#                  thumb on the scale — flag it when interpreting.
#
# Matrix: {orb, R} × {seed 42, 1729} = 4 train + 4 eval jobs.
# Dataset: reuses h4_regress_v13 (dt=0.05). NO datagen stage.
#
# Decision rules (per-bin Pearson seed-pooled n=100 + composition phase_err):
#   (A) decoupled clip:
#     - v17_orb [0.74,1.0) recovers to ≥ 0.90 with phase_err ≤ 0.3 rad AND ≥2.0
#       stays ≥ 0.97  →  coupling hypothesis confirmed; v17 unifies both regimes.
#       Ship v17_orb as the new baseline.
#     - [0.74,1.0) still < 0.85 or seed |Δ| > 0.15  →  decoupling insufficient;
#       fall back to staged-residual or the R-selected regime split (5/27 option 4).
#   (B) orb vs R:
#     - v17_orb beats v17_R by a clear per-bin margin (esp. short/borderline R,
#       where chemistry should matter most)  →  supports the paper claim.
#     - v17_R matches v17_orb  →  geometry alone suffices; the orb-feature story
#       is not load-bearing for this observable set. Report honestly.
#
# Stages (composable; default = train + eval):
#   --train  submit training jobs (4 = 2 input-modes × 2 seeds)
#   --eval   submit eval+plot+composition jobs (one per checkpoint)
#
# Usage:
#   bash slurm/regression_v17.sh                          # train + eval, all 4
#   bash slurm/regression_v17.sh --eval                   # eval-only on existing ckpts
#   bash slurm/regression_v17.sh --train --eval --exclude trig0019   # steer off bad node
# =======================================================================
set -euo pipefail

TAG="h4_regress_v17"
DATA_TAG="h4_regress_v13"   # reuse the dt=0.05 canonical dataset
OMEGA_OP_FLOOR="8.0"
DO_TRAIN=false; DO_EVAL=false
ANY_STAGE=false
EXCLUDE_NODES=""

usage() {
  cat << EOF
Usage: $0 [--train] [--eval] [--all] [--tag NAME] [--data_tag NAME] [--omega_op_floor F] [--exclude NODELIST]

Stages (composable; default = train + eval):
  --train     submit training jobs ({orb,R} × 2 seeds = 4)
  --eval      submit eval+plot+composition jobs (one per checkpoint)
  --all       explicit form of "train + eval"

Overrides:
  --tag NAME             model tag prefix (default: h4_regress_v17)
  --data_tag NAME        upstream dataset tag (default: h4_regress_v13)
  --omega_op_floor F     ω_floor in E_h (default: 8.0)
  --exclude NODELIST     SLURM --exclude for all jobs (e.g. trig0019, the node that
                         crashed v16 with "No CUDA GPUs are available").
EOF
}

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

echo "=== Regression v17 (decoupled clipping + orb-vs-R ablation; dataset=${DATA_TAG}, ω_floor=${OMEGA_OP_FLOOR}) ==="
echo "  tag:      ${TAG}"
echo "  stages:   train=${DO_TRAIN} eval=${DO_EVAL}"
echo "  data:     ${DATA_PATH}"
[ -n "$EXCLUDE_NODES" ] && echo "  exclude:  ${EXCLUDE_NODES}"
echo ""

if [ ! -f "${DATA_PATH}" ]; then
  echo "ERROR: dataset missing at ${DATA_PATH}"
  echo "v17 reuses the v13 dt=0.05 dataset; run slurm/regression_v13.sh --data first."
  exit 1
fi

SUBMITTED=()

# Architecture shared by both input modes: v15_explicit + residual trunk, decoupled
# clipping (residual trunk clipped at 1.0, explicit branch unthrottled).
COMMON_ARCH_FLAGS="--adaptive_bandwidth --omega_op_floor ${OMEGA_OP_FLOOR} --soft_omega_floor --explicit_amplitude --amp_rank 16 --with_residual --grad_clip 0 --residual_grad_clip 1.0"

# Input-mode ablation. orb = chemically informed; R = geometry only.
declare -a VARIANTS=(
  "v17_orb:${COMMON_ARCH_FLAGS} --use_orb_features --standardize_orb_energies"
  "v17_R:${COMMON_ARCH_FLAGS}"
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
