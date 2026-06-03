#!/bin/bash
# =======================================================================
# Regression v18b: orb-arm clipping probe (diagnose the v18_orb collapse).
#
# WHY THIS EXISTS
#   v18_orb is a literal re-run of the v15_explicit config that scored
#   overall temporal-Pearson 0.915 (short-R 0.935) — same dataset
#   (h4_regress_v13, dt=0.05), same 150k steps, same --alpha_corr 1.0,
#   same arch flags, --use_orb_features hardcoded. Yet v18_orb collapsed:
#   overall 0.323, short-R 0.022, and the R baseline (0.737) now BEATS it,
#   inverting manuscript pillar 4. The composition diagnostic localizes the
#   failure to the orb amp_net + freq_net: envelope_pearson ≈ 0 and
#   phase_err ≈ π/2 (random) across all but the longest R.
#
#   The collapse hid in training because val_corr (cross-observable Pearson,
#   per (R,t) across K obs) stayed healthy at ~0.954 — that is NOT the
#   per-observable TEMPORAL Pearson that eval/spectroscopy needs
#   (see regressor_trainer.py:67-74). What the logs DID show is a loss
#   explosion: train_mse hit ~1.3e5 (max 3.6e5) for orb, vs ~480-6k for R,
#   under v18's deliberate "no trunk ⇒ stable unclipped" decision (grad_clip 0).
#
#   HYPOTHESIS: the "stable unclipped" premise held for scalar-R input but
#   FAILED for the 4-dim orb input — unclipped loss spikes corrupt the
#   delicate amp_net/freq_net composition that the diagnostic shows broken.
#
#   THIS PROBE: re-run v18_orb EXACTLY, changing ONE thing — restore
#   gradient clipping (--grad_clip 1.0). Single variant, single seed (42),
#   train+eval. Fast/cheap decisive test.
#
# DECISION RULE (per-bin temporal Pearson + composition envelope/phase):
#   - envelope_pearson recovers toward v15 levels (short/borderline ≫ 0) AND
#     train_mse stops exploding  → the clipping removal caused the collapse.
#     Restore clipping in v18 proper, re-run the full {orb,R}×{42,1729} matrix.
#   - still collapsed (envelope ≈ 0, phase ≈ π/2)  → NOT clipping; bisect the
#     standardization-into-forward refactor / omega_op plumbing added in v16/v17
#     (suspect double-standardization or a buffer-vs-dataset mismatch on the
#     orb path). v15 and v18 share one uncommitted working tree, so reconstruct
#     v15's code state to diff.
#
# Arch = v18_orb exactly + clipping:
#   --adaptive_bandwidth --omega_op_floor 8.0 --soft_omega_floor
#   --explicit_amplitude --amp_rank 16
#   --use_orb_features --standardize_orb_energies
#   --grad_clip 1.0                          <-- the ONLY change vs v18_orb
#   (NO --with_residual: there is no residual trunk; grad_clip applies jointly
#    to all params via the trainer's joint-clip branch.)
#
# Usage:
#   bash slurm/regression_v18b_orb_clip.sh                 # train + eval
#   bash slurm/regression_v18b_orb_clip.sh --eval          # eval-only on existing ckpt
#   bash slurm/regression_v18b_orb_clip.sh --exclude trig0019,trig0034
# =======================================================================
set -euo pipefail

TAG="h4_regress_v18b"
DATA_TAG="h4_regress_v13"   # same dt=0.05 canonical dataset as v18
OMEGA_OP_FLOOR="8.0"
GRAD_CLIP="1.0"             # the single variable under test
SEED=42
DO_TRAIN=false; DO_EVAL=false
ANY_STAGE=false
EXCLUDE_NODES=""

usage() {
  cat << EOF
Usage: $0 [--train] [--eval] [--all] [--seed N] [--grad_clip F] [--tag NAME] [--data_tag NAME] [--omega_op_floor F] [--exclude NODELIST]

Stages (composable; default = train + eval):
  --train     submit the training job (orb + clipping, one seed)
  --eval      submit eval+plot+composition (one checkpoint)
  --all       explicit form of "train + eval"

Overrides:
  --seed N               training seed (default: 42)
  --grad_clip F          max grad norm under test (default: 1.0)
  --tag NAME             model tag prefix (default: h4_regress_v18b)
  --data_tag NAME        upstream dataset tag (default: h4_regress_v13)
  --omega_op_floor F     ω_floor in E_h (default: 8.0)
  --exclude NODELIST     SLURM --exclude (e.g. trig0019,trig0034 — flaky GPU nodes).
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --train)            DO_TRAIN=true; ANY_STAGE=true ;;
    --eval)             DO_EVAL=true; ANY_STAGE=true ;;
    --all)              DO_TRAIN=true; DO_EVAL=true; ANY_STAGE=true ;;
    --seed)             SEED="$2"; shift ;;
    --grad_clip)        GRAD_CLIP="$2"; shift ;;
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

PARTITION="compute_full_node"
ACCOUNT="rrg-aspuru"

mkdir -p logs

echo "=== Regression v18b (orb clipping probe; dataset=${DATA_TAG}, ω_floor=${OMEGA_OP_FLOOR}, grad_clip=${GRAD_CLIP}) ==="
echo "  tag:      ${TAG}"
echo "  stages:   train=${DO_TRAIN} eval=${DO_EVAL}"
echo "  data:     ${DATA_PATH}"
echo "  arch:     v18_orb (explicit amp, amp_rank 16, NO residual) + --grad_clip ${GRAD_CLIP}"
echo "  seed:     ${SEED}"
[ -n "$EXCLUDE_NODES" ] && echo "  exclude:  ${EXCLUDE_NODES}"
echo ""

if [ ! -f "${DATA_PATH}" ]; then
  echo "ERROR: dataset missing at ${DATA_PATH}"
  echo "v18b reuses the v13 dt=0.05 dataset; run slurm/regression_v13.sh --data first."
  exit 1
fi

# v18_orb architecture + restored joint gradient clipping. This is the ONLY
# delta vs the collapsed v18_orb run (slurm/regression_v18.sh, v18_orb variant).
VARIANT_FLAGS="--adaptive_bandwidth --omega_op_floor ${OMEGA_OP_FLOOR} --soft_omega_floor --explicit_amplitude --amp_rank 16 --use_orb_features --standardize_orb_energies --grad_clip ${GRAD_CLIP}"

SEED_TAG="${TAG}_orb_clip_s${SEED}"
MODEL_DIR="results/fermionic_pipeline/regression/${SEED_TAG}_model"
CKPT="${MODEL_DIR}/regressor.pt"
JOB_TRAIN=""
SUBMITTED=()

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
  echo "[submitted] train orb_clip seed=${SEED}: job ${JOB_TRAIN}"
  SUBMITTED+=("orb_clip seed=${SEED}: train=${JOB_TRAIN}")
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
  echo "[submitted] eval  orb_clip seed=${SEED}: job ${JOB_EVAL}${DEP_NOTE}"
  SUBMITTED+=("orb_clip seed=${SEED}: eval=${JOB_EVAL}")
fi

if [ "${#SUBMITTED[@]}" -gt 0 ]; then
  echo ""
  echo "=== Submitted ==="
  for line in "${SUBMITTED[@]}"; do
    echo "  ${line}"
  done
  echo ""
  echo "After eval lands, compare against the collapsed v18_orb_s42:"
  echo "  composition envelope_pearson (short→long) was: 0.007 0.003 0.049 0.167 0.631"
  echo "  phase_err_mean (short→long) was:               1.083 1.178 1.637 1.523 0.925"
  echo "  overall temporal Pearson was 0.339 (vs v15_explicit 0.915)."
fi
