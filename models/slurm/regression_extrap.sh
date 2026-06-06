#!/bin/bash
# Regression extrapolation study (manuscript pillar 1 / Section-2 conditions).
#
# WHY: Luis asked for the coherence heatmap with a training-data bounding box,
#   showing performance INSIDE vs OUTSIDE the box and whether it matches Section 2
#   (Prop 1 = in-box learnability only; Prop 2 = beyond-horizon aliasing + the
#   high-frequency limit that bites at small R). The training data live on
#   R in [0.5, 3.0], t in [0, 300]; this builds an EXTENDED grid that brackets
#   that box (compressed R<0.5, stretched R>3.0, and double the horizon t<=600),
#   evaluates the shipped v18-orb model on it, and draws the box on the heatmap.
#
# Coarser than the training grid (r_step 0.05 vs 0.01) — enough for the heatmap,
# ~half the v13 datagen cost, comfortably inside the 24h ceiling. Datagen is
# CPU-bound: BLAS threads are pinned to 1/worker and parallelised over n_workers.
#
# Usage (from $SCRATCH/generative-quantum-states):
#   bash slurm/regression_extrap.sh --all                 # datagen -> heatmap (chained)
#   bash slurm/regression_extrap.sh --data                # datagen + omega_op only
#   bash slurm/regression_extrap.sh --eval                # heatmap only (dataset must exist)
#   bash slurm/regression_extrap.sh --all --exclude trig0019,trig0034
set -euo pipefail

ACCOUNT="rrg-aspuru"
PARTITION="compute_full_node"
TAG="h4_regress_extrap"
# Shipped model = the v18-orb HF artifact (orb arm, seed 42).
CKPT="results/fermionic_pipeline/regression/h4_regress_v18_v18_orb_s42_model/regressor.pt"
# Extended grid (brackets the training box R[0.5,3.0] t[0,300]); dt=0.05.
R_START=0.30; R_END=3.50; R_STEP=0.05
T_MAX=600.0;  N_TIMES=12001; N_Q=500; N_WORKERS=16
# Training box drawn on the heatmap:
TRAIN_R="0.5 3.0"; TRAIN_T="0 300"
EXCLUDE_NODES=""
DO_DATA=false; DO_EVAL=false

while [ $# -gt 0 ]; do
  case "$1" in
    --data)       DO_DATA=true ;;
    --eval)       DO_EVAL=true ;;
    --all)        DO_DATA=true; DO_EVAL=true ;;
    --tag)        TAG="$2"; shift ;;
    --checkpoint) CKPT="$2"; shift ;;
    --exclude)    EXCLUDE_NODES="$2"; shift ;;
    -h|--help)    grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
    *)            echo "unknown arg: $1"; exit 1 ;;
  esac
  shift
done
if [ "$DO_DATA" = false ] && [ "$DO_EVAL" = false ]; then DO_DATA=true; DO_EVAL=true; fi

DATA_PATH="results/fermionic_pipeline/regression/${TAG}/regression_targets.h5"
SAVE_DIR="results/fermionic_pipeline/regression/${TAG}/plots"

echo "=== Regression extrapolation (${TAG}) ==="
echo "  grid:   R[${R_START},${R_END}] step ${R_STEP}  |  t[0,${T_MAX}] n_times ${N_TIMES}  |  n_q ${N_Q}"
echo "  box:    train R[${TRAIN_R}] t[${TRAIN_T}]"
echo "  stages: data=${DO_DATA} eval=${DO_EVAL}"
[ -n "$EXCLUDE_NODES" ] && echo "  exclude: ${EXCLUDE_NODES}"

# ── Stage: datagen + omega_op (24h) ───────────────────────────────────────
JOB_DATA=""
if [ "$DO_DATA" = true ]; then
  cat > "slurm/_extrap_data_${TAG}.sh" << 'EOF'
#!/bin/bash
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2
source "$HOME/envs/gqs/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
cd "$SCRATCH/generative-quantum-states"
export PYTHONUNBUFFERED=1
# Pin BLAS to 1 thread/worker (else NumPy/OpenBLAS forks ~96 threads each and
# thrashes the node) — this is what keeps datagen fast enough to finish on time.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
EOF
  cat >> "slurm/_extrap_data_${TAG}.sh" << EOF
python3 -m fermionic_pipeline.data.regression_dataset \\
  --output ${DATA_PATH} \\
  --n_atoms 4 \\
  --r_start ${R_START} --r_end ${R_END} \\
  --r_step ${R_STEP} \\
  --t_max ${T_MAX} --n_times ${N_TIMES} --n_q ${N_Q} \\
  --n_workers ${N_WORKERS}
python3 -m fermionic_pipeline.data.compute_omega_op \\
  --data_path ${DATA_PATH}
EOF
  chmod +x "slurm/_extrap_data_${TAG}.sh"

  SBATCH_D=(
    --parsable
    --partition="${PARTITION}"
    --job-name="extrap-data-${TAG}"
    --output="logs/extrap_data_${TAG}_%j.out"
    --error="logs/extrap_data_${TAG}_%j.err"
    --time=24:00:00
    --gpus-per-node=4
    --cpus-per-task=16
    --account="${ACCOUNT}"
  )
  [ -n "$EXCLUDE_NODES" ] && SBATCH_D+=(--exclude="${EXCLUDE_NODES}")
  JOB_DATA=$(sbatch "${SBATCH_D[@]}" "slurm/_extrap_data_${TAG}.sh")
  echo "[submitted] datagen+omega_op: job ${JOB_DATA} (24h)"
fi

# ── Stage: extrapolation heatmap (depends on datagen) ─────────────────────
if [ "$DO_EVAL" = true ]; then
  if [ -z "$JOB_DATA" ] && [ ! -f "${DATA_PATH}" ]; then
    echo "ERROR: dataset missing at ${DATA_PATH}; pass --data (or --all) first."
    exit 1
  fi
  cat > "slurm/_extrap_eval_${TAG}.sh" << 'EOF'
#!/bin/bash
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2
source "$HOME/envs/gqs/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
cd "$SCRATCH/generative-quantum-states"
export PYTHONUNBUFFERED=1
EOF
  cat >> "slurm/_extrap_eval_${TAG}.sh" << EOF
python3 -m fermionic_pipeline.eval.extrapolation_heatmap \\
  --data_path ${DATA_PATH} \\
  --checkpoint ${CKPT} \\
  --save_dir ${SAVE_DIR} \\
  --train_r_range ${TRAIN_R} --train_t_range ${TRAIN_T} \\
  --device cuda
EOF
  chmod +x "slurm/_extrap_eval_${TAG}.sh"

  SBATCH_E=(
    --parsable
    --partition="${PARTITION}"
    --job-name="extrap-eval-${TAG}"
    --output="logs/extrap_eval_${TAG}_%j.out"
    --error="logs/extrap_eval_${TAG}_%j.err"
    --time=02:00:00
    --gpus-per-node=4
    --cpus-per-task=4
    --account="${ACCOUNT}"
  )
  [ -n "$JOB_DATA" ] && SBATCH_E+=(--dependency="afterok:${JOB_DATA}")
  [ -n "$EXCLUDE_NODES" ] && SBATCH_E+=(--exclude="${EXCLUDE_NODES}")
  JOB_EVAL=$(sbatch "${SBATCH_E[@]}" "slurm/_extrap_eval_${TAG}.sh")
  echo "[submitted] extrapolation heatmap: job ${JOB_EVAL}${JOB_DATA:+ (afterok:${JOB_DATA})}"
  echo "  output: ${SAVE_DIR}/coherence_heatmap.pdf"
fi
