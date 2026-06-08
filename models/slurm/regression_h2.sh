#!/bin/bash
# Train a NEW-architecture (v18-orb) FSR on H2, so we can run the R/t
# extrapolation study on H2 with the current model (the only existing H2 model is
# the OLD v10 arch on HF — what the collaborator ran). H2 is single-reference and
# tiny (4 qubits, 28 degree-2 Majorana observables), so this is cheap.
#
# Same architecture + training box as the H4 v18-orb run (R[0.5,3.0], t[0,300],
# dt=0.05), so the H2 result is methodologically consistent with the H4 extrap.
# After this completes, run the extrapolation eval with:
#   bash slurm/regression_extrap.sh --all --n_atoms 2 \
#     --tag h2_regress_extrap \
#     --checkpoint results/fermionic_pipeline/regression/h2_regress_v18_orb_s42_model/regressor.pt
#
# Usage (from $SCRATCH/generative-quantum-states):
#   bash slurm/regression_h2.sh --all              # datagen+omega_op -> train (chained)
#   bash slurm/regression_h2.sh --data             # datagen only
#   bash slurm/regression_h2.sh --train            # train only (dataset must exist)
#   bash slurm/regression_h2.sh --all --exclude trig0019,trig0034
set -euo pipefail

ACCOUNT="rrg-aspuru"
PARTITION="compute_full_node"
TAG="h2_regress_v18"
SEED=42
EXCLUDE_NODES=""
DO_DATA=false; DO_TRAIN=false

while [ $# -gt 0 ]; do
  case "$1" in
    --data)     DO_DATA=true ;;
    --train)    DO_TRAIN=true ;;
    --all)      DO_DATA=true; DO_TRAIN=true ;;
    --tag)      TAG="$2"; shift ;;
    --seed)     SEED="$2"; shift ;;
    --exclude)  EXCLUDE_NODES="$2"; shift ;;
    -h|--help)  grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
    *)          echo "unknown arg: $1"; exit 1 ;;
  esac
  shift
done
if [ "$DO_DATA" = false ] && [ "$DO_TRAIN" = false ]; then DO_DATA=true; DO_TRAIN=true; fi

DATA_PATH="results/fermionic_pipeline/regression/${TAG}/regression_targets.h5"
MODEL_DIR="results/fermionic_pipeline/regression/${TAG}_orb_s${SEED}_model"

echo "=== H2 v18-orb training (${TAG}) ==="
echo "  data:   ${DATA_PATH}  (H2, R[0.5,3.0] step 0.01, t[0,300] dt=0.05, n_q 500)"
echo "  model:  ${MODEL_DIR}  (explicit-amp + adaptive bw + orb, grad_clip 1.0, seed ${SEED})"
echo "  stages: data=${DO_DATA} train=${DO_TRAIN}"
[ -n "$EXCLUDE_NODES" ] && echo "  exclude: ${EXCLUDE_NODES}"

# ── Stage: in-box datagen + omega_op (CPU, 24h) ───────────────────────────
JOB_DATA=""
if [ "$DO_DATA" = true ]; then
  cat > "slurm/_h2_data_${TAG}.sh" << 'EOF'
#!/bin/bash
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2
source "$HOME/envs/gqs/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
cd "$SCRATCH/generative-quantum-states"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
EOF
  cat >> "slurm/_h2_data_${TAG}.sh" << EOF
python3 -m fermionic_pipeline.data.regression_dataset \\
  --output ${DATA_PATH} \\
  --n_atoms 2 \\
  --r_start 0.5 --r_end 3.0 --r_step 0.01 \\
  --t_max 300.0 --n_times 6001 --n_q 500 \\
  --n_workers 16
python3 -m fermionic_pipeline.data.compute_omega_op --data_path ${DATA_PATH}
EOF
  chmod +x "slurm/_h2_data_${TAG}.sh"
  SBATCH_D=(--parsable --partition="${PARTITION}" --job-name="h2-data-${TAG}"
    --output="logs/h2_data_${TAG}_%j.out" --error="logs/h2_data_${TAG}_%j.err"
    --time=24:00:00 --gpus-per-node=4 --cpus-per-task=16 --account="${ACCOUNT}")
  [ -n "$EXCLUDE_NODES" ] && SBATCH_D+=(--exclude="${EXCLUDE_NODES}")
  JOB_DATA=$(sbatch "${SBATCH_D[@]}" "slurm/_h2_data_${TAG}.sh")
  echo "[submitted] H2 datagen+omega_op: job ${JOB_DATA}"
fi

# ── Stage: train (GPU; v18-orb recipe, identical to H4) ────────────────────
if [ "$DO_TRAIN" = true ]; then
  if [ -z "$JOB_DATA" ] && [ ! -f "${DATA_PATH}" ]; then
    echo "ERROR: dataset missing at ${DATA_PATH}; pass --data (or --all) first."
    exit 1
  fi
  cat > "slurm/_h2_train_${TAG}.sh" << 'EOF'
#!/bin/bash
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2
source "$HOME/envs/gqs/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
cd "$SCRATCH/generative-quantum-states"
export PYTHONUNBUFFERED=1
EOF
  cat >> "slurm/_h2_train_${TAG}.sh" << EOF
python3 -m fermionic_pipeline.training.regressor_trainer \\
  --data_path ${DATA_PATH} \\
  --save_dir ${MODEL_DIR} \\
  --device cuda --seed ${SEED} \\
  --steps 150000 --batch_size 256 --lr 1e-3 --final_lr 1e-7 --warmup_frac 0.05 \\
  --weight_decay 5e-4 --d_hidden 768 --n_layers 6 --n_fourier 256 --fourier_scale 20.0 \\
  --conditioned_frequencies --freq_net_hidden 128 --freq_net_layers 3 \\
  --adaptive_bandwidth --omega_op_floor 8.0 --soft_omega_floor \\
  --explicit_amplitude --amp_rank 16 --use_orb_features --standardize_orb_energies \\
  --grad_clip 1.0 --alpha_corr 1.0 --eval_every 2000
EOF
  chmod +x "slurm/_h2_train_${TAG}.sh"
  SBATCH_T=(--parsable --partition="${PARTITION}" --job-name="h2-train-${TAG}"
    --output="logs/h2_train_${TAG}_%j.out" --error="logs/h2_train_${TAG}_%j.err"
    --time=01:30:00 --gpus-per-node=4 --cpus-per-task=4 --account="${ACCOUNT}")
  [ -n "$JOB_DATA" ] && SBATCH_T+=(--dependency="afterok:${JOB_DATA}")
  [ -n "$EXCLUDE_NODES" ] && SBATCH_T+=(--exclude="${EXCLUDE_NODES}")
  JOB_TRAIN=$(sbatch "${SBATCH_T[@]}" "slurm/_h2_train_${TAG}.sh")
  echo "[submitted] H2 train: job ${JOB_TRAIN}${JOB_DATA:+ (afterok:${JOB_DATA})}"
  echo "  checkpoint -> ${MODEL_DIR}/regressor.pt"
fi
