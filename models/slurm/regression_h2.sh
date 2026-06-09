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
# NOTE (6/09): the first H2 model (variant=orb) hit the train-vs-eval gap —
# val_corr 0.9996 but in-box temporal r̄ 0.79, weak at stretched R>2.3. Knobs added
# to address it WITHOUT a blind sweep: --alpha_temporal_corr (make the loss track
# the temporal deliverable, not the saturated val_corr) and --omega_op_floor (tune
# to H2's omega_op scale; 8.0 was an H4 value). --variant names the config so they
# don't overwrite. --diag runs the composition diagnostic (envelope vs phase).
#
# Usage (from $SCRATCH/generative-quantum-states):
#   bash slurm/regression_h2.sh --all              # datagen+omega_op -> train (chained)
#   bash slurm/regression_h2.sh --train            # train only (dataset must exist)
#   bash slurm/regression_h2.sh --diag --variant orb              # diagnose the existing model
#   bash slurm/regression_h2.sh --train --diag --variant tcorr \
#        --alpha_temporal_corr 1.0                 # retrain w/ temporal-corr loss + diagnose
#   bash slurm/regression_h2.sh --train --variant floor2 --omega_op_floor 2.0
#   bash slurm/regression_h2.sh --all --exclude trig0019,trig0034
#
# Inspect H2's omega_op scale first (to set --omega_op_floor):
#   python3 -c "import h5py;f=h5py.File('results/fermionic_pipeline/regression/h2_regress_v18/regression_targets.h5');w=f['omega_op'][:];print('omega_op min %.2f max %.2f mean %.2f'%(w.min(),w.max(),w.mean()))"
set -euo pipefail

ACCOUNT="rrg-aspuru"
PARTITION="compute_full_node"
TAG="h2_regress_v18"
SEED=42
EXCLUDE_NODES=""
# Tunable knobs (the H2 train-vs-eval gap: val_corr 0.9996 but temporal r̄ 0.79).
# VARIANT suffixes the model dir so configs don't overwrite each other.
VARIANT="orb"
ALPHA_CORR=1.0          # cross-observable corr (the v18-orb default)
ALPHA_TCORR=0.0         # temporal corr — set >0 so the loss tracks the DELIVERABLE
OMEGA_FLOOR=8.0         # adaptive-bandwidth floor; tune to H2's omega_op scale
DO_DATA=false; DO_TRAIN=false; DO_DIAG=false

while [ $# -gt 0 ]; do
  case "$1" in
    --data)                  DO_DATA=true ;;
    --train)                 DO_TRAIN=true ;;
    --diag)                  DO_DIAG=true ;;   # composition diagnostic (envelope vs phase) after train
    --all)                   DO_DATA=true; DO_TRAIN=true ;;
    --tag)                   TAG="$2"; shift ;;
    --seed)                  SEED="$2"; shift ;;
    --variant)               VARIANT="$2"; shift ;;
    --alpha_corr)            ALPHA_CORR="$2"; shift ;;
    --alpha_temporal_corr)   ALPHA_TCORR="$2"; shift ;;
    --omega_op_floor)        OMEGA_FLOOR="$2"; shift ;;
    --exclude)               EXCLUDE_NODES="$2"; shift ;;
    -h|--help)               grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
    *)                       echo "unknown arg: $1"; exit 1 ;;
  esac
  shift
done
if [ "$DO_DATA" = false ] && [ "$DO_TRAIN" = false ] && [ "$DO_DIAG" = false ]; then DO_DATA=true; DO_TRAIN=true; fi

DATA_PATH="results/fermionic_pipeline/regression/${TAG}/regression_targets.h5"
MODEL_DIR="results/fermionic_pipeline/regression/${TAG}_${VARIANT}_s${SEED}_model"

echo "=== H2 v18 training (${TAG}, variant=${VARIANT}) ==="
echo "  data:   ${DATA_PATH}  (H2, R[0.5,3.0] step 0.01, t[0,300] dt=0.05, n_q 500)"
echo "  model:  ${MODEL_DIR}"
echo "  loss:   alpha_corr=${ALPHA_CORR}  alpha_temporal_corr=${ALPHA_TCORR}  omega_op_floor=${OMEGA_FLOOR}"
echo "  stages: data=${DO_DATA} train=${DO_TRAIN} diag=${DO_DIAG}"
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

# ── Stage: train (GPU; v18 explicit-amp recipe, knobs configurable) ────────
ST="${TAG}_${VARIANT}"          # per-config tag so generated scripts/logs don't clobber
JOB_TRAIN=""
if [ "$DO_TRAIN" = true ]; then
  if [ -z "$JOB_DATA" ] && [ ! -f "${DATA_PATH}" ]; then
    echo "ERROR: dataset missing at ${DATA_PATH}; pass --data (or --all) first."
    exit 1
  fi
  cat > "slurm/_h2_train_${ST}.sh" << 'EOF'
#!/bin/bash
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2
source "$HOME/envs/gqs/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
cd "$SCRATCH/generative-quantum-states"
export PYTHONUNBUFFERED=1
EOF
  cat >> "slurm/_h2_train_${ST}.sh" << EOF
python3 -m fermionic_pipeline.training.regressor_trainer \\
  --data_path ${DATA_PATH} \\
  --save_dir ${MODEL_DIR} \\
  --device cuda --seed ${SEED} \\
  --steps 150000 --batch_size 256 --lr 1e-3 --final_lr 1e-7 --warmup_frac 0.05 \\
  --weight_decay 5e-4 --d_hidden 768 --n_layers 6 --n_fourier 256 --fourier_scale 20.0 \\
  --conditioned_frequencies --freq_net_hidden 128 --freq_net_layers 3 \\
  --adaptive_bandwidth --omega_op_floor ${OMEGA_FLOOR} --soft_omega_floor \\
  --explicit_amplitude --amp_rank 16 --use_orb_features --standardize_orb_energies \\
  --grad_clip 1.0 --alpha_corr ${ALPHA_CORR} --alpha_temporal_corr ${ALPHA_TCORR} --eval_every 2000
EOF
  chmod +x "slurm/_h2_train_${ST}.sh"
  SBATCH_T=(--parsable --partition="${PARTITION}" --job-name="h2-train-${ST}"
    --output="logs/h2_train_${ST}_%j.out" --error="logs/h2_train_${ST}_%j.err"
    --time=01:30:00 --gpus-per-node=4 --cpus-per-task=4 --account="${ACCOUNT}")
  [ -n "$JOB_DATA" ] && SBATCH_T+=(--dependency="afterok:${JOB_DATA}")
  [ -n "$EXCLUDE_NODES" ] && SBATCH_T+=(--exclude="${EXCLUDE_NODES}")
  JOB_TRAIN=$(sbatch "${SBATCH_T[@]}" "slurm/_h2_train_${ST}.sh")
  echo "[submitted] H2 train: job ${JOB_TRAIN}${JOB_DATA:+ (afterok:${JOB_DATA})}"
  echo "  checkpoint -> ${MODEL_DIR}/regressor.pt"
fi

# ── Stage: composition diagnostic (localize envelope vs phase failure) ─────
if [ "$DO_DIAG" = true ]; then
  if [ -z "$JOB_TRAIN" ] && [ ! -f "${MODEL_DIR}/regressor.pt" ]; then
    echo "ERROR: checkpoint missing at ${MODEL_DIR}/regressor.pt; train first (or --train --diag)."
    exit 1
  fi
  cat > "slurm/_h2_diag_${ST}.sh" << 'EOF'
#!/bin/bash
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2
source "$HOME/envs/gqs/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
cd "$SCRATCH/generative-quantum-states"
export PYTHONUNBUFFERED=1
EOF
  cat >> "slurm/_h2_diag_${ST}.sh" << EOF
python3 -m fermionic_pipeline.eval.composition_diagnostic \\
  --data_path ${DATA_PATH} \\
  --checkpoint ${MODEL_DIR}/regressor.pt \\
  --save_dir ${MODEL_DIR}/eval \\
  --device cuda
EOF
  chmod +x "slurm/_h2_diag_${ST}.sh"
  SBATCH_G=(--parsable --partition="${PARTITION}" --job-name="h2-diag-${ST}"
    --output="logs/h2_diag_${ST}_%j.out" --error="logs/h2_diag_${ST}_%j.err"
    --time=00:40:00 --gpus-per-node=4 --cpus-per-task=4 --account="${ACCOUNT}")
  [ -n "$JOB_TRAIN" ] && SBATCH_G+=(--dependency="afterok:${JOB_TRAIN}")
  [ -n "$EXCLUDE_NODES" ] && SBATCH_G+=(--exclude="${EXCLUDE_NODES}")
  JOB_DIAG=$(sbatch "${SBATCH_G[@]}" "slurm/_h2_diag_${ST}.sh")
  echo "[submitted] H2 composition diagnostic: job ${JOB_DIAG}${JOB_TRAIN:+ (afterok:${JOB_TRAIN})}"
  echo "  -> ${MODEL_DIR}/eval/composition_diagnostic.json"
fi
