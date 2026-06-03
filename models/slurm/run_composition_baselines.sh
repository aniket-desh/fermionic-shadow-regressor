#!/bin/bash
# =======================================================================
# Composition diagnostic baselines: v10 + v11 (both seeds).
#
# Back-fills the new per-observable composition diagnostic on the existing
# v10 / v11 checkpoints so v12 sweep results have something to compare to.
# Cheap — minutes per checkpoint.
#
# Usage (on Trillium):
#   cd $SCRATCH/generative-quantum-states
#   bash slurm/run_composition_baselines.sh
# =======================================================================
set -euo pipefail

PARTITION="compute_full_node"
ACCOUNT="rrg-aspuru"

# (tag | dataset path | model dir) tuples
JOBS=(
  "v10|results/fermionic_pipeline/regression/h4_regress_v10/regression_targets.h5|results/fermionic_pipeline/regression/h4_regress_v10_model"
  "v11_s42|results/fermionic_pipeline/regression/h4_regress_v11/regression_targets.h5|results/fermionic_pipeline/regression/h4_regress_v11_s42_model"
  "v11_s1729|results/fermionic_pipeline/regression/h4_regress_v11/regression_targets.h5|results/fermionic_pipeline/regression/h4_regress_v11_s1729_model"
)

mkdir -p logs

echo "=== Composition diagnostic baselines ==="

SUBMITTED=()
for spec in "${JOBS[@]}"; do
  IFS='|' read -r TAG DATA_PATH MODEL_DIR <<< "$spec"
  CKPT="${MODEL_DIR}/regressor.pt"

  if [ ! -f "${DATA_PATH}" ]; then
    echo "ERROR: dataset missing at ${DATA_PATH}"
    exit 1
  fi
  if [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint missing at ${CKPT}"
    exit 1
  fi

  cat > "slurm/_comp_diag_${TAG}.sh" << 'EOF'
#!/bin/bash
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2
source "$HOME/envs/gqs/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
cd "$SCRATCH/generative-quantum-states"
export PYTHONUNBUFFERED=1
EOF
  cat >> "slurm/_comp_diag_${TAG}.sh" << EOF
python3 -m fermionic_pipeline.eval.composition_diagnostic \\
  --data_path ${DATA_PATH} \\
  --checkpoint ${CKPT} \\
  --save_dir ${MODEL_DIR}/eval \\
  --device cuda
EOF
  chmod +x "slurm/_comp_diag_${TAG}.sh"

  JOB=$(sbatch --parsable \
    --partition=${PARTITION} \
    --account=${ACCOUNT} \
    --job-name="comp-diag-${TAG}" \
    --output="logs/comp_diag_${TAG}_%j.out" \
    --error="logs/comp_diag_${TAG}_%j.err" \
    --time=02:00:00 \
    --gpus-per-node=4 \
    --cpus-per-task=4 \
    "slurm/_comp_diag_${TAG}.sh")
  echo "[submitted] ${TAG}: job ${JOB}"
  SUBMITTED+=("${TAG}: ${JOB}")
done

echo ""
echo "=== Submitted ==="
for line in "${SUBMITTED[@]}"; do
  echo "  ${line}"
done
