#!/bin/bash
# =======================================================================
# H2 FiLM transformer with Fourier time embedding
#
# This is the key test: direct shadows already produce clean spectral
# peaks on H2. If Fourier time features fix the time representation,
# synthetic shadows should now also produce spectral peaks.
#
# Usage: cd $SCRATCH/generative-quantum-states && bash slurm/h2_fourier.sh
# =======================================================================
set -euo pipefail

TAG="${1:-h2-fourier-v1}"
CONFIG="fermionic_pipeline/configs/h2_fourier.yaml"

PARTITION="compute_full_node"
ACCOUNT="rrg-aspuru"

echo "=== H2 Fourier Time Embedding ==="
echo "  tag:    ${TAG}"
echo "  config: ${CONFIG}"
echo ""

# Use the existing fermionic_pipeline.sh pattern: datagen + train + eval in one job
cat > "slurm/_run_${TAG}.sh" << EOF
#!/bin/bash
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2 scipy-stack/2024a
source "\$HOME/envs/gqs/bin/activate"
cd "\$SCRATCH/generative-quantum-states"
export PYTHONUNBUFFERED=1

echo "=== [1/2] Data Generation + Training ==="
python3 -m fermionic_pipeline.scripts.train \\
  --config ${CONFIG} \\
  --device cuda \\
  --tag ${TAG} \\
  --n_workers 8

SAVE_DIR="results/fermionic_pipeline/${TAG}/H2"

echo ""
echo "=== [2/2] Evaluation ==="
python3 -m fermionic_pipeline.scripts.evaluate \\
  --config ${CONFIG} \\
  --checkpoint "\${SAVE_DIR}/checkpoint.pt" \\
  --device cuda \\
  --n_workers 8 \\
  --save_dir "\${SAVE_DIR}"

echo ""
echo "=== Done ==="
EOF
chmod +x "slurm/_run_${TAG}.sh"

JOB=$(sbatch --parsable \
  --partition=${PARTITION} \
  --job-name="${TAG}" \
  --output="logs/${TAG}_%j.out" \
  --error="logs/${TAG}_%j.err" \
  --time=24:00:00 \
  --gpus-per-node=4 \
  --cpus-per-task=8 \
  --account=${ACCOUNT} \
  "slurm/_run_${TAG}.sh")
echo "[submitted] job ${JOB}"
echo "Monitor: squeue -u \$USER"
echo "Results: \$SCRATCH/generative-quantum-states/results/fermionic_pipeline/${TAG}/H2/"
