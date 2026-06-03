#!/bin/bash
# =======================================================================
# Observable expectation loss experiments: H2 + H4
#
# Usage: cd $SCRATCH/generative-quantum-states && bash slurm/obsloss.sh
# =======================================================================
set -euo pipefail

PARTITION="compute_full_node"
ACCOUNT="rrg-aspuru"

echo "=== Observable Expectation Loss Experiments ==="
echo ""

# ── H2: full pipeline (datagen + train + eval) ──────────────────
TAG_H2="h2-obsloss-v1"
cat > "slurm/_run_${TAG_H2}.sh" << 'ENDSCRIPT'
#!/bin/bash
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2 scipy-stack/2024a
source "$HOME/envs/gqs/bin/activate"
cd "$SCRATCH/generative-quantum-states"
export PYTHONUNBUFFERED=1

CONFIG="fermionic_pipeline/configs/h2_obsloss.yaml"
TAG="h2-obsloss-v1"
SAVE_DIR="results/fermionic_pipeline/${TAG}/H2"

echo "=== [1/2] Data + Train ==="
python3 -m fermionic_pipeline.scripts.train \
  --config "$CONFIG" --device cuda --tag "$TAG" --n_workers 8

echo ""
echo "=== [2/2] Evaluate ==="
python3 -m fermionic_pipeline.scripts.evaluate \
  --config "$CONFIG" \
  --checkpoint "${SAVE_DIR}/checkpoint.pt" \
  --device cuda --n_workers 8 --save_dir "${SAVE_DIR}"

echo "=== H2 Done ==="
ENDSCRIPT
chmod +x "slurm/_run_${TAG_H2}.sh"

JOB_H2=$(sbatch --parsable \
  --partition=${PARTITION} \
  --job-name="${TAG_H2}" \
  --output="logs/${TAG_H2}_%j.out" \
  --error="logs/${TAG_H2}_%j.err" \
  --time=24:00:00 \
  --gpus-per-node=4 \
  --cpus-per-task=8 \
  --account=${ACCOUNT} \
  "slurm/_run_${TAG_H2}.sh")
echo "[submitted] H2 obsloss: job ${JOB_H2}"

# ── H4: full pipeline ───────────────────────────────────────────
TAG_H4="h4-obsloss-v1"
cat > "slurm/_run_${TAG_H4}.sh" << 'ENDSCRIPT'
#!/bin/bash
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2 scipy-stack/2024a
source "$HOME/envs/gqs/bin/activate"
cd "$SCRATCH/generative-quantum-states"
export PYTHONUNBUFFERED=1

CONFIG="fermionic_pipeline/configs/h4_obsloss.yaml"
TAG="h4-obsloss-v1"
SAVE_DIR="results/fermionic_pipeline/${TAG}/H4"

echo "=== [1/2] Data + Train ==="
python3 -m fermionic_pipeline.scripts.train \
  --config "$CONFIG" --device cuda --tag "$TAG" --n_workers 8

echo ""
echo "=== [2/2] Evaluate ==="
python3 -m fermionic_pipeline.scripts.evaluate \
  --config "$CONFIG" \
  --checkpoint "${SAVE_DIR}/checkpoint.pt" \
  --device cuda --n_workers 8 --save_dir "${SAVE_DIR}"

echo "=== H4 Done ==="
ENDSCRIPT
chmod +x "slurm/_run_${TAG_H4}.sh"

JOB_H4=$(sbatch --parsable \
  --partition=${PARTITION} \
  --job-name="${TAG_H4}" \
  --output="logs/${TAG_H4}_%j.out" \
  --error="logs/${TAG_H4}_%j.err" \
  --time=24:00:00 \
  --gpus-per-node=4 \
  --cpus-per-task=8 \
  --account=${ACCOUNT} \
  "slurm/_run_${TAG_H4}.sh")
echo "[submitted] H4 obsloss: job ${JOB_H4}"

echo ""
echo "=== Submitted ==="
echo "  H2: ${JOB_H2}"
echo "  H4: ${JOB_H4}"
echo "Monitor: squeue -u \$USER"
