#!/bin/bash
#SBATCH --job-name=ferm-pipeline
#SBATCH --output=logs/ferm-pipeline_%j.out
#SBATCH --error=logs/ferm-pipeline_%j.err
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account=rrg-aspuru
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aniketrd

# ── Config ───────────────────────────────────────────────────────
# Trillium uses --export=NONE, so env vars from the shell are stripped.
# Pass via: sbatch --export=ALL,MOLECULE=h4_fast,TAG=fast slurm/fermionic_pipeline.sh
# Or just edit these defaults directly.
MOLECULE=${MOLECULE:-h4}
TAG=${TAG:-prod}
DEVICE=${DEVICE:-cuda}
SKIP_DATAGEN=${SKIP_DATAGEN:-false}
EVAL_ONLY=${EVAL_ONLY:-false}
N_WORKERS=${N_WORKERS:-8}

# ── Environment ─────────────────────────────────────────────────
module load StdEnv/2023 python/3.11 cuda/12.2 scipy-stack/2024a
source "$HOME/envs/gqs/bin/activate"

WORKDIR="$SCRATCH/generative-quantum-states"
cd "$WORKDIR"
mkdir -p logs

# Force unbuffered Python output so logs appear in real time
export PYTHONUNBUFFERED=1

CONFIG="fermionic_pipeline/configs/${MOLECULE}.yaml"
N_ATOMS=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['data']['n_atoms'])" 2>/dev/null || echo "${MOLECULE#h}")
SAVE_DIR="results/fermionic_pipeline/${TAG}"
DATA_PATH="${SAVE_DIR}/H${N_ATOMS}/shadow_data.h5"
CKPT_PATH="${SAVE_DIR}/H${N_ATOMS}/checkpoint.pt"

echo "=== Fermionic Pipeline ==="
echo "  molecule: ${MOLECULE}"
echo "  config:   ${CONFIG}"
echo "  tag:      ${TAG}"
echo "  device:   ${DEVICE}"
echo "  job:      ${SLURM_JOB_ID}"
echo ""

# ── Step 1+2: Data generation + Training ─────────────────────────
if [ "$EVAL_ONLY" = "true" ]; then
    echo "=== Skipping datagen + training (EVAL_ONLY=true) ==="
elif [ "$SKIP_DATAGEN" = "false" ]; then
    echo "=== [1/3] Data Generation + Training ==="
    python -m fermionic_pipeline.scripts.train \
        --config "$CONFIG" \
        --device "$DEVICE" \
        --tag "$TAG" \
        --n_workers "$N_WORKERS"
    echo "=== Data generation + training complete ==="
else
    echo "=== [1/3] Skipping datagen (SKIP_DATAGEN=true) ==="
    echo "=== [2/3] Training only ==="
    python -m fermionic_pipeline.scripts.train \
        --config "$CONFIG" \
        --skip_datagen \
        --data_path "$DATA_PATH" \
        --device "$DEVICE" \
        --tag "$TAG"
fi

# ── Step 3: Evaluate ────────────────────────────────────────────
echo ""
echo "=== [3/3] Evaluation ==="
python -m fermionic_pipeline.scripts.evaluate \
    --config "$CONFIG" \
    --checkpoint "$CKPT_PATH" \
    --device "$DEVICE" \
    --n_workers "$N_WORKERS" \
    --save_dir "${SAVE_DIR}/H${N_ATOMS}"

echo ""
echo "=== Fermionic Pipeline complete ==="
echo "Results at: ${SAVE_DIR}"
