#!/bin/bash
#SBATCH --job-name=h4-experiment
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --partition=compute_full_node
#SBATCH --account=rrg-aspuru
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aniketrd

# ── Config (override via environment) ────────────────────────────
TAG=${TAG:-full_sweep}
CHAIN_LENGTHS=${CHAIN_LENGTHS:-"6 8 10"}

# ── Environment ─────────────────────────────────────────────────
module load StdEnv/2023 python/3.11 cuda/12.2 scipy-stack/2024a
source "$HOME/envs/gqs/bin/activate"

WORKDIR="$SCRATCH/generative-quantum-states"
cd "$WORKDIR"

echo "=== H Chain Experiment: tag=$TAG  chains=$CHAIN_LENGTHS ==="

python h4_experiment.py \
    --chain_lengths $CHAIN_LENGTHS \
    --r_step 0.025 --n_test 20 --shots 5000 \
    --d_model 128 --n_layers 2 --n_heads 4 --d_ff 256 \
    --iterations 25000 --batch_size 256 \
    --lr 1e-3 --final_lr 1e-7 --warmup_frac 0.05 \
    --eval_samples 10000 --device cuda --tag "$TAG"

echo "=== H Chain Experiment complete ==="
