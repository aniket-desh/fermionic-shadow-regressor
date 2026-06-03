#!/bin/bash
# Orchestrator: submits all jobs with dependencies so they run in sequence.
# Run this from $SCRATCH/generative-quantum-states on Trillium.
#
# Usage: bash slurm/run_all.sh [ROWS] [COLS]
# Example: bash slurm/run_all.sh 4 4

set -e

ROWS=${1:-4}
COLS=${2:-4}

export ROWS COLS

WORKDIR="$SCRATCH/generative-quantum-states"
cd "$WORKDIR"

echo "=== Heisenberg ${ROWS}x${COLS} full pipeline ==="
echo "  working dir: $WORKDIR"
echo ""

mkdir -p logs

# Step 1: Generate data (CPU only)
JOB1=$(sbatch --parsable slurm/01_generate_data.sh)
echo "[1/5] Data generation submitted: job $JOB1"

# Step 2: Train transformer (1 GPU)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/02_train_transformer.sh)
echo "[2/5] Training submitted: job $JOB2 (after $JOB1)"

# Steps 3-5 need the results directory, which is only known after training.
echo ""
echo "=== Steps 1-2 submitted ==="
echo ""
echo "After job $JOB2 (training) finishes, find the results dir and submit the rest:"
echo ""
echo "  export RESULTS_DIR=\$(ls -td results/conditional_heisenberg_*/ 2>/dev/null | head -1)"
echo "  echo \$RESULTS_DIR"
echo "  sbatch slurm/03_sample_transformer.sh              # step 3"
echo "  # then after step 3 finishes:"
echo "  sbatch slurm/04_evaluate.sh                        # step 4"
echo "  # then after step 4 finishes:"
echo "  export MODEL_PROPS_DIR=\${RESULTS_DIR}/properties/test/model"
echo "  sbatch slurm/05_plot.sh                            # step 5"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "View logs:    tail -f logs/datagen_${JOB1}.out"
