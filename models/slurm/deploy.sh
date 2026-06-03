#!/bin/bash
# Run this LOCALLY to sync code to Trillium and (optionally) kick off the pipeline.
#
# Usage:
#   bash slurm/deploy.sh          # sync only
#   bash slurm/deploy.sh --run    # sync + submit jobs

set -e

REMOTE="aniketrd@trillium-gpu.scinet.utoronto.ca"
# Code goes to $SCRATCH which is writable on compute nodes.
# $SCRATCH on Trillium is /scratch/t/trash/aniketrd (varies by group).
# We use a relative path under the user's scratch.
REMOTE_DIR="\$SCRATCH/generative-quantum-states"

echo "=== Syncing to Trillium (\$SCRATCH) ==="
# First, ensure the directory exists
ssh "$REMOTE" "mkdir -p $REMOTE_DIR"

rsync -avz --exclude '__pycache__' \
           --exclude '*.pyc' \
           --exclude 'quantum-cgm/' \
           --exclude 'data/2d_heisenberg/' \
           --exclude 'results/' \
           --exclude 'logs/' \
           --exclude '.git/' \
           ./ "${REMOTE}:${REMOTE_DIR}/"

echo "=== Sync complete ==="

if [ "$1" = "--run" ]; then
    echo ""
    echo "=== Submitting pipeline ==="
    ssh "$REMOTE" "cd ${REMOTE_DIR} && bash slurm/run_all.sh 4 4"
fi
