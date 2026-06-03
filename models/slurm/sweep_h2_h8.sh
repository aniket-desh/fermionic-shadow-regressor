#!/bin/bash
# Submit fermionic pipeline jobs for all H2–H8 molecules.
# Usage: bash slurm/sweep_h2_h8.sh [TAG] [N_WORKERS]
#   e.g. bash slurm/sweep_h2_h8.sh sweep_v1 8

TAG=${1:-sweep_v1}
N_WORKERS=${2:-8}

for mol in h2 h3 h4 h5 h6 h7 h8; do
    echo "Submitting $mol (tag=$TAG, workers=$N_WORKERS)"
    sbatch --export=ALL,MOLECULE=$mol,TAG=$TAG,N_WORKERS=$N_WORKERS slurm/fermionic_pipeline.sh
done
