#!/bin/bash
# Run the dipole sample-efficiency BO figure on the cluster (CPU-only; reads the
# v13 h5 + dipole_coeffs.npz + v18-orb checkpoint, all on $SCRATCH). Threshold is
# a relative RMS error vs the exact dipole, DFT-level (5%) by default
# (Hait & Head-Gordon, JCTC 2018). Extra args pass through to the python module,
# e.g. `--rel_error 0.06`, `--molecule H2`, `--R 0.65 1.5`.
#
# Needs on the cluster (scp up first if missing): scripts/bo/dipole_bo.py and
# fermionic_pipeline/bo/ (the vendored GP).
#
# Usage (from $SCRATCH/generative-quantum-states):
#   bash slurm/dipole_bo.sh
#   bash slurm/dipole_bo.sh --rel_error 0.06
set -euo pipefail

ACCOUNT="rrg-aspuru"
PARTITION="compute_full_node"
PASS_ARGS="$*"

cat > "slurm/_dipole_bo.sh" << 'EOF'
#!/bin/bash
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2
source "$HOME/envs/gqs/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
cd "$SCRATCH/generative-quantum-states"
export PYTHONUNBUFFERED=1
EOF
echo "python3 -m scripts.bo.dipole_bo ${PASS_ARGS}" >> "slurm/_dipole_bo.sh"
chmod +x "slurm/_dipole_bo.sh"

JOB=$(sbatch --parsable \
  --partition="${PARTITION}" --job-name="dipole_bo" \
  --output="logs/dipole_bo_%j.out" --error="logs/dipole_bo_%j.err" \
  --time=00:40:00 --gpus-per-node=4 --cpus-per-task=8 --account="${ACCOUNT}" \
  "slurm/_dipole_bo.sh")
echo "[submitted] dipole_bo: job ${JOB}"
echo "  args: ${PASS_ARGS:-<defaults: H4, rel_error 0.05>}"
echo "  watch: tail -f logs/dipole_bo_${JOB}.out"
echo "  output: results/fermionic_pipeline/regression/h4_regress_v13/bo/dipole_bo_sweep_h4.pdf"
