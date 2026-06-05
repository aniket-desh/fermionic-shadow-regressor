#!/bin/bash
# Dipole-coefficient precompute for the chemist-facing BO (manuscript pillar 3).
#
# WHY: the FSR predicts 120 degree-2 Majorana expectations D_mu(R,t). The dipole
#   <mu^a(R,t)> = c0^a(R) + sum_mu c^a_mu(R) D_mu(R,t) is a one-body (hence linear)
#   functional of those channels; its FFT is the polarizability. This job computes
#   c^a_mu(R) once (needs pyscf/PennyLane, here in the gqs venv) and saves a small
#   dipole_coeffs.npz the laptop-side BO contracts against predictions / h5 targets.
#
# CPU-only (no CUDA needed); light (~minutes for the 251-geometry H4 grid).
#
# Usage (from $SCRATCH/generative-quantum-states):
#   bash slurm/dipole_coeffs.sh --self_test    # validate conventions first (1 geom)
#   bash slurm/dipole_coeffs.sh                # full run over the dataset R grid
#   bash slurm/dipole_coeffs.sh --data_h5 PATH --n_atoms 4
#
# The --self_test job MUST print "SELF-TEST PASSED" before trusting the full run.
set -euo pipefail

ACCOUNT="rrg-aspuru"
PARTITION="compute_full_node"
DATA_H5="results/fermionic_pipeline/regression/h4_regress_v13/regression_targets.h5"
N_ATOMS=4
SELF_TEST=false
EXCLUDE_NODES=""

while [ $# -gt 0 ]; do
  case "$1" in
    --self_test)  SELF_TEST=true ;;
    --data_h5)    DATA_H5="$2"; shift ;;
    --n_atoms)    N_ATOMS="$2"; shift ;;
    --account)    ACCOUNT="$2"; shift ;;
    --partition)  PARTITION="$2"; shift ;;
    --exclude)    EXCLUDE_NODES="$2"; shift ;;
    -h|--help)    grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
    *)            echo "unknown arg: $1"; exit 1 ;;
  esac
  shift
done

if [ "$SELF_TEST" = true ]; then
  TAG="dipole_selftest"; PY_ARGS="--self_test --data_h5 ${DATA_H5} --n_atoms ${N_ATOMS}"; WALLTIME="00:15:00"
else
  TAG="dipole_coeffs";   PY_ARGS="--data_h5 ${DATA_H5} --n_atoms ${N_ATOMS}";            WALLTIME="00:30:00"
fi

# Generated job script — same env prelude as the regression drivers (proven).
cat > "slurm/_${TAG}.sh" << 'EOF'
#!/bin/bash
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2
source "$HOME/envs/gqs/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
cd "$SCRATCH/generative-quantum-states"
export PYTHONUNBUFFERED=1
EOF
cat >> "slurm/_${TAG}.sh" << EOF
python3 -m scripts.bo.compute_dipole_coeffs ${PY_ARGS}
EOF
chmod +x "slurm/_${TAG}.sh"

SBATCH_ARGS=(
  --parsable
  --partition="${PARTITION}"
  --job-name="${TAG}"
  --output="logs/${TAG}_%j.out"
  --error="logs/${TAG}_%j.err"
  --time="${WALLTIME}"
  --gpus-per-node=4
  --cpus-per-task=4
  --account="${ACCOUNT}"
)
[ -n "$EXCLUDE_NODES" ] && SBATCH_ARGS+=(--exclude="${EXCLUDE_NODES}")

echo "=== Dipole coeffs (${TAG}) ==="
echo "  data_h5:   ${DATA_H5}"
echo "  self_test: ${SELF_TEST}   walltime: ${WALLTIME}"
JOB=$(sbatch "${SBATCH_ARGS[@]}" "slurm/_${TAG}.sh")
echo "[submitted] ${TAG}: job ${JOB}"
echo "  watch: tail -f logs/${TAG}_${JOB}.out"
