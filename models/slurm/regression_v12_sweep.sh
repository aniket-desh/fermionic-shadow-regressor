#!/bin/bash
# =======================================================================
# v12 floor sweep: ω_floor ∈ {3, 5, 8} E_h, three parallel jobs.
#
# Each invocation calls regression_v12.sh with a different floor + tag, so
# the three sweeps train into separate model dirs without overwriting:
#   results/fermionic_pipeline/regression/h4_regress_v12_f{3,5,8}_s{42,1729}_model
#
# Pass-through stage flags work too — same composable semantics as
# regression_v12.sh. Default = full pipeline (--data --train --eval).
#
# Usage (on Trillium):
#   cd $SCRATCH/generative-quantum-states
#   bash slurm/regression_v12_sweep.sh                 # full pipeline, all 3 floors
#   bash slurm/regression_v12_sweep.sh --eval          # eval-only on existing v12 checkpoints
#   bash slurm/regression_v12_sweep.sh --train --eval  # skip data prep
# =======================================================================
set -euo pipefail

FLOORS=(3.0 5.0 8.0)

if [ $# -eq 0 ]; then
  STAGES=(--data --train --eval)
else
  STAGES=("$@")
fi

echo "=== v12 floor sweep ==="
echo "  floors: ${FLOORS[*]}"
echo "  stages: ${STAGES[*]}"
echo ""

for F in "${FLOORS[@]}"; do
  TAG="h4_regress_v12_f${F%.*}"   # 3.0 -> f3, 5.0 -> f5, 8.0 -> f8
  echo ""
  echo "============================================================"
  echo "  ω_floor = ${F}    tag = ${TAG}"
  echo "============================================================"
  bash slurm/regression_v12.sh \
    --omega_op_floor "${F}" \
    --tag "${TAG}" \
    "${STAGES[@]}"
done

echo ""
echo "=== Sweep submission complete ==="
echo "Watch with: squeue -u \$USER | grep reg-"
