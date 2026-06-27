#!/usr/bin/env bash
# Briefing #3 Step 2: dump coherence_grid.npz next to existing n2/beh2 extended heatmaps.
# Re-runs extrapolation_heatmap only (checkpoints + extended datasets already on disk).
set -euo pipefail
cd "$(dirname "$0")/.."            # -> models/
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
R=results/fermionic_pipeline/regression

dump() {  # $1=mol $2=trainR0 $3=trainR1 $4=t_max
  local m=$1 r0=$2 r1=$3 tmax=$4
  local mdir=$R/${m}_regress_v1_orb_s42_model
  echo "[$m extended-heatmap+grid] train R[$r0,$r1] t[0,$tmax]"
  python -m fermionic_pipeline.eval.extrapolation_heatmap \
    --data_path $R/${m}_regress_v1_extrap/regression_targets.h5 \
    --checkpoint $mdir/regressor.pt \
    --save_dir $mdir/plots_extrap \
    --train_r_range "$r0" "$r1" --train_t_range 0 "$tmax" \
    --omega_op_source train-interp \
    --train_data_path $R/${m}_regress_v1/regression_targets.h5 --device cuda
}

dump n2   0.9 2.3 500
dump beh2 1.0 3.0 500
echo "=== STEP2 GRIDS DONE ==="
