#!/usr/bin/env bash
# Briefing #6: H4 extended coherence grid. NO retrain (checkpoint pulled from HF).
# HF lacked the training .h5 and the ckpt has no train_omega_op, so per user we regen the
# FULL fine training grid (only to get the per-geometry omega_op table for train-interp),
# then the coarse extended eval set, then the heatmap+grid. N_WORKERS=15 (NOT nproc).
set -euo pipefail
cd "$(dirname "$0")/.."            # -> models/
export CUDA_VISIBLE_DEVICES=0
R=results/fermionic_pipeline/regression
TRAIN=$R/h4_regress_v1/regression_targets.h5
EX=$R/h4_regress_v1_extrap/regression_targets.h5
MDIR=$R/h4_regress_v1_orb_s42_model
CKPT=$MDIR/regressor.pt
NW=${NW:-15}
mkdir -p $R/h4_regress_v1 $R/h4_regress_v1_extrap
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "[h4 TRAIN regen] R[0.50,3.00] step0.01 (251) t_max300 n_times6001 NW=$NW  (~5h, fine grid)"
python -m fermionic_pipeline.data.regression_dataset --output "$TRAIN" --molecule h4 \
  --r_start 0.50 --r_end 3.00 --r_step 0.01 --t_max 300 --n_times 6001 --n_q 500 --n_workers "$NW"
python -m fermionic_pipeline.data.compute_omega_op --data_path "$TRAIN"

echo "[h4 EXTENDED datagen] R[0.40,3.30] step0.05 (59) t_max600 n_times1201 NW=$NW  (coarse, fast)"
python -m fermionic_pipeline.data.regression_dataset --output "$EX" --molecule h4 \
  --r_start 0.40 --r_end 3.30 --r_step 0.05 --t_max 600 --n_times 1201 --n_q 500 --n_workers "$NW"
python -m fermionic_pipeline.data.compute_omega_op --data_path "$EX"

echo "[h4 HEATMAP] extrapolation_heatmap train-interp -> coherence_grid.npz + heatmap.pdf"
python -m fermionic_pipeline.eval.extrapolation_heatmap \
  --data_path "$EX" --checkpoint "$CKPT" --save_dir "$MDIR/plots_extrap" \
  --train_r_range 0.50 3.00 --train_t_range 0 300 \
  --omega_op_source train-interp --train_data_path "$TRAIN" --device cuda
echo "=== H4 GRID DONE ==="
