#!/usr/bin/env bash
# Briefing #4 Step 2: retrain LiH on buffered box [0.8,3.4] so R=3.2 becomes interior.
# v18-orb recipe VERBATIM (floor 0.5); ONLY change vs v1 is the train box (1.0,3.2 -> 0.8,3.4).
# Tag lih_regress_v2 (v1 kept intact). N_WORKERS=15 (NOT nproc=96 -> ~16-CPU cgroup quota).
set -euo pipefail
cd "$(dirname "$0")/.."            # -> models/
export CUDA_VISIBLE_DEVICES=0
RES=results/fermionic_pipeline/regression
TAG=lih_regress_v2
DATA=$RES/$TAG/regression_targets.h5
MDIR=$RES/${TAG}_orb_s42_model
CKPT=$MDIR/regressor.pt
NW=${NW:-15}
mkdir -p "$RES/$TAG"

echo "[lih v2 datagen] box[0.8,3.4] step0.01 t_max1200 n_times4801 NW=$NW"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m fermionic_pipeline.data.regression_dataset --output "$DATA" --molecule lih \
  --r_start 0.8 --r_end 3.4 --r_step 0.01 --t_max 1200 --n_times 4801 --n_q 500 --n_workers "$NW"
python -m fermionic_pipeline.data.compute_omega_op --data_path "$DATA"

echo "[lih v2 train] v18-orb verbatim, floor 0.5, seed 42"
python -m fermionic_pipeline.training.regressor_trainer --data_path "$DATA" --save_dir "$MDIR" \
  --device cuda --seed 42 --steps 150000 --batch_size 256 --lr 1e-3 --final_lr 1e-7 \
  --warmup_frac 0.05 --weight_decay 5e-4 --d_hidden 768 --n_layers 6 --n_fourier 256 \
  --fourier_scale 20.0 --conditioned_frequencies --freq_net_hidden 128 --freq_net_layers 3 \
  --adaptive_bandwidth --omega_op_floor 0.5 --soft_omega_floor --explicit_amplitude \
  --amp_rank 16 --grad_clip 1.0 --use_orb_features --standardize_orb_energies \
  --alpha_corr 1.0 --eval_every 2000

echo "[lih v2 eval] regressor_eval + plot_regression (in-box heatmap + grid), non-oracle"
python -m fermionic_pipeline.eval.regressor_eval --data_path "$DATA" --checkpoint "$CKPT" \
  --save_dir "$MDIR/eval" --omega_op_source train-interp --device cuda --ljung_box_p 0.06
python -m fermionic_pipeline.eval.plot_regression --data_path "$DATA" --checkpoint "$CKPT" \
  --save_dir "$MDIR/plots" --omega_op_source train-interp --device cuda --ljung_box_p 0.06
echo "=== LIH V2 DONE ==="
