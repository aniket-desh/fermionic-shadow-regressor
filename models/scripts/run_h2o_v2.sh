#!/usr/bin/env bash
# H2O re-run at longer horizon per RUNPOD_H2O_RERUN.md (briefing #2).
# v18-orb recipe VERBATIM; ONLY change vs v1 is the horizon (t_max 800->2500,
# n_times 2001->6251 at same dt=0.4). floor 2.2 unchanged. Tag h2o_regress_v2
# (does NOT clobber v1). N_WORKERS=15 (NOT nproc=96 -> ~16-CPU cgroup quota).
set -euo pipefail
cd "$(dirname "$0")/.."            # -> models/
export CUDA_VISIBLE_DEVICES=0
RESULTS=results/fermionic_pipeline/regression
TAG=h2o_regress_v2
DATA=$RESULTS/$TAG/regression_targets.h5
MDIR=$RESULTS/${TAG}_orb_s42_model
CKPT=$MDIR/regressor.pt
NW=${NW:-15}
mkdir -p "$RESULTS/$TAG"

echo "[v2 datagen] h2o t_max=2500 n_times=6251 dt=0.4 box[0.7,1.5] step0.01 (81 geoms) NW=$NW"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m fermionic_pipeline.data.regression_dataset --output "$DATA" --molecule h2o \
  --r_start 0.7 --r_end 1.5 --r_step 0.01 --t_max 2500 --n_times 6251 --n_q 500 --n_workers "$NW"
python -m fermionic_pipeline.data.compute_omega_op --data_path "$DATA"

echo "[v2 train] v18-orb verbatim, floor 2.2, seed 42"
python -m fermionic_pipeline.training.regressor_trainer --data_path "$DATA" --save_dir "$MDIR" \
  --device cuda --seed 42 --steps 150000 --batch_size 256 --lr 1e-3 --final_lr 1e-7 \
  --warmup_frac 0.05 --weight_decay 5e-4 --d_hidden 768 --n_layers 6 --n_fourier 256 \
  --fourier_scale 20.0 --conditioned_frequencies --freq_net_hidden 128 --freq_net_layers 3 \
  --adaptive_bandwidth --omega_op_floor 2.2 --soft_omega_floor --explicit_amplitude \
  --amp_rank 16 --grad_clip 1.0 --use_orb_features --standardize_orb_energies \
  --alpha_corr 1.0 --eval_every 2000

echo "[v2 eval] regressor_eval + plots(incl in-box coherence heatmap) + composition (non-oracle)"
python -m fermionic_pipeline.eval.regressor_eval --data_path "$DATA" --checkpoint "$CKPT" \
  --save_dir "$MDIR/eval" --omega_op_source train-interp --device cuda --ljung_box_p 0.06
python -m fermionic_pipeline.eval.plot_regression --data_path "$DATA" --checkpoint "$CKPT" \
  --save_dir "$MDIR/plots" --omega_op_source train-interp --device cuda --ljung_box_p 0.06
python -m fermionic_pipeline.eval.composition_diagnostic --data_path "$DATA" --checkpoint "$CKPT" \
  --save_dir "$MDIR/eval" --omega_op_source train-interp --device cuda
echo "=== H2O V2 DONE ==="
