#!/usr/bin/env bash
# Recovery (GPU0): beh2 full GPU stage + n2 tail (plots/composition/heatmap).
# beh2 & n2 datasets already exist on disk; n2 train+eval already done.
# Mirrors the gpu() flags in run_molecules_runpod.sh exactly.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
# keep BLAS modest so eval/plot don't fight the concurrent h2o datagen for the ~16-CPU quota
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
R=results/fermionic_pipeline/regression
SEED=42

train_one() {  # $1=mol $2=floor
  local m=$1 fl=$2
  local mdir=$R/${m}_regress_v1_orb_s${SEED}_model
  local data=$R/${m}_regress_v1/regression_targets.h5
  echo "[$m train] floor=$fl"
  python -m fermionic_pipeline.training.regressor_trainer --data_path "$data" --save_dir "$mdir" \
    --device cuda --seed "$SEED" --steps 150000 --batch_size 256 --lr 1e-3 --final_lr 1e-7 \
    --warmup_frac 0.05 --weight_decay 5e-4 --d_hidden 768 --n_layers 6 --n_fourier 256 \
    --fourier_scale 20.0 --conditioned_frequencies --freq_net_hidden 128 --freq_net_layers 3 \
    --adaptive_bandwidth --omega_op_floor "$fl" --soft_omega_floor --explicit_amplitude \
    --amp_rank 16 --grad_clip 1.0 --use_orb_features --standardize_orb_energies \
    --alpha_corr 1.0 --eval_every 2000
}

eval_plots_heatmap() {  # $1=mol $2=trainR0 $3=trainR1 $4=t_max  ; $5=do_eval(1/0)
  local m=$1 r0=$2 r1=$3 tmax=$4 do_eval=$5
  local mdir=$R/${m}_regress_v1_orb_s${SEED}_model
  local data=$R/${m}_regress_v1/regression_targets.h5
  local exdata=$R/${m}_regress_v1_extrap/regression_targets.h5
  local ckpt=$mdir/regressor.pt
  if [ "$do_eval" = "1" ]; then
    echo "[$m eval]"
    python -m fermionic_pipeline.eval.regressor_eval --data_path "$data" --checkpoint "$ckpt" \
      --save_dir "$mdir/eval" --omega_op_source train-interp --device cuda --ljung_box_p 0.06
  fi
  echo "[$m plots]"
  python -m fermionic_pipeline.eval.plot_regression --data_path "$data" --checkpoint "$ckpt" \
    --save_dir "$mdir/plots" --omega_op_source train-interp --device cuda --ljung_box_p 0.06
  echo "[$m composition]"
  python -m fermionic_pipeline.eval.composition_diagnostic --data_path "$data" --checkpoint "$ckpt" \
    --save_dir "$mdir/eval" --omega_op_source train-interp --device cuda
  echo "[$m heatmap]"
  python -m fermionic_pipeline.eval.extrapolation_heatmap --data_path "$exdata" --checkpoint "$ckpt" \
    --save_dir "$mdir/plots_extrap" --train_r_range "$r0" "$r1" --train_t_range 0 "$tmax" \
    --omega_op_source train-interp --train_data_path "$data" --device cuda
}

echo "=== beh2: train + eval + plots + composition + heatmap ==="
train_one beh2 0.6
eval_plots_heatmap beh2 1.0 3.0 500 1

echo "=== n2: tail (eval already done) -> plots + composition + heatmap ==="
eval_plots_heatmap n2 0.9 2.3 500 0

echo "=== GPU0 RECOVERY DONE ==="
