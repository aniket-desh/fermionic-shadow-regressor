#!/usr/bin/env bash
# One v18-orb run (+ optional ablation) + non-oracle eval. Env-driven.
#   MOL SAVE FLOOR [SEED=42] [ORB=1] [ADAPT=1] [AMPRANK=16] [DATA=<override>]
set -euo pipefail
cd "$(dirname "$0")/../.."          # -> models/
R=results/fermionic_pipeline/regression
: "${MOL:?}"; : "${SAVE:?}"; : "${FLOOR:?}"
SEED=${SEED:-42}; ORB=${ORB:-1}; ADAPT=${ADAPT:-1}; AMPRANK=${AMPRANK:-16}
DATA=${DATA:-$R/${MOL}_regress_v1/regression_targets.h5}
export OMP_NUM_THREADS=${OMP:-2} OPENBLAS_NUM_THREADS=${OMP:-2} MKL_NUM_THREADS=${OMP:-2}
mkdir -p "$SAVE"
ORBF=""; [ "$ORB" = 1 ] && ORBF="--use_orb_features --standardize_orb_energies"
ADAPTF=""; [ "$ADAPT" = 1 ] && ADAPTF="--adaptive_bandwidth --soft_omega_floor"
echo "[train] MOL=$MOL SEED=$SEED FLOOR=$FLOOR ORB=$ORB ADAPT=$ADAPT AMPRANK=$AMPRANK -> $SAVE"
python -m fermionic_pipeline.training.regressor_trainer --data_path "$DATA" --save_dir "$SAVE" \
  --device cuda --seed "$SEED" --steps 150000 --batch_size 256 --lr 1e-3 --final_lr 1e-7 \
  --warmup_frac 0.05 --weight_decay 5e-4 --d_hidden 768 --n_layers 6 --n_fourier 256 \
  --fourier_scale 20.0 --conditioned_frequencies --freq_net_hidden 128 --freq_net_layers 3 \
  $ADAPTF --omega_op_floor "$FLOOR" --explicit_amplitude --amp_rank "$AMPRANK" \
  --grad_clip 1.0 $ORBF --alpha_corr 1.0 --eval_every 2000
echo "[eval] $SAVE"
python -m fermionic_pipeline.eval.regressor_eval --data_path "$DATA" --checkpoint "$SAVE/regressor.pt" \
  --save_dir "$SAVE/eval" --omega_op_source train-interp --device cuda --ljung_box_p 0.06
echo "DONE_TRAINEVAL $SAVE"
