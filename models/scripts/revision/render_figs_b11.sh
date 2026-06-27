#!/usr/bin/env bash
# B11: re-render the 7 paper PDFs with the label fix. Eval-only, no retrain.
cd "$(dirname "$0")/../.."          # -> models/
RES=results/fermionic_pipeline/regression
declare -A R0=([h4]=0.5 [lih]=1.0 [n2]=0.9 [beh2]=1.0)
declare -A R1=([h4]=3.0 [lih]=3.2 [n2]=2.3 [beh2]=3.0)
declare -A TM=([h4]=300 [lih]=1200 [n2]=500 [beh2]=500)
ck(){ echo "$RES/${1}_regress_v1_orb_s42_model"; }

extrap(){ local m=$1 gpu=$2
  CUDA_VISIBLE_DEVICES=$gpu python -m fermionic_pipeline.eval.extrapolation_heatmap \
    --data_path "$RES/${m}_regress_v1_extrap/regression_targets.h5" \
    --checkpoint "$(ck $m)/regressor.pt" --save_dir "$(ck $m)/plots_extrap" \
    --train_r_range "${R0[$m]}" "${R1[$m]}" --train_t_range 0 "${TM[$m]}" \
    --omega_op_source train-interp \
    --train_data_path "$RES/${m}_regress_v1/regression_targets.h5" --device cuda \
    > "runpod_logs/fig_${m}_extrap.log" 2>&1; echo "[done] extrap $m ($?)"; }

regr(){ local m=$1 gpu=$2
  CUDA_VISIBLE_DEVICES=$gpu python -m fermionic_pipeline.eval.plot_regression \
    --data_path "$RES/${m}_regress_v1/regression_targets.h5" \
    --checkpoint "$(ck $m)/regressor.pt" --save_dir "$(ck $m)/plots" --device cuda \
    > "runpod_logs/fig_${m}_regr.log" 2>&1; echo "[done] regr $m ($?)"; }

# pack across 2 GPUs
extrap h4 0 & extrap lih 1 & extrap beh2 0 & extrap n2 1 &
regr h4 0 & regr lih 1 &
wait
echo "FIG_RENDER_DONE"
