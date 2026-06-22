#!/usr/bin/env bash
# =======================================================================
# RunPod runner for the molecule-generalization batch (N2, BeH2, H2O).
#
# Mirrors slurm/regression_molecule.sh but for a single-pod, no-scheduler
# environment. Each molecule runs the UNCHANGED v18-orb recipe; the only
# per-molecule settings are the sampling grid and the bandwidth floor, both
# frozen by the line-spectrum pre-flight (architecture-log 6/15). All share the
# 8-qubit / 120-channel CAS(.,4) interface.
#
# Resource profile (read this before sizing the pod):
#   - datagen           = CPU, multiprocessing across geometries (the long pole)
#   - train/eval/heatmap = GPU, but the model is only ~12M params (light)
# So this script OVERLAPS each molecule's GPU stage with the NEXT molecule's
# CPU datagen, keeping the (single) GPU busy while the CPUs grind the next set.
#
# Run from the repo's models/ dir:
#   cd models && bash scripts/run_molecules_runpod.sh
# Options (env vars):
#   MOLS="n2 beh2 h2o"   which molecules (default all three)
#   SEED=42              training seed
#   N_WORKERS=<n>        datagen CPU workers (default = nproc)
#   N_Q=500              matchgate library size
# Two-GPU pods: run two subsets in parallel, e.g.
#   CUDA_VISIBLE_DEVICES=0 MOLS="n2 beh2" bash scripts/run_molecules_runpod.sh &
#   CUDA_VISIBLE_DEVICES=1 MOLS="h2o"     bash scripts/run_molecules_runpod.sh &
# =======================================================================
set -euo pipefail

MOLS=${MOLS:-"n2 beh2 h2o"}
SEED=${SEED:-42}
N_Q=${N_Q:-500}
N_WORKERS=${N_WORKERS:-$(nproc)}
RESULTS=results/fermionic_pipeline/regression
LOGS=runpod_logs; mkdir -p "$LOGS"

# ---- preflight: fail fast if the env is wrong (pyscf is the classic gotcha) ----
python - <<'PY' || { echo "ENV CHECK FAILED — fix before running"; exit 1; }
import sys
import torch
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}",
      f"device={torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "")
assert torch.cuda.is_available(), "no CUDA GPU visible"
try:
    import pyscf; print(f"pyscf {pyscf.__version__}")
except Exception:
    sys.exit("FATAL: pyscf missing -> orb features become all-zero and training aborts. `pip install pyscf`")
import pennylane; print(f"pennylane {pennylane.__version__}")
PY
echo "datagen workers: $N_WORKERS   seed: $SEED   N_Q: $N_Q"

# frozen per-molecule grid: train(R0 R1 step t_max n_times floor) + extended(R0 R1 step t_max n_times)
grid() {
  case "$1" in
    n2)   echo "0.9 2.3 0.01 500  1001 1.5   0.7 2.6 0.05 1000 2001" ;;
    beh2) echo "1.0 3.0 0.01 500  1001 0.6   0.8 3.4 0.05 1000 2001" ;;
    h2o)  echo "0.7 1.5 0.01 800  2001 2.2   0.6 2.0 0.05 1600 4001" ;;
    lih)  echo "1.0 3.2 0.01 1200 4801 0.5   0.8 3.6 0.05 2400 4801" ;;
    *)    echo "ERR"; return 1 ;;
  esac
}

datagen() {  # CPU: training dataset + extended dataset + omega_op ceiling for both
  local m=$1; read -r R0 R1 RS TMAX NT FL ER0 ER1 ERS ETMAX ENT <<< "$(grid "$m")"
  local tag=${m}_regress_v1
  local data=$RESULTS/$tag/regression_targets.h5
  local exdata=$RESULTS/${tag}_extrap/regression_targets.h5
  mkdir -p "$RESULTS/$tag" "$RESULTS/${tag}_extrap"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  echo "[$m datagen] train R[$R0,$R1] t_max=$TMAX n_times=$NT ($N_WORKERS workers)"
  python -m fermionic_pipeline.data.regression_dataset --output "$data" --molecule "$m" \
    --r_start "$R0" --r_end "$R1" --r_step "$RS" --t_max "$TMAX" --n_times "$NT" \
    --n_q "$N_Q" --n_workers "$N_WORKERS"
  python -m fermionic_pipeline.data.compute_omega_op --data_path "$data"
  echo "[$m datagen] extended R[$ER0,$ER1] t_max=$ETMAX n_times=$ENT"
  python -m fermionic_pipeline.data.regression_dataset --output "$exdata" --molecule "$m" \
    --r_start "$ER0" --r_end "$ER1" --r_step "$ERS" --t_max "$ETMAX" --n_times "$ENT" \
    --n_q "$N_Q" --n_workers "$N_WORKERS"
  python -m fermionic_pipeline.data.compute_omega_op --data_path "$exdata"
}

gpu() {  # GPU: v18-orb train -> eval (non-oracle) -> in/out-of-box coherence heatmap
  local m=$1; read -r R0 R1 RS TMAX NT FL _rest <<< "$(grid "$m")"
  local tag=${m}_regress_v1
  local data=$RESULTS/$tag/regression_targets.h5
  local exdata=$RESULTS/${tag}_extrap/regression_targets.h5
  local mdir=$RESULTS/${tag}_orb_s${SEED}_model
  local ckpt=$mdir/regressor.pt
  echo "[$m train] v18-orb verbatim, floor=$FL, seed=$SEED"
  python -m fermionic_pipeline.training.regressor_trainer --data_path "$data" --save_dir "$mdir" \
    --device cuda --seed "$SEED" --steps 150000 --batch_size 256 --lr 1e-3 --final_lr 1e-7 \
    --warmup_frac 0.05 --weight_decay 5e-4 --d_hidden 768 --n_layers 6 --n_fourier 256 \
    --fourier_scale 20.0 --conditioned_frequencies --freq_net_hidden 128 --freq_net_layers 3 \
    --adaptive_bandwidth --omega_op_floor "$FL" --soft_omega_floor --explicit_amplitude \
    --amp_rank 16 --grad_clip 1.0 --use_orb_features --standardize_orb_energies \
    --alpha_corr 1.0 --eval_every 2000
  echo "[$m eval] regressor_eval + plots + composition (non-oracle: --omega_op_source train-interp)"
  python -m fermionic_pipeline.eval.regressor_eval --data_path "$data" --checkpoint "$ckpt" \
    --save_dir "$mdir/eval" --omega_op_source train-interp --device cuda --ljung_box_p 0.06
  python -m fermionic_pipeline.eval.plot_regression --data_path "$data" --checkpoint "$ckpt" \
    --save_dir "$mdir/plots" --omega_op_source train-interp --device cuda --ljung_box_p 0.06
  python -m fermionic_pipeline.eval.composition_diagnostic --data_path "$data" --checkpoint "$ckpt" \
    --save_dir "$mdir/eval" --omega_op_source train-interp --device cuda
  echo "[$m heatmap] extended-grid in/out-of-box coherence"
  python -m fermionic_pipeline.eval.extrapolation_heatmap --data_path "$exdata" --checkpoint "$ckpt" \
    --save_dir "$mdir/plots_extrap" --train_r_range "$R0" "$R1" --train_t_range 0 "$TMAX" \
    --omega_op_source train-interp --train_data_path "$data" --device cuda
}

# ---- pipeline: GPU stage of molecule i overlaps the CPU datagen of molecule i+1 ----
read -r -a M <<< "$MOLS"
gpid=""
for m in "${M[@]}"; do
  datagen "$m" 2>&1 | tee "$LOGS/datagen_${m}.log"
  [ -n "$gpid" ] && wait "$gpid"            # don't double-book the GPU
  ( gpu "$m" 2>&1 | tee "$LOGS/gpu_${m}.log" ) &
  gpid=$!
done
[ -n "$gpid" ] && wait "$gpid"

echo ""
echo "=== ALL DONE ==="
for m in "${M[@]}"; do
  j=$RESULTS/${m}_regress_v1_orb_s${SEED}_model/eval/regressor_eval.json
  [ -f "$j" ] && echo "  $m -> $(python -c "import json;print('pearson_mean=%.4f'%json.load(open('$j')).get('pearson_mean',float('nan')))" 2>/dev/null || echo "$j")"
done
echo "Heatmaps: $RESULTS/<mol>_regress_v1_orb_s${SEED}_model/plots_extrap/coherence_heatmap.pdf"
