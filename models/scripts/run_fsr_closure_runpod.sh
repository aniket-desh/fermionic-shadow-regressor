#!/usr/bin/env bash
# FSR scientific-closure fleet. Run from models/; see RUNPOD_FSR_CLOSURE.md.
set -euo pipefail

STAGE=${1:-all}
RAW_ROOT=${RAW_ROOT:?Set RAW_ROOT to the HF snapshot containing h4/, lih/, beh2/, n2/, h2o/}
OUT=${OUT:-results/fsr_closure}
N_WORKERS=${N_WORKERS:-$(nproc)}
STEPS=${STEPS:-150000}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda}
mkdir -p "$OUT" "$OUT/logs"

check_env() {
  python -c 'import torch,pyscf,pennylane,h5py,scipy; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
  for mol in h4 lih beh2 n2 h2o; do
    test -f "$RAW_ROOT/$mol/regression_targets.h5" || { echo "missing $RAW_ROOT/$mol/regression_targets.h5"; exit 1; }
    test -f "$RAW_ROOT/$mol/regressor.pt" || { echo "missing $RAW_ROOT/$mol/regressor.pt"; exit 1; }
  done
}

common_flags() {
  printf '%s' "--device $DEVICE --seed $SEED --steps $STEPS --batch_size 256 --lr 1e-3 --final_lr 1e-7 --warmup_frac 0.05 --weight_decay 5e-4 --d_hidden 768 --n_layers 6 --n_fourier 256 --fourier_scale 20.0 --conditioned_frequencies --freq_net_hidden 128 --freq_net_layers 3 --adaptive_bandwidth --soft_omega_floor --explicit_amplitude --amp_rank 16 --grad_clip 1.0 --use_orb_features --standardize_orb_energies --alpha_corr 1.0 --eval_every 2000"
}

floor_for() {
  case "$1" in h4) echo 8.0;; lih) echo 0.5;; beh2) echo 0.6;; n2) echo 1.5;; h2o) echo 2.2;; esac
}

train_one() { # data outdir floor [extra flags]
  local data=$1 dir=$2 floor=$3; shift 3
  if [ -f "$dir/regressor.pt" ]; then echo "[skip] $dir/regressor.pt"; return; fi
  local log=${dir#"$OUT/"}; log=${log//\//_}
  mkdir -p "$dir"
  # shellcheck disable=SC2046
  python -m fermionic_pipeline.training.regressor_trainer \
    --data_path "$data" --save_dir "$dir" --omega_op_floor "$floor" \
    $(common_flags) "$@" 2>&1 | tee "$OUT/logs/${log}.log"
}

metrics() { # data checkpoint output
  local data=$1 ckpt=$2 output=$3
  [ -f "$output" ] && { echo "[skip] $output"; return; }
  python -m fermionic_pipeline.experiments.spectral_metrics \
    --data_path "$data" --checkpoint "$ckpt" --output "$output" --device "$DEVICE"
}

audit() { # data checkpoint output
  local data=$1 ckpt=$2 output=$3
  [ -f "$output" ] && { echo "[skip] $output"; return; }
  python -m fermionic_pipeline.experiments.architecture_audit \
    --data_path "$data" --checkpoint "$ckpt" --output "$output" --device "$DEVICE"
}

stage_splits() {
  python -m fermionic_pipeline.experiments.geometry_splits \
    --data_path "$RAW_ROOT/h4/regression_targets.h5" --output_dir "$OUT/splits/h4" \
    --counts 5 10 20 40 80 160 --spacings 0.05 0.10 0.20 \
    --block_widths 0.10 0.20 0.30 --block_centers 0.85 1.25
  python -m fermionic_pipeline.experiments.geometry_splits \
    --data_path "$RAW_ROOT/n2/regression_targets.h5" --output_dir "$OUT/splits/n2" \
    --counts 5 10 20 40 80 --spacings 0.05 0.10 0.20 \
    --block_widths 0.10 0.20 0.30 --block_centers 2.20
}

stage_geometry() {
  stage_splits
  for mol in h4 n2; do
    local data="$RAW_ROOT/$mol/regression_targets.h5" floor; floor=$(floor_for "$mol")
    for split in "$OUT/splits/$mol"/*.json; do
      [ "$(basename "$split")" = index.json ] && continue
      local tag dir
      tag=$(basename "$split" .json); dir="$OUT/geometry/$mol/$tag"
      train_one "$data" "$dir" "$floor" --train_r_indices_file "$split"
      metrics "$data" "$dir/regressor.pt" "$dir/spectral_metrics.json"
      python -m fermionic_pipeline.experiments.line_interpolation_baseline \
        --data_path "$data" --split "$split" --output "$dir/structured_baseline.json" --n_lines 32
    done
    local nt
    nt=$(python -c "import h5py; f=h5py.File('$data'); print(len(f['times']))")
    python -m fermionic_pipeline.experiments.geometry_summary \
      --geometry_root "$OUT/geometry/$mol" \
      --output_json "$OUT/geometry/$mol/geometry_summary.json" \
      --output_pdf "$OUT/geometry/$mol/geometry_learning_curve.pdf" \
      --n_times "$nt" --n_q 500 --born_samples 1
  done
}

stage_screen() {
  local data="$RAW_ROOT/h4/regression_targets.h5" root="$OUT/screen/h4"
  mkdir -p "$root/released"
  python -m fermionic_pipeline.experiments.train_only_screen --data_path "$data" \
    --checkpoint "$RAW_ROOT/h4/regressor.pt" --output "$root/train_only_screen.json" --safety 1.10
  local train_stride train_horizon
  train_stride=$(python -c "import json; print(json.load(open('$root/train_only_screen.json'))['train_t_stride'])")
  train_horizon=$(python -c "import json; print(json.load(open('$root/train_only_screen.json'))['used_horizon'])")
  train_one "$data" "$root/train_only_grid" 8.0 --omega_op_mode global-max \
    --omega_op_scale 1.10 --train_t_stride "$train_stride" --train_t_max "$train_horizon"
  metrics "$data" "$root/train_only_grid/regressor.pt" "$root/train_only_grid/spectral_metrics.json"
  python -m fermionic_pipeline.experiments.spectral_metrics --data_path "$data" \
    --checkpoint "$RAW_ROOT/h4/regressor.pt" --output "$root/released/train_interp.json" \
    --device "$DEVICE" --omega_op_source train-interp
  python -m fermionic_pipeline.experiments.spectral_metrics --data_path "$data" \
    --checkpoint "$RAW_ROOT/h4/regressor.pt" --output "$root/released/oracle_dataset.json" \
    --device "$DEVICE" --omega_op_source dataset
  local tag extra
  while IFS='|' read -r tag extra; do
    local dir="$root/$tag"
    # shellcheck disable=SC2086
    train_one "$data" "$dir" 8.0 $extra
    metrics "$data" "$dir/regressor.pt" "$dir/spectral_metrics.json"
  done <<EOF
scale_075|--omega_op_scale 0.75
scale_125|--omega_op_scale 1.25
scale_150|--omega_op_scale 1.50
global_train_max|--omega_op_mode global-max
horizon_150|--train_t_max 150
horizon_225|--train_t_max 225
floor_060|--omega_op_floor 6.0
floor_100|--omega_op_floor 10.0
floor_120|--omega_op_floor 12.0
EOF
}

stage_libraries() {
  local qa="$RAW_ROOT/h4/regression_targets.h5" root="$OUT/libraries/h4"
  mkdir -p "$root"
  local libraries=("$qa")
  for seed in 101 202 303; do
    local qb="$root/q_seed${seed}.h5"
    if [ ! -f "$qb" ]; then
      local partial="${qb}.partial"
      python -m fermionic_pipeline.experiments.regenerate_library --reference "$qa" \
        --output "$partial" --seed "$seed" --n_q 500 --n_workers "$N_WORKERS"
      python -m fermionic_pipeline.data.compute_omega_op --data_path "$partial"
      mv "$partial" "$qb"
    fi
    libraries+=("$qb")
  done
  local exact="$root/exact_majorana.h5"
  if [ ! -f "$exact" ]; then
    python -m fermionic_pipeline.experiments.exact_targets \
      --reference "$qa" --output "${exact}.partial" --n_workers "$N_WORKERS"
    mv "${exact}.partial" "$exact"
  fi
  # Released QA checkpoint, then three independently trained libraries.
  local ckpts=("$RAW_ROOT/h4/regressor.pt") names=(qa_released)
  for seed in 101 202 303; do
    local data="$root/q_seed${seed}.h5" dir="$root/train_seed${seed}"
    train_one "$data" "$dir" 8.0
    ckpts+=("$dir/regressor.pt"); names+=("train_seed${seed}")
  done
  for j in "${!ckpts[@]}"; do
    local trainlib=${libraries[$j]} qb=${libraries[$(((j + 1) % ${#libraries[@]}))]}
    python -m fermionic_pipeline.experiments.cross_library_eval \
      --checkpoint "${ckpts[$j]}" --qa "$trainlib" --qb "$qb" --exact "$exact" \
      --output "$root/${names[$j]}_cross_library.json" --device "$DEVICE"
  done
}

stage_gauge() {
  for mol in n2 h2o; do
    local data="$RAW_ROOT/$mol/regression_targets.h5" root="$OUT/gauge/$mol"
    local aligned="$root/aligned_targets.h5" floor; floor=$(floor_for "$mol")
    mkdir -p "$root"
    if [ ! -f "$aligned" ]; then
      python -m fermionic_pipeline.experiments.orbital_gauge_alignment \
        --data_path "$data" --output "${aligned}.partial" --report "$root/gauge_report.json"
      mv "${aligned}.partial" "$aligned"
    fi
    local dir="$root/aligned_model"
    train_one "$aligned" "$dir" "$floor"
    metrics "$aligned" "$dir/regressor.pt" "$dir/spectral_metrics.json"
    metrics "$data" "$RAW_ROOT/$mol/regressor.pt" "$root/raw_spectral_metrics.json"
  done
}

stage_spectra() {
  for mol in h4 lih beh2 n2 h2o; do
    local root="$OUT/spectra/$mol"; mkdir -p "$root"
    metrics "$RAW_ROOT/$mol/regression_targets.h5" "$RAW_ROOT/$mol/regressor.pt" \
      "$root/spectral_metrics.json"
  done
}

stage_architecture() {
  for mol in h4 lih beh2 n2 h2o; do
    local root="$OUT/architecture/$mol"; mkdir -p "$root"
    audit "$RAW_ROOT/$mol/regression_targets.h5" "$RAW_ROOT/$mol/regressor.pt" "$root/audit.json"
  done
  local data="$RAW_ROOT/h4/regression_targets.h5" root="$OUT/architecture/h4_variants"
  # Stability variants: five seeds, all reported (no restart exclusion).
  local variant flags s dir
  while IFS='|' read -r variant flags; do
    for s in 42 1729 7 13 101; do
      dir="$root/${variant}_s${s}"
      # shellcheck disable=SC2086
      train_one "$data" "$dir" 8.0 --seed "$s" $flags
      metrics "$data" "$dir/regressor.pt" "$dir/spectral_metrics.json"
      audit "$data" "$dir/regressor.pt" "$dir/architecture_audit.json"
    done
  done <<EOF
initial_condition|--enforce_initial_condition
ordered_frequencies|--ordered_frequencies
initial_and_ordered|--enforce_initial_condition --ordered_frequencies
EOF
  # Capacity sweeps use the pre-registered seed 42; the released K=256/rank=16
  # five-seed suite is already in the paper cache.
  local tag extra
  while IFS='|' read -r tag extra; do
    local dir="$root/$tag"
    # shellcheck disable=SC2086
    train_one "$data" "$dir" 8.0 $extra
    metrics "$data" "$dir/regressor.pt" "$dir/spectral_metrics.json"
    audit "$data" "$dir/regressor.pt" "$dir/architecture_audit.json"
  done <<EOF
k032|--n_fourier 32
k064|--n_fourier 64
k128|--n_fourier 128
rank04|--amp_rank 4
rank08|--amp_rank 8
rank32|--amp_rank 32
EOF
}

check_env
case "$STAGE" in
  splits) stage_splits;;
  geometry) stage_geometry;;
  screen) stage_screen;;
  libraries) stage_libraries;;
  gauge) stage_gauge;;
  spectra) stage_spectra;;
  architecture) stage_architecture;;
  all)
    stage_geometry; stage_screen; stage_libraries; stage_gauge; stage_spectra; stage_architecture;;
  *) echo "usage: $0 {splits|geometry|screen|libraries|gauge|spectra|architecture|all}"; exit 2;;
esac
echo "[done] FSR closure stage=$STAGE -> $OUT"
