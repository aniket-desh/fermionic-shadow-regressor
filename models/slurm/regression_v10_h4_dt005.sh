#!/bin/bash
# =======================================================================
# H4 v10 with 4× finer time grid (dt=0.05 a.u.) — baseline data-quality fix
#
# Diagnostic D1 on the existing v10 H4 dataset (4/28) showed the training
# targets are aliased for all R<0.74 Å: ω_max(R) reaches 22.96 E_h at R=0.5
# while ω_Nyquist = π/dt = 15.7 E_h with the v10 dt=0.20.
#
# This rerun fixes the data-side aliasing while holding the model architecture
# constant. Only datagen-side change: dt 0.20 → 0.05 (n_times 1500 → 6001),
# giving ω_Ny = 62.83 E_h, comfortable headroom for any R ≥ 0.5 Å.
# All other knobs (R-grid, observables, freq_net config, trunk dimensions,
# training schedule) are byte-identical to v10 to isolate the data fix.
#
# Scope of what this fixes (pre-flight 4/28, v10 H4 eval stratified by zone):
#   R<0.74 (aliased, n=6):       pearson 0.450  ← dt=0.05 should close this
#   R∈[0.74, 1.0) (borderline):  pearson 0.392  ← still failing without aliasing
#   R∈[1.0, 1.5) (non-aliased):  pearson 0.735  ← architecture-side weakness
#   R∈[1.5, 2.0) (non-aliased):  pearson 0.949
#   R∈[2.0, 3.0] (non-aliased):  pearson 0.990
#
# So this rerun is a baseline data-quality fix that v11 (and any v12) sits
# on top of, NOT a replacement for the architecture program. The borderline
# [0.74, 1.0) zone has no aliasing but fails as badly as the aliased zone —
# that failure is architectural and survives any dt refinement. v11's
# adaptive-bandwidth + Δ_ai warm-start program remains motivated.
#
# Storage: HDF5 grows ~4× from 362 MB → ~1.4 GB.
# Compute: datagen ~4× longer per-R (more time samples per ED simulation),
#          training similar wall-clock (more samples per geometry but same
#          total epochs against a larger dataset).
#
# Usage: cd $SCRATCH/generative-quantum-states && bash slurm/regression_v10_h4_dt005.sh
# =======================================================================
set -euo pipefail

TAG="${1:-h4_regress_v10_dt005}"
DATA_DIR="results/fermionic_pipeline/regression/${TAG}"
DATA_PATH="${DATA_DIR}/regression_targets.h5"
MODEL_DIR="results/fermionic_pipeline/regression/${TAG}_model"

PARTITION="compute_full_node"
ACCOUNT="rrg-aspuru"

echo "=== Regression v10 H4 (dt=0.05 datagen rerun) ==="
echo "  tag:     ${TAG}"
echo "  dataset: ${DATA_PATH}"
echo "  model:   ${MODEL_DIR}"
echo "  fix:     dt 0.20 -> 0.05 a.u. (n_times 1500 -> 6001), clears aliasing for R >= 0.5"
echo ""

# ── Datagen ──────────────────────────────────────────────────────
cat > "slurm/_datagen_${TAG}.sh" << 'EOF'
#!/bin/bash
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2
source "$HOME/envs/gqs/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
cd "$SCRATCH/generative-quantum-states"
export PYTHONUNBUFFERED=1
# Pin BLAS threads to 1 per worker — venv NumPy uses OpenBLAS (no scipy-stack
# MKL available due to NumPy 1.x ABI conflict), which otherwise spawns ~96
# threads per forked worker and thrashes the node.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
EOF
cat >> "slurm/_datagen_${TAG}.sh" << EOF
python3 -m fermionic_pipeline.data.regression_dataset \\
  --output ${DATA_PATH} \\
  --n_atoms 4 \\
  --r_start 0.5 --r_end 3.0 \\
  --r_step 0.01 \\
  --t_max 300.0 --n_times 6001 --n_q 500 \\
  --n_workers 16
EOF
chmod +x "slurm/_datagen_${TAG}.sh"

JOB_DATA=$(sbatch --parsable \
  --partition=${PARTITION} \
  --job-name="reg-data-${TAG}" \
  --output="logs/reg_data_${TAG}_%j.out" \
  --error="logs/reg_data_${TAG}_%j.err" \
  --time=48:00:00 \
  --gpus-per-node=4 \
  --cpus-per-task=16 \
  --account=${ACCOUNT} \
  "slurm/_datagen_${TAG}.sh")
echo "[submitted] datagen: job ${JOB_DATA} (48h walltime — 4x time samples per geometry)"

# ── Train ────────────────────────────────────────────────────────
# Identical to v10 train config, byte-for-byte — only the dataset differs.
cat > "slurm/_train_${TAG}.sh" << 'EOF'
#!/bin/bash
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2
source "$HOME/envs/gqs/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
cd "$SCRATCH/generative-quantum-states"
export PYTHONUNBUFFERED=1
EOF
cat >> "slurm/_train_${TAG}.sh" << EOF
python3 -m fermionic_pipeline.training.regressor_trainer \\
  --data_path ${DATA_PATH} \\
  --save_dir ${MODEL_DIR} \\
  --device cuda \\
  --steps 150000 \\
  --batch_size 256 \\
  --lr 1e-3 \\
  --final_lr 1e-7 \\
  --warmup_frac 0.05 \\
  --weight_decay 5e-4 \\
  --d_hidden 768 \\
  --n_layers 6 \\
  --n_fourier 256 \\
  --fourier_scale 20.0 \\
  --conditioned_frequencies \\
  --freq_net_hidden 128 \\
  --freq_net_layers 3 \\
  --use_orb_features \\
  --alpha_corr 1.0 \\
  --eval_every 2000
EOF
chmod +x "slurm/_train_${TAG}.sh"

JOB_TRAIN=$(sbatch --parsable \
  --dependency=afterok:${JOB_DATA} \
  --partition=${PARTITION} \
  --job-name="reg-train-${TAG}" \
  --output="logs/reg_train_${TAG}_%j.out" \
  --error="logs/reg_train_${TAG}_%j.err" \
  --time=24:00:00 \
  --gpus-per-node=4 \
  --cpus-per-task=4 \
  --account=${ACCOUNT} \
  "slurm/_train_${TAG}.sh")
echo "[submitted] train: job ${JOB_TRAIN} (after ${JOB_DATA})"

# ── Eval + Plot ──────────────────────────────────────────────────
cat > "slurm/_eval_${TAG}.sh" << 'EOF'
#!/bin/bash
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2
source "$HOME/envs/gqs/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
cd "$SCRATCH/generative-quantum-states"
export PYTHONUNBUFFERED=1
EOF
cat >> "slurm/_eval_${TAG}.sh" << EOF
python3 -m fermionic_pipeline.eval.regressor_eval \\
  --data_path ${DATA_PATH} \\
  --checkpoint ${MODEL_DIR}/regressor.pt \\
  --save_dir ${MODEL_DIR}/eval \\
  --device cuda --ljung_box_p 0.06
python3 -m fermionic_pipeline.eval.plot_regression \\
  --data_path ${DATA_PATH} \\
  --checkpoint ${MODEL_DIR}/regressor.pt \\
  --save_dir ${MODEL_DIR}/plots \\
  --device cuda --ljung_box_p 0.06
EOF
chmod +x "slurm/_eval_${TAG}.sh"

JOB_EVAL=$(sbatch --parsable \
  --dependency=afterok:${JOB_TRAIN} \
  --partition=${PARTITION} \
  --job-name="reg-eval-${TAG}" \
  --output="logs/reg_eval_${TAG}_%j.out" \
  --error="logs/reg_eval_${TAG}_%j.err" \
  --time=04:00:00 \
  --gpus-per-node=4 \
  --cpus-per-task=4 \
  --account=${ACCOUNT} \
  "slurm/_eval_${TAG}.sh")
echo "[submitted] eval:  job ${JOB_EVAL} (after ${JOB_TRAIN})"

echo ""
echo "=== Submitted ==="
echo "  datagen: ${JOB_DATA} -> train: ${JOB_TRAIN} -> eval+plot: ${JOB_EVAL}"
echo ""
echo "Expected outcome (pre-flight stratification on v10):"
echo "  R<0.74 (aliased):       expect pearson jump 0.45 -> ~0.9 (data fix removes aliasing)"
echo "  R∈[0.74, 1.5) (clean):  expect pearson roughly unchanged (architecture-side weakness)"
echo "  R≥1.5 (already strong): expect no change"
echo ""
echo "If [0.74, 1.5) ALSO improves substantially, that's evidence v10's freq_net was"
echo "being driven by aliasing leakage even at non-aliased R; otherwise the v11"
echo "adaptive-bandwidth + Δ_ai warm-start program is still load-bearing for that zone."
