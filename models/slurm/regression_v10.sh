#!/bin/bash
# =======================================================================
# Regression v10: HF orbital features + dense full-range sampling
#
# Two changes targeting remaining failure modes:
#
# 1. Dense sampling across ALL R (dR=0.01 everywhere, ~250 geometries)
#    - Fixes coarse tail (R>2.5) regression from v8/v9
#    - Same mechanism that fixed mid-R dip in v8
#
# 2. HF orbital energies as freq_net input
#    - Replaces scalar R with 4 spatial MO energies ε_i(R)
#    - Makes multi-reference transition at R≈0.7-1.0 explicitly visible
#    - freq_net: g_φ(ε₁,...,ε₄) → frequency shifts (was g_φ(R))
#
# Model: same as v9 (3.5M params, 6×768 trunk, 256 Fourier)
# Dataset: new datagen required (HF energies + dense grid)
#
# Usage: cd $SCRATCH/generative-quantum-states && bash slurm/regression_v10.sh
# =======================================================================
set -euo pipefail

TAG="${1:-h4_regress_v10}"
DATA_DIR="results/fermionic_pipeline/regression/${TAG}"
DATA_PATH="${DATA_DIR}/regression_targets.h5"
MODEL_DIR="results/fermionic_pipeline/regression/${TAG}_model"

PARTITION="compute_full_node"
ACCOUNT="rrg-aspuru"

echo "=== Regression v10 (HF orbital features + dense full-range) ==="
echo "  tag:     ${TAG}"
echo "  dataset: ${DATA_PATH}"
echo "  model:   ${MODEL_DIR}"
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
  --t_max 300.0 --n_times 1500 --n_q 500 \\
  --n_workers 16
EOF
chmod +x "slurm/_datagen_${TAG}.sh"

JOB_DATA=$(sbatch --parsable \
  --partition=${PARTITION} \
  --job-name="reg-data-${TAG}" \
  --output="logs/reg_data_${TAG}_%j.out" \
  --error="logs/reg_data_${TAG}_%j.err" \
  --time=24:00:00 \
  --gpus-per-node=4 \
  --cpus-per-task=16 \
  --account=${ACCOUNT} \
  "slurm/_datagen_${TAG}.sh")
echo "[submitted] datagen: job ${JOB_DATA}"

# ── Train ────────────────────────────────────────────────────────
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
