#!/bin/bash
# =======================================================================
# Regression v8: denser R grid + longer time evolution
#
# Changes from v7:
#   - t_max=300 (up from 100), n_times=1500 (dt=0.2 preserved)
#     -> 3x better frequency resolution (Δω ≈ 0.021 Eₕ vs 0.063 Eₕ)
#   - r_dense_cutoff=2.5 (up from 1.5) to fix mid-R dip
#     -> ~211 geometries (200 dense + 11 coarse)
#   - n_q=500 (down from 1000) — same convergence quality as shadow
#     pipeline (σ²=15²/500=0.45 per entry), halves datagen cost
#   - n_workers=16 (up from 8) — Trillium GPU nodes have 96 cores
#   - 300k training steps (up from 200k)
#   - coherence heatmap plot added
#
# Estimated datagen: ~211 geom / 16 workers × ~3000s/geom ≈ 11h
#
# Usage: cd $SCRATCH/generative-quantum-states && bash slurm/regression_v8.sh
# =======================================================================
set -euo pipefail

TAG="${1:-h4_regress_v8}"
DATA_DIR="results/fermionic_pipeline/regression/${TAG}"
DATA_PATH="${DATA_DIR}/regression_targets.h5"
MODEL_DIR="results/fermionic_pipeline/regression/${TAG}_model"

PARTITION="compute_full_node"
ACCOUNT="rrg-aspuru"

echo "=== Regression v8 (dense R + t_max=300) ==="
echo "  tag:     ${TAG}"
echo "  dataset: ${DATA_PATH}"
echo ""

# ── Datagen ──────────────────────────────────────────────────────
cat > "slurm/_datagen_${TAG}.sh" << 'EOF'
#!/bin/bash
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2 scipy-stack/2024a
source "$HOME/envs/gqs/bin/activate"
cd "$SCRATCH/generative-quantum-states"
export PYTHONUNBUFFERED=1
EOF
cat >> "slurm/_datagen_${TAG}.sh" << EOF
python3 -m fermionic_pipeline.data.regression_dataset \\
  --output ${DATA_PATH} \\
  --n_atoms 4 \\
  --r_start 0.5 --r_end 3.0 \\
  --r_step 0.05 \\
  --r_dense_cutoff 2.5 --r_dense_step 0.01 \\
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
  --cpus-per-task=32 \
  --account=${ACCOUNT} \
  "slurm/_datagen_${TAG}.sh")
echo "[submitted] datagen: job ${JOB_DATA}"

# ── Train ────────────────────────────────────────────────────────
cat > "slurm/_train_${TAG}.sh" << 'EOF'
#!/bin/bash
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.2 scipy-stack/2024a
source "$HOME/envs/gqs/bin/activate"
cd "$SCRATCH/generative-quantum-states"
export PYTHONUNBUFFERED=1
EOF
cat >> "slurm/_train_${TAG}.sh" << EOF
python3 -m fermionic_pipeline.training.regressor_trainer \\
  --data_path ${DATA_PATH} \\
  --save_dir ${MODEL_DIR} \\
  --device cuda \\
  --steps 300000 \\
  --batch_size 256 \\
  --lr 1e-3 \\
  --final_lr 1e-7 \\
  --warmup_frac 0.05 \\
  --weight_decay 1e-4 \\
  --d_hidden 512 \\
  --n_layers 4 \\
  --n_fourier 128 \\
  --fourier_scale 15.0 \\
  --conditioned_frequencies \\
  --freq_net_hidden 64 \\
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
module load StdEnv/2023 python/3.11 cuda/12.2 scipy-stack/2024a
source "$HOME/envs/gqs/bin/activate"
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
