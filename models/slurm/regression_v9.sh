#!/bin/bash
# =======================================================================
# Regression v9: scaled capacity, reuses v8 dataset
#
# Same data as v8 (211 geom, t_max=300, n_times=1500, n_q=500).
# Model changes:
#   - d_hidden: 512 → 768 (wider trunk)
#   - n_layers: 4 → 6 (deeper trunk)
#   - n_fourier: 128 → 256 (more spectral slots)
#   - fourier_scale: 15 → 20 (wider frequency range)
#   - freq_net_hidden: 64 → 128, freq_net_layers: 2 → 3 (deeper freq net)
#   - weight_decay: 1e-4 → 5e-4 (stronger regularization for larger model)
#   - steps: 300k → 150k (v8 overfit by 34k; bigger model + stronger reg)
#   - ~3.5M params (up from 990k)
#
# Hypothesis: if capacity was the v8 bottleneck, short-R and coarse-tail
# should improve. If unchanged, the problem is data density or architecture.
#
# Usage: cd $SCRATCH/generative-quantum-states && bash slurm/regression_v9.sh
# =======================================================================
set -euo pipefail

TAG="${1:-h4_regress_v9}"
DATA_PATH="results/fermionic_pipeline/regression/h4_regress_v8/regression_targets.h5"
MODEL_DIR="results/fermionic_pipeline/regression/${TAG}_model"

PARTITION="compute_full_node"
ACCOUNT="rrg-aspuru"

echo "=== Regression v9 (scaled capacity, v8 data) ==="
echo "  tag:     ${TAG}"
echo "  dataset: ${DATA_PATH} (reused from v8)"
echo "  model:   ${MODEL_DIR}"
echo ""

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
  --alpha_corr 1.0 \\
  --eval_every 2000
EOF
chmod +x "slurm/_train_${TAG}.sh"

JOB_TRAIN=$(sbatch --parsable \
  --partition=${PARTITION} \
  --job-name="reg-train-${TAG}" \
  --output="logs/reg_train_${TAG}_%j.out" \
  --error="logs/reg_train_${TAG}_%j.err" \
  --time=24:00:00 \
  --gpus-per-node=4 \
  --cpus-per-task=4 \
  --account=${ACCOUNT} \
  "slurm/_train_${TAG}.sh")
echo "[submitted] train: job ${JOB_TRAIN}"

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
echo "  train: ${JOB_TRAIN} -> eval+plot: ${JOB_EVAL}"
echo "  (no datagen — reusing v8 dataset)"
