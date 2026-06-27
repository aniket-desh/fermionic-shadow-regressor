#!/usr/bin/env bash
# B10 2D N_Q sweep (coarse grid). h4+lih at N_Q in {50,100,250,1000} (the N_Q=500 point reuses
# 3A's exact datasets/models). Datagen + train v18-orb + eval (own held-out). N_WORKERS=15.
set -uo pipefail
cd "$(dirname "$0")/../.."          # -> models/
FS=results/fermionic_pipeline/regression/revision/3A_fs            # reuse N_Q=500 exact here
RR=results/fermionic_pipeline/regression/revision/2D_nq
mkdir -p "$RR"
declare -A FL=([h4]=8.0 [lih]=0.5)
declare -A TM=([h4]=300 [lih]=1200)
declare -A R0=([h4]=0.5 [lih]=1.0)
declare -A R1=([h4]=3.0 [lih]=3.2)
NQS="50 100 250 1000"

echo "=== PHASE 1: datagen (8 coarse datasets at varied N_Q) ==="
for mol in h4 lih; do for nq in $NQS; do
  data=$RR/${mol}_nq${nq}/regression_targets.h5; mkdir -p "$RR/${mol}_nq${nq}"
  [ -f "$data" ] && { echo "[skip] $data"; continue; }
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  echo "[datagen] $mol N_Q=$nq"
  python -m fermionic_pipeline.data.regression_dataset --output "$data" --molecule "$mol" \
    --r_start "${R0[$mol]}" --r_end "${R1[$mol]}" --r_step 0.05 --t_max "${TM[$mol]}" \
    --n_times 1001 --n_q "$nq" --n_workers 15
  python -m fermionic_pipeline.data.compute_omega_op --data_path "$data"
done; done

echo "=== PHASE 2: train + eval (own held-out) ==="
JOBF=runpod_logs/jobs_2Dnq.txt; : > "$JOBF"
for mol in h4 lih; do for nq in $NQS; do
  data=$RR/${mol}_nq${nq}/regression_targets.h5
  echo "MOL=$mol SEED=42 FLOOR=${FL[$mol]} DATA=$data SAVE=$RR/${mol}_nq${nq}_model OMP=2 bash scripts/revision/train_eval.sh > runpod_logs/nq_${mol}_${nq}.log 2>&1" >> "$JOBF"
done; done
bash scripts/revision/fleet_runner.sh "$JOBF" 8
echo "NQSWEEP_ALL_DONE"
