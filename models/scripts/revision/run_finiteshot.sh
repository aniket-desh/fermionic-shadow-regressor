#!/usr/bin/env bash
# B10 3A finite-shot study (coarse grid). Phase 1: datagen h4+lih at S in {inf,10,100,1000}
# (inf = exact-marginal reference). Phase 2: train v18-orb on each, EVAL AGAINST THE EXACT
# (S=inf) held-out -> "did it learn the true signal despite shot noise". N_WORKERS=15.
set -uo pipefail
cd "$(dirname "$0")/../.."          # -> models/
RR=results/fermionic_pipeline/regression/revision/3A_fs
mkdir -p "$RR"
declare -A FL=([h4]=8.0 [lih]=0.5)
declare -A TM=([h4]=300 [lih]=1200)
declare -A R0=([h4]=0.5 [lih]=1.0)
declare -A R1=([h4]=3.0 [lih]=3.2)
SVALS="inf 10 100 1000"

gen(){ # $1=mol $2=S
  local mol=$1 S=$2 data=$RR/${1}_S${2}/regression_targets.h5
  mkdir -p "$RR/${mol}_S${S}"
  [ -f "$data" ] && { echo "[skip datagen] $data exists"; return; }
  local sf=""; [ "$S" != inf ] && sf="--shots $S --shot_seed 42"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  echo "[datagen] $mol S=$S"
  python -m fermionic_pipeline.data.regression_dataset --output "$data" --molecule "$mol" \
    --r_start "${R0[$mol]}" --r_end "${R1[$mol]}" --r_step 0.05 --t_max "${TM[$mol]}" \
    --n_times 1001 --n_q 500 --n_workers 15 $sf
  python -m fermionic_pipeline.data.compute_omega_op --data_path "$data"
}

echo "=== PHASE 1: datagen (8 coarse datasets) ==="
for mol in h4 lih; do for S in $SVALS; do gen "$mol" "$S"; done; done

echo "=== PHASE 2: train + eval (each vs exact held-out) ==="
JOBF=runpod_logs/jobs_3Afs.txt; : > "$JOBF"
for mol in h4 lih; do
  exact=$RR/${mol}_Sinf/regression_targets.h5
  for S in $SVALS; do
    data=$RR/${mol}_S${S}/regression_targets.h5
    echo "MOL=$mol SEED=42 FLOOR=${FL[$mol]} DATA=$data EVAL_DATA=$exact SAVE=$RR/${mol}_S${S}_model OMP=2 bash scripts/revision/train_eval.sh > runpod_logs/fs_${mol}_S${S}.log 2>&1" >> "$JOBF"
  done
done
bash scripts/revision/fleet_runner.sh "$JOBF" 8
echo "FINITESHOT_ALL_DONE"
