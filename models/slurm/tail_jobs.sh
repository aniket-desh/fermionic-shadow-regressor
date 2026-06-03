#!/bin/bash
# Temp monitor — cat .out + .err for every log matching a list of job IDs.
# Defaults to the v12 sweep + composition baselines (479349–479363).
#
# Usage:
#   bash slurm/tail_jobs.sh                       # default set
#   bash slurm/tail_jobs.sh 479349 479352 479360  # explicit subset

if [ $# -eq 0 ]; then
  JOBS=(479349 479350 479351
        479352 479353 479354 479355 479356 479357
        479358 479359 479360 479361 479362 479363)
else
  JOBS=("$@")
fi

for JOB in "${JOBS[@]}"; do
  found=0
  for f in logs/*_${JOB}.out logs/*_${JOB}.err; do
    [ -f "$f" ] || continue
    found=1
    echo ""
    echo "============================================================"
    echo "  $(basename "$f")"
    echo "============================================================"
    cat "$f"
  done
  if [ "$found" -eq 0 ]; then
    echo ""
    echo "[pending] job ${JOB} has no log files yet"
  fi
done
