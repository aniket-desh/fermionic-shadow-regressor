#!/usr/bin/env bash
# Run a job list across 2 GPUs with a concurrency cap. Each job line is a full
# shell command (with its own > log redirect); GPU is assigned round-robin.
set -uo pipefail
JOBFILE=$1; MAX=${2:-8}
mapfile -t JOBS < "$JOBFILE"
total=0; for j in "${JOBS[@]}"; do [ -n "$j" ] && total=$((total+1)); done
echo "[fleet] $total jobs, max $MAX concurrent, 2 GPUs round-robin"
running=0; i=0
for job in "${JOBS[@]}"; do
  [ -z "$job" ] && continue
  gpu=$((i % 2))
  eval "CUDA_VISIBLE_DEVICES=$gpu $job" &
  i=$((i+1)); running=$((running+1))
  if [ "$running" -ge "$MAX" ]; then wait -n; running=$((running-1)); fi
done
wait
echo "FLEET_DONE $JOBFILE"
