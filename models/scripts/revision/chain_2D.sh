#!/usr/bin/env bash
# Auto-launch 2D N_Q sweep when 3A finite-shot finishes (keeps the pod working).
cd "$(dirname "$0")/../.."
while ! grep -q FINITESHOT_ALL_DONE runpod_logs/run_3Afs.log 2>/dev/null; do sleep 60; done
echo "[chain] 3A done -> launching 2D N_Q sweep"
bash scripts/revision/run_nqsweep.sh > runpod_logs/run_2Dnq.log 2>&1
echo "[chain] CHAIN_DONE_2D"
