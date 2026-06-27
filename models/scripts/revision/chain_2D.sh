#!/usr/bin/env bash
# After the 3A finite-shot RETRAIN fleet: run 2D N_Q sweep, then finalize (curves+figs),
# then drop a sentinel the watcher keys on.
cd "$(dirname "$0")/../.."
while ! grep -q FLEET_DONE runpod_logs/fleet_3Afs_retrain.log 2>/dev/null; do sleep 60; done
echo "[chain] 3A retrain done -> launching 2D N_Q sweep"
bash scripts/revision/run_nqsweep.sh > runpod_logs/run_2Dnq.log 2>&1
echo "[chain] 2D done -> finalize (compute curves + figures)"
python3 scripts/revision/finalize_b10.py > runpod_logs/finalize_b10.log 2>&1
echo "[chain] CHAIN_DONE_2D"
touch runpod_logs/B10_FINALIZE_READY
