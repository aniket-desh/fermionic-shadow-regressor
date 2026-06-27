#!/usr/bin/env bash
# Auto-launch the 1B ablation fleet the moment the 1A fleet finishes (keeps GPUs busy).
cd "$(dirname "$0")/../.."          # -> models/
while ! grep -q FLEET_DONE runpod_logs/fleet_1A.log 2>/dev/null; do sleep 60; done
echo "[chain] 1A fleet done -> launching 1B"
bash scripts/revision/fleet_runner.sh runpod_logs/jobs_1B.txt 8 > runpod_logs/fleet_1B.log 2>&1
echo "[chain] CHAIN_DONE_1B"
