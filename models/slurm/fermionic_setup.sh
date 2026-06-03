#!/bin/bash
# One-time setup: install additional deps for the fermionic pipeline.
# Run AFTER setup_env.sh, on an interactive node or login node.

set -e

module load StdEnv/2023 python/3.11 cuda/12.2 scipy-stack/2024a
source "$HOME/envs/gqs/bin/activate"

echo "=== Installing fermionic pipeline dependencies ==="

pip install cirq-core openfermion h5py pyyaml

echo "=== Fermionic setup complete ==="
echo "Test with: python -c 'import cirq; import openfermion; import h5py; print(\"OK\")'"
