#!/bin/bash
# One-time setup script for Trillium cluster.
# Run this interactively after SSHing in and after deploy.sh has synced the code.

set -e

echo "=== Setting up generative-quantum-states on Trillium ==="

# Load modules
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load scipy-stack/2024a

# Create virtualenv in $HOME (persists across jobs)
VENV_DIR="$HOME/envs/gqs"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtualenv at $VENV_DIR ..."
    python -m venv --system-site-packages "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Install PyTorch with CUDA support
pip install --no-index torch torchvision

# Install remaining deps
pip install pennylane pennylane-lightning tqdm joblib \
    matplotlib seaborn scikit-learn pandas tensorboard \
    qutip jax neural-tangents

echo "=== Setup complete ==="
echo "Activate with: source $HOME/envs/gqs/bin/activate"
echo "Code lives at: \$SCRATCH/generative-quantum-states/"
