"""
Standalone data generation script.

Usage:
    python -m fermionic_pipeline.scripts.generate_data --config fermionic_pipeline/configs/h4.yaml --output data/fermionic_shadows/H4_shadows.h5
"""

import os
import sys
import argparse

import yaml
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fermionic_pipeline.data.generate_shadows import generate_chain_data
from fermionic_pipeline.data.dataset import FermionicShadowDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dc = cfg["data"]
    rng = np.random.default_rng(dc["seed"])

    R_values = np.round(
        np.arange(dc["r_start"], dc["r_end"] + dc["r_step"] / 2, dc["r_step"]), 2
    )
    times = np.linspace(0, dc["t_max"], dc["n_times"])

    print(f"Generating H{dc['n_atoms']} fermionic shadow data")
    print(
        f"  {len(R_values)} geometries x {len(times)} times x {dc['n_shadows']} shadows"
    )
    print(f"  Total: {len(R_values) * len(times) * dc['n_shadows']} snapshots")

    outcomes_dict, metadata = generate_chain_data(
        n_atoms=dc["n_atoms"],
        R_values=R_values,
        times=times,
        n_shadows=dc["n_shadows"],
        rng=rng,
    )

    n_qubits = metadata["n_qubits"]

    # Split train/test
    test_idx = np.sort(rng.choice(len(R_values), size=dc["n_test_geom"], replace=False))
    test_R = set(R_values[test_idx].tolist())
    train_R = set(R_values.tolist()) - test_R

    train_dict = {k: v for k, v in outcomes_dict.items() if k[0] in train_R}
    test_dict = {k: v for k, v in outcomes_dict.items() if k[0] in test_R}

    train_ds = FermionicShadowDataset.from_outcomes_dict(train_dict, n_qubits)
    test_ds = FermionicShadowDataset.from_outcomes_dict(test_dict, n_qubits)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    train_ds.save_hdf5(args.output, split="train")
    test_ds.save_hdf5(args.output, split="test")

    print(f"Train: {len(train_ds)} samples, Test: {len(test_ds)} samples")
    print(f"Test R: {sorted(test_R)}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
