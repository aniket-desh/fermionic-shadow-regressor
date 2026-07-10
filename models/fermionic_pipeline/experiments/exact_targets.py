"""Build exact degree-two Majorana targets on an existing HDF5 grid.

This removes both the finite matchgate-library residual and Born-shot noise,
allowing the QA -> exact leg of the cross-library experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from fermionic_pipeline.data.generate_shadows import (
    build_molecule_hamiltonian, molecule_n_electrons, prepare_initial_state, time_evolve,
)
from fermionic_pipeline.data.regression_dataset import RegressionDatasetHandle
from fermionic_pipeline.data.majorana_observables import exact_majorana_channels


def copy_structure(src, dst):
    for key, value in src.attrs.items():
        dst.attrs[key] = value
    for key in src.keys():
        if key != "expectations":
            src.copy(key, dst)


def _geometry_worker(args):
    i, molecule, R, times, keys = args
    H, nq = build_molecule_hamiltonian(molecule, float(R))
    psi0, _ = prepare_initial_state(H, nq, molecule_n_electrons(molecule))
    states = time_evolve(H, psi0, times)
    Psi = np.column_stack([states[t] for t in times])
    return i, exact_majorana_channels(Psi, keys, nq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="HDF5 supplying R/t grid and channel order")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_workers", type=int, default=1)
    args = ap.parse_args()
    h = RegressionDatasetHandle(args.reference)
    with h5py.File(args.reference, "r") as f:
        molecule = f.attrs.get("molecule", f"h{h.n_atoms}")
        if isinstance(molecule, bytes):
            molecule = molecule.decode()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.reference, "r") as src, h5py.File(args.output, "w") as dst:
        copy_structure(src, dst)
        dst.attrs["target_mode"] = "exact-majorana"
        dst.attrs["source_reference"] = str(args.reference)
        exp = dst.create_dataset(
            "expectations", shape=(len(h.R_values), len(h.times), h.n_observables),
            dtype=np.float64, chunks=(1, len(h.times), h.n_observables), compression="lzf",
        )
        jobs = [(i, molecule, float(R), h.times, h.observable_keys)
                for i, R in enumerate(h.R_values)]
        if args.n_workers > 1:
            from multiprocessing import Pool
            with Pool(args.n_workers) as pool:
                for done, (i, values) in enumerate(pool.imap_unordered(_geometry_worker, jobs), 1):
                    exp[i] = values
                    print(f"[exact {done}/{len(jobs)}] R={h.R_values[i]:.4f}", flush=True)
        else:
            for done, job in enumerate(jobs, 1):
                i, values = _geometry_worker(job); exp[i] = values
                print(f"[exact {done}/{len(jobs)}] R={h.R_values[i]:.4f}", flush=True)
        cfg = json.loads(dst.attrs.get("config_json", "{}"))
        cfg["target_mode"] = "exact-majorana"
        dst.attrs["config_json"] = json.dumps(cfg)
    print(f"[done] exact targets -> {args.output}")


if __name__ == "__main__":
    main()
