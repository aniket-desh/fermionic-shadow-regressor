#!/usr/bin/env python3
"""R_eq per molecule = argmin_R E_0(R) -- the bond length that minimizes the
active-space ground-state energy. H4 (a symmetric model chain) has no literature
equilibrium, so we define R_eq the same way for ALL molecules: the energy minimum
of the same Hamiltonian the pipeline builds. Used to normalize the cross-molecule
overlay figure's x-axis (R/R_eq).

Needs PySCF/PennyLane (the Hamiltonian builder), so run where datagen runs:
    cd models && python -m scripts.compute_req
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fermionic_pipeline.data.generate_shadows import build_molecule_hamiltonian

# Fine R scans (Angstrom) bracketing each equilibrium.
SCANS = {
    "h4":   (0.60, 1.60, 0.01),
    "lih":  (1.20, 2.20, 0.01),
    "beh2": (1.00, 1.80, 0.01),
    "n2":   (0.90, 1.60, 0.01),
}


def ground_energy(mol, R):
    H_sparse, _ = build_molecule_hamiltonian(mol, R)
    return float(np.linalg.eigvalsh(H_sparse.toarray()).min())


def main():
    out = {}
    for mol, (a, b, step) in SCANS.items():
        Rs = np.round(np.arange(a, b + 1e-9, step), 3)
        E = np.array([ground_energy(mol, float(R)) for R in Rs])
        i = int(np.argmin(E))
        out[mol] = float(Rs[i])
        print(f"{mol:5s}  R_eq = {Rs[i]:.3f} Ang   (E0 = {E[i]:.6f} Ha)")
    print("\nR_EQ =", {k: round(v, 3) for k, v in out.items()})


if __name__ == "__main__":
    main()
