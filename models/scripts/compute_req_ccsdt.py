#!/usr/bin/env python3
"""R_eq per molecule via a CCSD(T)/cc-pVTZ energy scan + parabola fit.

This is the DOCUMENTED level for the paper's R/R_eq normalization (Luis: calculated
equilibria are fine if we state the method). It is a FULL-molecule CCSD(T) calculation
(not the STO-3G active space used for the dynamics, whose minimal-CAS energy gives
unphysical ~2x-too-long bonds). For each molecule we scan the bond length near its
equilibrium, evaluate the CCSD(T) total energy, and fit a parabola to locate the minimum.

H4 is the symmetric equally-spaced linear chain, so its R_eq is the optimal uniform
nearest-neighbour spacing at this level.

Needs pyscf. Run where datagen runs:  cd models && python -m scripts.compute_req_ccsdt
"""
import numpy as np
from pyscf import gto, scf, cc

BASIS = "cc-pvtz"

# scan windows (Angstrom) bracketing each equilibrium; widen if the parabola min
# lands at a boundary (the script warns).
SCANS = {
    "n2":   (1.00, 1.25),
    "lih":  (1.45, 1.80),
    "beh2": (1.22, 1.46),
    "h4":   (0.80, 1.25),
}


def geometry(mol, R):
    if mol == "n2":
        return f"N 0 0 0; N 0 0 {R}"
    if mol == "lih":
        return f"Li 0 0 0; H 0 0 {R}"
    if mol == "beh2":            # linear symmetric, Be at centre
        return f"Be 0 0 0; H 0 0 {R}; H 0 0 {-R}"
    if mol == "h4":              # equally spaced linear chain
        return "; ".join(f"H 0 0 {i * R}" for i in range(4))
    raise ValueError(mol)


def ccsdt_energy(mol, R):
    m = gto.M(atom=geometry(mol, R), basis=BASIS, verbose=0)
    mf = scf.RHF(m).run()
    mycc = cc.CCSD(mf).run()
    return mf.e_tot + mycc.e_corr + mycc.ccsd_t()   # CCSD(T) total


def main():
    out = {}
    for mol, (lo, hi) in SCANS.items():
        Rs = np.linspace(lo, hi, 9)
        Es = np.array([ccsdt_energy(mol, float(R)) for R in Rs])
        a, b, _ = np.polyfit(Rs, Es, 2)
        Rmin = -b / (2 * a)
        flag = "" if lo < Rmin < hi else "  <-- min at scan edge, widen window!"
        out[mol] = round(float(Rmin), 4)
        print(f"{mol:5s}  R_eq = {Rmin:.4f} Ang   ({BASIS} CCSD(T), parabola fit){flag}")
    print(f"\nR_EQ_CALC = {out}    # level: CCSD(T)/{BASIS}")


if __name__ == "__main__":
    main()
