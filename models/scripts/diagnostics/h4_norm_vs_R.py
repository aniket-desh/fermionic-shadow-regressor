"""
H4 spectral-norm and Bohr-frequency-bandwidth profile vs R.

Computes ‖H(R)‖ (= max |E_a|) across a user-specified R-range for linear H4 in
STO-3G with active_electrons=active_orbitals=4, then derives the Prop-2
worst-case Nyquist requirement Δt ≤ π / (2‖H‖) along with operational ω_op
fractions (99%, 99.9%) inferred from the eigenvalue gap distribution of the
Hartree-Fock-projected dynamics. Operational ω_op here is approximated as the
quantile of the Bohr-frequency support weighted by HF overlap with each
eigenstate, which is what actually appears in the time-series signal.

Output: JSON table at scripts/diagnostics/h4_norm_vs_R.json with one record per
R, plus a printed summary including dt recommendations.

Usage:
    python3 -m scripts.diagnostics.h4_norm_vs_R \
        --R_min 0.5 --R_max 3.0 --R_step 0.05
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from fermionic_pipeline.data.generate_shadows import (
    build_hydrogen_chain_hamiltonian,
    prepare_initial_state,
)


def hf_state_unexcited(n_qubits: int, n_electrons: int) -> np.ndarray:
    """Plain HF Slater determinant in JW: lowest n_electrons spin-orbitals occupied."""
    psi = np.zeros(2 ** n_qubits, dtype=np.complex128)
    bitstr = (1 << n_electrons) - 1  # |1...10...0>
    psi[bitstr] = 1.0
    return psi


def bohr_frequency_quantile(eigvals: np.ndarray, weights: np.ndarray, q: float) -> float:
    """
    Quantile of the |omega_ab| distribution weighted by |c_a|^2 |c_b|^2 where
    c_a = <a|psi_HF>. This is the spectral-mass-weighted Bohr-frequency support.
    """
    # Outer product of weights -> (D,D) joint
    w = weights[:, None] * weights[None, :]
    om = np.abs(eigvals[:, None] - eigvals[None, :])
    flat_om = om.ravel()
    flat_w = w.ravel()
    # remove diagonal (omega=0 doesn't carry oscillation)
    keep = flat_om > 1e-12
    flat_om = flat_om[keep]
    flat_w = flat_w[keep]
    if flat_w.sum() < 1e-30:
        return 0.0
    order = np.argsort(flat_om)
    flat_om = flat_om[order]
    flat_w = flat_w[order]
    cum = np.cumsum(flat_w) / flat_w.sum()
    idx = int(np.searchsorted(cum, q))
    idx = min(idx, len(flat_om) - 1)
    return float(flat_om[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R_min", type=float, default=0.5)
    ap.add_argument("--R_max", type=float, default=3.0)
    ap.add_argument("--R_step", type=float, default=0.05)
    ap.add_argument("--n_atoms", type=int, default=4)
    ap.add_argument("--out", type=str,
                    default="scripts/diagnostics/h4_norm_vs_R.json")
    ap.add_argument("--margin", type=float, default=2.0,
                    help="Safety factor over Prop-2 bound: dt = π/(2·margin·‖H‖)")
    args = ap.parse_args()

    Rs = np.arange(args.R_min, args.R_max + 1e-9, args.R_step)
    records = []
    print(f"{'R':>5} {'‖H‖':>9} {'E0':>9} {'gap10':>8} {'ω_op99':>8} {'ω_op999':>8} "
          f"{'Δt_max(2x)':>12} {'N_T(T=300)':>11}  t/sec")
    for R in Rs:
        t0 = time.time()
        R = float(round(R, 4))
        H_sparse, n_qubits = build_hydrogen_chain_hamiltonian(args.n_atoms, R)
        H_dense = np.asarray(H_sparse.todense())
        eigvals, eigvecs = np.linalg.eigh(H_dense)

        psi_hf = hf_state_unexcited(n_qubits, n_electrons=args.n_atoms)
        # |c_a|^2 = |<a|HF>|^2
        coeffs = eigvecs.conj().T @ psi_hf
        weights = np.abs(coeffs) ** 2

        H_norm = float(np.max(np.abs(eigvals)))
        spread = float(eigvals.max() - eigvals.min())  # tightest Bohr-freq upper bound
        E0 = float(eigvals[0])
        gap10 = float(eigvals[1] - eigvals[0]) if len(eigvals) > 1 else float("nan")
        omega_op_99 = bohr_frequency_quantile(eigvals, weights, 0.99)
        omega_op_999 = bohr_frequency_quantile(eigvals, weights, 0.999)

        dt_max_worstcase = float(np.pi / (args.margin * spread))
        n_T = int(np.ceil(300.0 / dt_max_worstcase))

        elapsed = time.time() - t0
        records.append({
            "R": R, "n_qubits": n_qubits,
            "H_norm": H_norm, "E_max_minus_E_min": spread,
            "E0": E0, "gap10": gap10,
            "omega_op_99": omega_op_99, "omega_op_999": omega_op_999,
            "dt_max_prop2_with_margin": dt_max_worstcase,
            "N_T_for_T300": n_T,
        })
        print(f"{R:>5.2f} {H_norm:>9.4f} {E0:>9.4f} {gap10:>8.4f} "
              f"{omega_op_99:>8.4f} {omega_op_999:>8.4f} "
              f"{dt_max_worstcase:>12.4f} {n_T:>11d}  {elapsed:.1f}s")

    out = {
        "params": {
            "R_min": args.R_min, "R_max": args.R_max, "R_step": args.R_step,
            "n_atoms": args.n_atoms, "margin": args.margin,
            "basis": "sto-3g", "active_electrons": args.n_atoms,
            "active_orbitals": args.n_atoms,
        },
        "records": records,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {args.out}  ({len(records)} R values)")

    # summary recommendation
    Rmin_rec = min(records, key=lambda r: r["dt_max_prop2_with_margin"])
    print()
    print(f"=== Prop-2 recommendation (margin={args.margin}x) ===")
    print(f"  Tightest constraint at R={Rmin_rec['R']:.2f}:")
    print(f"    spread (E_max-E_min) = {Rmin_rec['E_max_minus_E_min']:.3f} E_h")
    print(f"    Δt ≤ {Rmin_rec['dt_max_prop2_with_margin']:.4f} a.u.")
    print(f"    N_T for T=300 a.u. ≥ {Rmin_rec['N_T_for_T300']}")
    # operational alternative
    omega_op_max = max(r["omega_op_999"] for r in records)
    dt_op = float(np.pi / (args.margin * omega_op_max))
    print(f"  Operational alternative (ω_op_999 worst across R = {omega_op_max:.3f} E_h):")
    print(f"    Δt ≤ {dt_op:.4f} a.u.   (covers 99.9% of HF-weighted Bohr support)")
    print(f"    N_T for T=300 a.u. ≥ {int(np.ceil(300.0 / dt_op))}")


if __name__ == "__main__":
    main()
