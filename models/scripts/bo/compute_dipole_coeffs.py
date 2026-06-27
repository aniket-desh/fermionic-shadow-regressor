"""Compute the dipole observable's decomposition into the FSR's degree-2 Majorana
channels, per geometry, and save it as a small .npz for the (laptop-side) BO.

WHY: the FSR predicts the 120 degree-2 Majorana expectations
``D_mu(R,t) = <Gamma_mu(t)>`` (mu over ``majorana_2pt_keys`` = combinations(16,2)).
The electronic dipole ``mu^a = sum_pq d^a_pq a†_p a_q`` (a in {x,y,z}) is a
one-body operator, hence a linear combination of those same channels:

    <mu^a(R,t)>  =  c0^a(R)  +  sum_mu  c^a_mu(R) * D_mu(R,t).

So once we ship ``c^a_mu(R)``, the time-dependent dipole trace (whose FFT is the
polarizability) is a pyscf-free contraction of the FSR predictions or the exact
shadow targets. ONLY this script needs pyscf/PennyLane; it runs on the cluster
(gqs venv) and emits ``dipole_coeffs.npz``.

CONVENTION SAFETY: the Hamiltonian, the initial state, and the targets are all
built with PennyLane's Jordan-Wigner via the repo helpers. We build mu^a with
the SAME PennyLane JW (``qml.qchem.dipole_moment``) and project onto Gamma_mu
built with the repo's exact ``_majorana_pair_action``. An end-to-end self-test
(``--self_test``) checks <mu^a(0)> reconstructed from c against the direct
expectation on psi0; if any unit/ordering/endianness assumption is off, it fails
loudly BEFORE the full run. RUN ``--self_test`` FIRST.

Usage (cluster, gqs venv):
    # 1) validate conventions on a single geometry (cheap):
    python3 -m scripts.bo.compute_dipole_coeffs --self_test \
        --data_h5 results/fermionic_pipeline/regression/h4_regress_v13/regression_targets.h5
    # 2) full run over the dataset's R grid:
    python3 -m scripts.bo.compute_dipole_coeffs \
        --data_h5 results/fermionic_pipeline/regression/h4_regress_v13/regression_targets.h5 \
        --out results/fermionic_pipeline/regression/h4_regress_v13/dipole_coeffs.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ── Majorana convention, copied VERBATIM from the target generator so c aligns
#    exactly with the trained channels (do not "improve" these). ──────────────

def _gamma_action(indices, majorana_index, n_qubits):
    qubit = majorana_index // 2
    is_y = majorana_index % 2 == 1
    bit_shift = n_qubits - 1 - qubit
    bit = (indices >> bit_shift) & 1
    target = indices ^ (1 << bit_shift)
    if qubit == 0:
        z_phase = np.ones_like(indices, dtype=np.complex128)
    else:
        parity = np.zeros_like(indices, dtype=np.int64)
        for previous in range(qubit):
            parity ^= (indices >> (n_qubits - 1 - previous)) & 1
        z_phase = np.where(parity == 0, 1.0, -1.0).astype(np.complex128)
    if is_y:
        z_phase *= np.where(bit == 0, 1j, -1j)
    return target, z_phase


def _majorana_pair_action(*, n_qubits, p, q):
    """Gamma_pq = -i gamma_p gamma_q (p<q): returns (target_idx, phase) so that
    Gamma|b> = phase[b] |target[b]>. ``target`` is an involution."""
    if p == q:
        raise ValueError("Majorana pair requires distinct indices.")
    if p > q:
        p, q = q, p
    indices = np.arange(1 << n_qubits, dtype=np.int64)
    first_target, first_phase = _gamma_action(indices, q, n_qubits)
    second_target, second_phase = _gamma_action(first_target, p, n_qubits)
    return second_target, (-1j) * first_phase * second_phase


def _majorana_expectation(psi, target_idx, phase):
    return float(np.real_if_close(np.sum(np.conj(psi[target_idx]) * phase * psi), tol=1000).real)


def _trace_gamma_times(matrix, target_idx, phase):
    """Tr(Gamma @ matrix) where Gamma|b> = phase[b]|target[b]>, target involutive.
    (Gamma@M)_bb = sum_a Gamma_ba M_ab = phase[target_b] * M[target_b, b]."""
    return complex(np.sum(phase[target_idx] * matrix[target_idx, np.arange(matrix.shape[0])]))


# ── PennyLane dipole operator (same JW as the Hamiltonian / psi0) ────────────

def _dipole_matrices(molecule, R, n_qubits):
    """Return [mu_x, mu_y, mu_z] as dense (2^n, 2^n) matrices in PennyLane's JW,
    built in the SAME active space as the Hamiltonian (build_molecule_hamiltonian).

    H-chains keep the original full-space path byte-for-byte; named molecules
    (LiH, ...) freeze the core via qml.qchem.active_space with the SAME
    (active_electrons, active_orbitals) partition the Hamiltonian uses, so mu^a
    acts on the same n_qubits as the Gamma_mu channels."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from fermionic_pipeline.data.generate_shadows import (
        _parse_molecule, _MOLECULE_SPECS, _molecule_coordinates,
    )
    import pennylane as qml

    kind, payload = _parse_molecule(molecule)
    if kind == "hchain":  # full active space — original H-chain path, unchanged
        n_atoms = payload
        symbols = ["H"] * n_atoms
        coordinates = np.array([[i * R, 0.0, 0.0] for i in range(n_atoms)], dtype=float)
        mol = qml.qchem.Molecule(symbols, coordinates, charge=0, mult=1, basis_name="sto-3g")
        ops = qml.qchem.dipole_moment(mol, mapping="jordan_wigner")()  # [x, y, z]
    else:                 # named molecule — same core/active partition as the Hamiltonian
        spec = _MOLECULE_SPECS[payload]
        coordinates = _molecule_coordinates(spec, R).reshape(-1, 3)
        mol = qml.qchem.Molecule(spec["symbols"], coordinates, charge=spec["charge"],
                                 mult=spec["mult"], basis_name="sto-3g")
        core, active = qml.qchem.active_space(
            mol.n_electrons, mol.n_orbitals, mult=spec["mult"],
            active_electrons=spec["ae"], active_orbitals=spec["ao"])
        ops = qml.qchem.dipole_moment(mol, core=core, active=active,
                                      mapping="jordan_wigner")()  # [x, y, z]
    mats = []
    for op in ops:
        m = qml.matrix(op, wire_order=range(n_qubits))
        mats.append(np.asarray(m, dtype=np.complex128))
    return mats


def _coeffs_for_geometry(molecule, R, n_qubits, keys):
    """c^a_mu = Tr(Gamma_mu mu^a)/2^n  and  c0^a = Tr(mu^a)/2^n."""
    mats = _dipole_matrices(molecule, R, n_qubits)
    dim = 1 << n_qubits
    c = np.zeros((3, len(keys)), dtype=float)
    c0 = np.zeros(3, dtype=float)
    actions = [_majorana_pair_action(n_qubits=n_qubits, p=k[0], q=k[1]) for k in keys]
    for a, mat in enumerate(mats):
        c0[a] = np.real(np.trace(mat)) / dim
        for j, (tgt, ph) in enumerate(actions):
            val = _trace_gamma_times(mat, tgt, ph) / dim
            assert abs(val.imag) < 1e-8, f"dipole coeff not real ({val}) — convention mismatch"
            c[a, j] = val.real
    return c, c0, mats, actions


def _self_test(molecule, R, n_qubits, keys, h5_expect_t0=None):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from fermionic_pipeline.data.generate_shadows import (
        build_molecule_hamiltonian, prepare_initial_state, molecule_n_electrons,
    )

    H_sparse, nq = build_molecule_hamiltonian(molecule, R)
    assert nq == n_qubits, f"n_qubits mismatch: {nq} != {n_qubits}"
    psi0, _eigvals = prepare_initial_state(
        H_sparse, n_qubits, n_electrons=molecule_n_electrons(molecule))
    psi0 = np.asarray(psi0, dtype=np.complex128).reshape(-1)

    c, c0, mats, actions = _coeffs_for_geometry(molecule, R, n_qubits, keys)
    gamma_exp = np.array([_majorana_expectation(psi0, tgt, ph) for tgt, ph in actions])

    print(f"  R={R:.3f}  n_qubits={n_qubits}  K={len(keys)}")
    ok = True
    for a, name in enumerate("xyz"):
        direct = float(np.real(np.conj(psi0) @ (mats[a] @ psi0)))
        from_c = c0[a] + float(c[a] @ gamma_exp)
        diff = abs(direct - from_c)
        flag = "OK" if diff < 1e-7 else "FAIL"
        ok = ok and diff < 1e-7
        print(f"  <mu_{name}(0)>  direct={direct:+.6e}  from c·<Gamma>={from_c:+.6e}  |Δ|={diff:.1e}  [{flag}]")
    if h5_expect_t0 is not None:
        d = float(np.max(np.abs(gamma_exp - h5_expect_t0)))
        print(f"  <Gamma>(psi0) vs h5 expectations[:,0]  max|Δ|={d:.1e}  "
              f"[{'OK' if d < 1e-6 else 'CHECK (different psi0/seed?)'}]")
    if not ok:
        raise SystemExit("SELF-TEST FAILED: dipole decomposition does not reconstruct <mu(0)>. "
                         "Do NOT trust the coeffs — investigate the JW/geometry/unit convention.")
    print("  SELF-TEST PASSED.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_h5", type=Path, required=True,
                    help="dataset h5 — supplies the R grid (and t=0 targets for the self-test)")
    ap.add_argument("--n_atoms", type=int, default=4,
                    help="linear H-chain length (used when --molecule is not given)")
    ap.add_argument("--molecule", type=str, default=None,
                    help="named active-space molecule (e.g. 'lih'); default uses --n_atoms H-chain")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--self_test", action="store_true",
                    help="validate conventions on the first geometry, then exit")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import h5py
    from fermionic_pipeline.data.regression_dataset import majorana_2pt_keys
    from fermionic_pipeline.data.generate_shadows import _parse_molecule, _MOLECULE_SPECS

    # molecule = named spec if given, else the H-chain length (int)
    molecule = args.molecule if args.molecule is not None else args.n_atoms
    kind, payload = _parse_molecule(molecule)
    n_qubits = 2 * payload if kind == "hchain" else 2 * _MOLECULE_SPECS[payload]["ao"]
    keys = majorana_2pt_keys(2 * n_qubits)  # 2*n_qubits Majorana modes
    assert n_qubits != 8 or len(keys) == 120

    with h5py.File(args.data_h5, "r") as f:
        R_values = np.array(f["R_values"][...], dtype=float)
        expect_t0 = np.array(f["expectations"][:, 0, :], dtype=float) if "expectations" in f else None

    if args.self_test:
        print(f"=== SELF-TEST (single geometry, molecule={molecule}) ===")
        _self_test(molecule, float(R_values[len(R_values) // 2]), n_qubits, keys,
                   h5_expect_t0=expect_t0[len(R_values) // 2] if expect_t0 is not None else None)
        return

    print(f"=== Computing dipole coeffs over {len(R_values)} geometries (molecule={molecule}, n_qubits={n_qubits}, K={len(keys)}) ===")
    c_all = np.zeros((len(R_values), 3, len(keys)), dtype=float)
    c0_all = np.zeros((len(R_values), 3), dtype=float)
    for i, R in enumerate(R_values):
        c, c0, _, _ = _coeffs_for_geometry(molecule, float(R), n_qubits, keys)
        c_all[i], c0_all[i] = c, c0
        if i % 25 == 0:
            print(f"  [{i+1}/{len(R_values)}] R={R:.3f}")

    out = args.out or args.data_h5.with_name("dipole_coeffs.npz")
    np.savez(out,
             R_values=R_values,
             c_x=c_all[:, 0, :], c_y=c_all[:, 1, :], c_z=c_all[:, 2, :],
             c0=c0_all,
             observable_keys=np.array(keys, dtype=int),
             n_qubits=n_qubits)
    print(f"[ok] saved {out}  (c shape {c_all.shape}; ship this small file to the laptop)")


if __name__ == "__main__":
    main()
