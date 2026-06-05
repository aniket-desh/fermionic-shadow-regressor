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

def _dipole_matrices(n_atoms, R, n_qubits):
    """Return [mu_x, mu_y, mu_z] as dense (2^n, 2^n) matrices in PennyLane's JW,
    built identically to how build_hydrogen_chain_hamiltonian builds H."""
    import pennylane as qml

    symbols = ["H"] * n_atoms
    coordinates = np.array([[i * R, 0.0, 0.0] for i in range(n_atoms)], dtype=float)
    mol = qml.qchem.Molecule(symbols, coordinates, charge=0, mult=1, basis_name="sto-3g")
    ops = qml.qchem.dipole_moment(mol, mapping="jordan_wigner")()  # [x, y, z]
    mats = []
    for op in ops:
        m = qml.matrix(op, wire_order=range(n_qubits))
        mats.append(np.asarray(m, dtype=np.complex128))
    return mats


def _coeffs_for_geometry(n_atoms, R, n_qubits, keys):
    """c^a_mu = Tr(Gamma_mu mu^a)/2^n  and  c0^a = Tr(mu^a)/2^n."""
    mats = _dipole_matrices(n_atoms, R, n_qubits)
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


def _self_test(n_atoms, R, n_qubits, keys, h5_expect_t0=None):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from fermionic_pipeline.data.generate_shadows import (
        build_hydrogen_chain_hamiltonian, prepare_initial_state,
    )

    H_sparse, nq = build_hydrogen_chain_hamiltonian(n_atoms, R)
    assert nq == n_qubits, f"n_qubits mismatch: {nq} != {n_qubits}"
    psi0 = prepare_initial_state(H_sparse, n_qubits, n_electrons=n_atoms)
    psi0 = np.asarray(psi0, dtype=np.complex128).reshape(-1)

    c, c0, mats, actions = _coeffs_for_geometry(n_atoms, R, n_qubits, keys)
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
    ap.add_argument("--n_atoms", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--self_test", action="store_true",
                    help="validate conventions on the first geometry, then exit")
    args = ap.parse_args()

    import h5py
    from fermionic_pipeline.data.regression_dataset import majorana_2pt_keys

    with h5py.File(args.data_h5, "r") as f:
        R_values = np.array(f["R_values"][...], dtype=float)
        expect_t0 = np.array(f["expectations"][:, 0, :], dtype=float) if "expectations" in f else None

    n_qubits = 2 * args.n_atoms
    keys = majorana_2pt_keys(2 * n_qubits)  # 2*n_qubits Majorana modes
    assert len(keys) == 120 if args.n_atoms == 4 else True

    if args.self_test:
        print("=== SELF-TEST (single geometry) ===")
        _self_test(args.n_atoms, float(R_values[len(R_values) // 2]), n_qubits, keys,
                   h5_expect_t0=expect_t0[len(R_values) // 2] if expect_t0 is not None else None)
        return

    print(f"=== Computing dipole coeffs over {len(R_values)} geometries (n_qubits={n_qubits}, K={len(keys)}) ===")
    c_all = np.zeros((len(R_values), 3, len(keys)), dtype=float)
    c0_all = np.zeros((len(R_values), 3), dtype=float)
    for i, R in enumerate(R_values):
        c, c0, _, _ = _coeffs_for_geometry(args.n_atoms, float(R), n_qubits, keys)
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
