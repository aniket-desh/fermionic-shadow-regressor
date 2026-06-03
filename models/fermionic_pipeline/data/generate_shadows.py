"""
Data generation pipeline for fermionic shadow spectroscopy.

Generates fermionic (matchgate) classical shadow data for linear hydrogen
chains using PennyLane for Hamiltonian construction and statevector
preparation, and the Zhao-Miyake matchgate decomposition for the
shadow measurement protocol.

Pipeline:
  1. Build molecular Hamiltonian H(x) via PennyLane qchem
  2. Compute ground state |psi_0> via exact diagonalization
  3. Prepare initial state with nonzero overlap on first excited state
  4. Time-evolve: |psi(t)> = e^{-iHt} |psi_0>
  5. At each (x, t), collect N_s fermionic shadow snapshots (Q, b)
"""

import os
import sys
import argparse

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

# Vendored matchgate circuit compiler (from Zhao-Miyake submodule)
_VENDOR_DIR = os.path.join(_SCRIPT_DIR, "..", "vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_VENDOR_DIR))


# ── Pure-numpy matchgate gate application ────────────────────────────
# Replaces cirq circuit construction + cirq.unitary() with direct
# statevector manipulation via tensor contractions.

def _apply_1q(psi, gate, q, n):
    """Apply 2x2 gate to qubit q of statevector psi (in-place reshape)."""
    d0 = 1 << q
    d1 = 1 << (n - q - 1)
    psi = psi.reshape(d0, 2, d1)
    psi = np.einsum("ij,ajb->aib", gate, psi)
    return psi.reshape(-1)


def _apply_2q(psi, gate, q, n):
    """Apply 4x4 gate to adjacent qubits (q, q+1) of statevector psi."""
    d0 = 1 << q
    d1 = 1 << (n - q - 2)
    psi = psi.reshape(d0, 4, d1)
    psi = np.einsum("ij,ajb->aib", gate, psi)
    return psi.reshape(-1)


def _zpow(t):
    """ZPowGate(exponent=t): diag(1, e^{i*pi*t})."""
    return np.array([[1, 0], [0, np.exp(1j * np.pi * t)]], dtype=complex)


def _xxpow(t):
    """XXPowGate(exponent=t): exp(i*pi*t/2 * X⊗X)."""
    c = np.cos(np.pi * t / 2)
    s = np.sin(np.pi * t / 2)
    return np.array([
        [c, 0, 0, 1j * s],
        [0, c, 1j * s, 0],
        [0, 1j * s, c, 0],
        [1j * s, 0, 0, c],
    ], dtype=complex)


def _yypow(t):
    """YYPowGate(exponent=t): exp(i*pi*t/2 * Y⊗Y)."""
    c = np.cos(np.pi * t / 2)
    s = np.sin(np.pi * t / 2)
    return np.array([
        [c, 0, 0, -1j * s],
        [0, c, 1j * s, 0],
        [0, 1j * s, c, 0],
        [-1j * s, 0, 0, c],
    ], dtype=complex)


_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _apply_majorana_rotation(psi, p, q, theta, n_qubits):
    """Apply exp(-theta/2 * gamma_p * gamma_q) to statevector.

    Mirrors the logic of majorana_rotation_gate but applies gates
    directly to the statevector instead of building cirq operations.
    """
    if np.isclose(theta, 0.0):
        return psi

    if p > q:
        p, q = q, p
    d = q - p

    if d == 1:
        if p % 2 == 0:
            i = p // 2
            psi = _apply_1q(psi, _zpow(theta / np.pi), i, n_qubits)
        else:
            i = (p - 1) // 2
            psi = _apply_2q(psi, _xxpow(theta / np.pi), i, n_qubits)

    elif d == 2:
        if p % 2 == 0:
            i = p // 2
            psi = _apply_1q(psi, _zpow(-0.5), i, n_qubits)
            psi = _apply_2q(psi, _xxpow(-theta / np.pi), i, n_qubits)
            psi = _apply_1q(psi, _zpow(0.5), i, n_qubits)
        else:
            i = (p - 1) // 2
            psi = _apply_1q(psi, _zpow(-0.5), i + 1, n_qubits)
            psi = _apply_2q(psi, _xxpow(theta / np.pi), i, n_qubits)
            psi = _apply_1q(psi, _zpow(0.5), i + 1, n_qubits)

    elif d == 3 and p % 2 == 0:
        i = p // 2
        psi = _apply_2q(psi, _yypow(-theta / np.pi), i, n_qubits)

    else:
        raise ValueError(f"Majorana indices ({p},{q}) do not map onto 2-local gates.")

    return psi


def _apply_sign_gates(psi, signs, n_qubits):
    """Apply the diagonal sign matrix to the statevector.

    Mirrors majorana_sign_gates but applies directly to statevector.
    """
    signs = np.sign(signs)
    N = len(signs)
    n_modes = N // 2

    pauli_tableau = np.zeros(N, dtype=int)
    for p in range(n_modes):
        if signs[2 * p] == signs[2 * p + 1]:
            if signs[2 * p] == -1.0:
                pauli_tableau[p + n_modes] += 1
        else:
            pauli_tableau[p] += 1
            if signs[2 * p] == 1.0:
                for q in range(p + 1, n_modes):
                    pauli_tableau[q + n_modes] += 1
            else:
                for q in range(p, n_modes):
                    pauli_tableau[q + n_modes] += 1

    pauli_tableau %= 2

    for p in range(n_modes):
        pauli = (pauli_tableau[p], pauli_tableau[p + n_modes])
        if pauli == (0, 1):
            psi = _apply_1q(psi, _Z, p, n_qubits)
        elif pauli == (1, 0):
            psi = _apply_1q(psi, _X, p, n_qubits)
        elif pauli == (1, 1):
            psi = _apply_1q(psi, _Y, p, n_qubits)

    return psi


def _apply_standard_majorana_givens(psi, orth_mat, index_offset, n_qubits):
    """Apply a standard Givens decomposition of orth_mat to the statevector."""
    from optimal_matchgate_circuit import standard_givens_decomposition

    rotations, signs = standard_givens_decomposition(orth_mat)

    full_signs = np.ones(2 * n_qubits)
    full_signs[index_offset:index_offset + len(signs)] = signs
    psi = _apply_sign_gates(psi, full_signs, n_qubits)

    for r, s, theta in rotations:
        psi = _apply_majorana_rotation(
            psi, r + index_offset, s + index_offset, theta, n_qubits
        )

    return psi


def _apply_matchgate(Q, psi, n_qubits):
    """Apply matchgate unitary defined by orthogonal Q to statevector.

    Pure numpy — no cirq circuit construction or simulation.
    Uses the Zhao-Miyake block decomposition (arXiv:2310.03071).
    """
    from optimal_matchgate_circuit import majorana_block_decomposition

    psi = psi.copy().astype(complex)

    left_rotations, right_rotations, signs = majorana_block_decomposition(Q)

    # Right rotations (4x4 blocks)
    for p, _, orth_mat in right_rotations:
        psi = _apply_standard_majorana_givens(psi, orth_mat, p, n_qubits)

    # Diagonal sign matrix
    psi = _apply_sign_gates(psi, signs, n_qubits)

    # Single Givens rotation from left_rotations[0]
    p, _, theta = left_rotations[0]
    psi = _apply_1q(psi, _zpow(theta / np.pi), p // 2, n_qubits)

    # Remaining left rotations (4x4 blocks)
    for p, _, orth_mat in left_rotations[1:]:
        psi = _apply_standard_majorana_givens(psi, orth_mat, p, n_qubits)

    return psi


def sample_fermionic_shadows_statevector(statevector, n_qubits, n_shadows, rng=None):
    """Sample fermionic shadow snapshots from a statevector.

    Uses pure numpy matchgate application (no cirq overhead).

    Args:
        statevector: (2^n,) complex numpy array
        n_qubits: number of qubits
        n_shadows: number of shadow snapshots
        rng: numpy random Generator

    Returns:
        list of [permutation, bitstring] pairs
            permutation: list of ints (length 2n)
            bitstring: list of ints (length n)
    """
    if rng is None:
        rng = np.random.default_rng()

    N = 2 * n_qubits
    outcomes = []

    for _ in range(n_shadows):
        # TODO: sample from full B(2n) (signed permutations) instead of S_{2n}.
        # Currently sampling plain permutations; the manuscript §6.1 specifies
        # G = B(2n) = S_{2n} ⋉ {±1}^{2n}. Requires updating the shadow
        # estimator to handle the additional sign factors.
        perm = rng.permutation(N).tolist()
        Q = np.eye(N)[:, perm]

        # Apply matchgate unitary directly to statevector (no cirq)
        psi_rotated = _apply_matchgate(Q, statevector, n_qubits)

        # Born sampling
        probs = np.abs(psi_rotated) ** 2
        probs = probs / probs.sum()

        idx = rng.choice(len(probs), p=probs)
        bitstring = [(idx >> (n_qubits - 1 - i)) & 1 for i in range(n_qubits)]

        outcomes.append([perm, bitstring])

    return outcomes


def build_hydrogen_chain_hamiltonian(n_atoms, R, return_pennylane=False):
    """Build molecular Hamiltonian for linear H chain at bond length R.

    Args:
        n_atoms: number of hydrogen atoms
        R: bond length in Angstroms
        return_pennylane: if True, also return the PennyLane Hamiltonian
            object (needed for Trotter decomposition)

    Returns:
        H_sparse: sparse Hamiltonian matrix
        n_qubits: number of qubits
        H_pl: (only if return_pennylane=True) PennyLane Hamiltonian
    """
    import pennylane as qml

    symbols = ["H"] * n_atoms
    coordinates = np.array([[i * R, 0.0, 0.0] for i in range(n_atoms)]).flatten()

    H, n_qubits = qml.qchem.molecular_hamiltonian(
        symbols,
        coordinates,
        charge=0,
        mult=1,
        basis="sto-3g",
        active_electrons=n_atoms,
        active_orbitals=n_atoms,
    )

    H_sparse = H.sparse_matrix()
    if return_pennylane:
        return H_sparse, n_qubits, H
    return H_sparse, n_qubits


def prepare_initial_state(H_sparse, n_qubits, n_electrons=None):
    """Prepare initial state with symmetry-breaking excitations.

    Creates a superposition of the Hartree-Fock state with single
    excitations that break particle-number and spin symmetry, giving
    nonzero overlap with eigenstates in multiple symmetry sectors.

    Without excitations, |HF> = |1...10...0> conserves particle number
    and S_z, restricting spectral content to a single symmetry sector.
    Adding HOMO->LUMO and cross-spin excitations accesses many more
    eigenstates, producing rich multi-frequency dynamics.

    Args:
        H_sparse: sparse Hamiltonian matrix
        n_qubits: number of qubits (= number of spin-orbitals)
        n_electrons: number of electrons. If None, falls back to
            ground + first-excited eigenstate superposition.

    Returns:
        psi_0: initial statevector
        eigvals: array of energy eigenvalues (for exact gap comparison)
    """
    dim = 2**n_qubits
    n_eigs = min(dim, 10)
    if dim <= 64:
        eigvals = np.linalg.eigvalsh(H_sparse.toarray())
    else:
        eigvals, _ = eigsh(H_sparse.tocsc(), k=n_eigs, which="SA")
    eigvals = np.sort(eigvals)

    if n_electrons is not None:
        # HF state: occupy the first n_electrons spin-orbitals
        hf_idx = sum(1 << (n_qubits - 1 - i) for i in range(n_electrons))
        psi_0 = np.zeros(dim, dtype=complex)
        psi_0[hf_idx] = 1.0

        # Add single excitations to break symmetry.
        # Each excitation flips one occupied bit off and one virtual bit on,
        # accessing different (particle-number, S_z) sectors.
        n_virt = n_qubits - n_electrons
        excitations = []
        for occ in range(n_electrons):
            for virt in range(n_electrons, n_qubits):
                exc_idx = hf_idx ^ (1 << (n_qubits - 1 - occ)) ^ (1 << (n_qubits - 1 - virt))
                excitations.append(exc_idx)
                if len(excitations) >= 4:
                    break
            if len(excitations) >= 4:
                break

        # Superposition: sqrt(0.8)|HF> + sqrt(0.2/n_exc) sum_i |exc_i>
        n_exc = len(excitations)
        if n_exc > 0:
            amp_exc = np.sqrt(0.2 / n_exc)
            psi_0[hf_idx] = np.sqrt(0.8)
            for exc_idx in excitations:
                psi_0[exc_idx] = amp_exc
        psi_0 /= np.linalg.norm(psi_0)
    else:
        _, eigvecs = eigsh(H_sparse.tocsc(), k=2, which="SA")
        gs = eigvecs[:, 0].real
        gs = gs / np.linalg.norm(gs)
        es = eigvecs[:, 1].real
        es = es / np.linalg.norm(es)
        theta = np.arcsin(np.sqrt(0.1))
        psi_0 = np.cos(theta) * gs + np.sin(theta) * es
        psi_0 = psi_0 / np.linalg.norm(psi_0)

    return psi_0, eigvals


def time_evolve(H_sparse, psi_0, times):
    """Compute |psi(t)> = e^{-iHt} |psi_0> for each t.

    Diagonalizes H once, then computes all time points analytically:
        H = V @ diag(E) @ V†
        |psi(t)> = V @ diag(e^{-iEt}) @ (V† @ |psi_0>)

    This is O(n^3) once for diag + O(n^2) per time point, vs O(n^3) per
    time point with repeated expm calls.

    Returns:
        dict mapping t -> statevector
    """
    H_dense = H_sparse.toarray()

    # Diagonalize once: H = V diag(E) V†
    E, V = np.linalg.eigh(H_dense)
    # Project initial state into eigenbasis
    c = V.conj().T @ psi_0  # expansion coefficients

    states = {}
    for t in times:
        if t == 0.0:
            states[t] = psi_0.copy()
        else:
            # e^{-iHt}|psi_0> = V @ diag(e^{-iEt}) @ c
            phases = np.exp(-1j * E * t)
            states[t] = V @ (phases * c)
    return states


def compute_hf_orbital_energies(n_atoms, R, active_orbitals=None):
    """Compute Hartree-Fock orbital energies for a hydrogen chain.

    Uses PySCF if available (exact RHF orbital energies), otherwise falls
    back to diagonalizing the one-electron core Hamiltonian from PennyLane
    (approximate but avoids the PySCF dependency).

    Args:
        n_atoms: number of hydrogen atoms
        R: bond length in Angstroms
        active_orbitals: number of active orbitals (default: n_atoms)

    Returns:
        mo_energies: (n_active,) array of spatial MO energies
    """
    if active_orbitals is None:
        active_orbitals = n_atoms

    from pyscf import gto, scf
    atom_str = "; ".join(f"H {i * R} 0.0 0.0" for i in range(n_atoms))
    mol = gto.M(atom=atom_str, basis="sto-3g", charge=0, spin=0, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    # Active space: select orbitals around Fermi level
    n_occ = n_atoms // 2
    start = max(0, n_occ - active_orbitals // 2)
    end = start + active_orbitals
    return mf.mo_energy[start:end].copy()


def decompose_hamiltonian_pauli(H_pennylane, n_qubits):
    """Extract Pauli term sparse matrices and coefficients from a PennyLane Hamiltonian.

    Returns:
        coeffs: list of float coefficients
        pauli_mats: list of sparse CSC matrices (one per Pauli string)

    The Identity term is separated out — it contributes only a global phase
    and is applied analytically in the Trotter stepper.
    """
    import pennylane as qml

    raw_coeffs, ops = H_pennylane.terms()
    wire_order = list(range(n_qubits))

    coeffs = []
    pauli_mats = []
    identity_coeff = 0.0

    for c, op in zip(raw_coeffs, ops):
        c_val = float(c.real) if hasattr(c, 'real') else float(c)
        if isinstance(op, qml.Identity):
            identity_coeff += c_val
            continue
        mat = qml.matrix(op, wire_order=wire_order)
        pauli_mats.append(sp.csc_matrix(mat))
        coeffs.append(c_val)

    return coeffs, pauli_mats, identity_coeff


def _apply_pauli_exp(psi, coeff_dt, pauli_mat):
    """Apply exp(-i * coeff_dt * P) to statevector psi.

    Uses P^2 = I for Pauli strings:
        exp(-i * alpha * P) = cos(alpha) * I - i * sin(alpha) * P
    """
    c = np.cos(coeff_dt)
    s = np.sin(coeff_dt)
    return c * psi - 1j * s * (pauli_mat @ psi)


def trotter_step(psi, coeffs, pauli_mats, identity_coeff, dt, order=2):
    """Apply one Trotter step of size dt.

    order=1: first-order (Lie-Trotter)
        exp(-iHdt) ≈ prod_j exp(-i h_j dt)

    order=2: second-order (Suzuki-Trotter S2)
        exp(-iHdt) ≈ prod_j exp(-i h_j dt/2) * prod_j_rev exp(-i h_j dt/2)
    """
    # Global phase from identity term
    psi = psi * np.exp(-1j * identity_coeff * dt)

    if order == 1:
        for c, P in zip(coeffs, pauli_mats):
            psi = _apply_pauli_exp(psi, c * dt, P)
    elif order == 2:
        half_dt = dt / 2.0
        # Forward sweep with dt/2
        for c, P in zip(coeffs, pauli_mats):
            psi = _apply_pauli_exp(psi, c * half_dt, P)
        # Reverse sweep with dt/2
        for c, P in zip(reversed(coeffs), reversed(pauli_mats)):
            psi = _apply_pauli_exp(psi, c * half_dt, P)
    else:
        raise ValueError(f"Trotter order must be 1 or 2, got {order}")

    return psi


def _default_trotter_dt(coeffs):
    """Heuristic Trotter step size: dt = 0.5 / sum(|coeffs|).

    The 1-norm sum(|alpha_j|) upper-bounds the spectral norm. Using
    c=0.5 keeps per-step error conservative for second-order Trotter.
    """
    norm_1 = sum(abs(c) for c in coeffs)
    return 0.5 / norm_1


def time_evolve_trotter(H_pennylane, n_qubits, psi_0, times,
                        trotter_dt=None, trotter_order=2):
    """Trotterized time evolution: evolve |psi_0> to each time in times.

    Evolves sequentially from t=0, saving snapshots at each target time.
    This avoids restarting from |psi_0> for each t_k.

    Args:
        H_pennylane: PennyLane Hamiltonian (Sum of Pauli terms)
        n_qubits: number of qubits
        psi_0: initial statevector
        times: sorted array of target times
        trotter_dt: step size. If None, uses heuristic 0.5/||H||_1.
        trotter_order: 1 or 2 (default 2)

    Returns:
        dict mapping t -> statevector (same interface as time_evolve)
    """
    coeffs, pauli_mats, identity_coeff = decompose_hamiltonian_pauli(
        H_pennylane, n_qubits
    )

    if trotter_dt is None:
        trotter_dt = _default_trotter_dt(coeffs)

    sorted_times = np.sort(times)
    states = {}

    psi = psi_0.copy().astype(complex)
    current_t = 0.0

    for t in sorted_times:
        if t == 0.0:
            states[t] = psi.copy()
            continue

        # Evolve from current_t to t
        remaining = t - current_t
        n_steps = max(1, int(np.ceil(remaining / trotter_dt)))
        dt_actual = remaining / n_steps

        for _ in range(n_steps):
            psi = trotter_step(psi, coeffs, pauli_mats, identity_coeff,
                               dt_actual, order=trotter_order)

        # Renormalize to prevent drift
        psi /= np.linalg.norm(psi)
        states[t] = psi.copy()
        current_t = t

    return states


def validate_trotter(n_atoms, R, times, trotter_dt=None, trotter_order=2):
    """Compare exact vs Trotterized evolution at a single geometry.

    Prints per-time-point fidelity, max statevector error, and — if used
    with the regression pipeline — signal matrix entry errors.

    Returns:
        dict with fidelity, max_error arrays and summary statistics
    """
    import pennylane as qml

    # Build Hamiltonian — need both sparse (for exact) and PennyLane (for Trotter)
    symbols = ["H"] * n_atoms
    coordinates = np.array([[i * R, 0.0, 0.0] for i in range(n_atoms)]).flatten()
    H_pl, n_qubits = qml.qchem.molecular_hamiltonian(
        symbols, coordinates, charge=0, mult=1, basis="sto-3g",
        active_electrons=n_atoms, active_orbitals=n_atoms,
    )
    H_sparse = H_pl.sparse_matrix()

    psi_0, eigvals = prepare_initial_state(H_sparse, n_qubits, n_electrons=n_atoms)

    # Exact evolution
    states_exact = time_evolve(H_sparse, psi_0, times)

    # Trotter evolution
    states_trotter = time_evolve_trotter(
        H_pl, n_qubits, psi_0, times,
        trotter_dt=trotter_dt, trotter_order=trotter_order,
    )

    coeffs, _, _ = decompose_hamiltonian_pauli(H_pl, n_qubits)
    dt_used = trotter_dt if trotter_dt is not None else _default_trotter_dt(coeffs)

    fidelities = []
    max_errors = []

    print(f"[validate] H{n_atoms} R={R:.2f} | dt={dt_used:.4f} | order={trotter_order}")
    print(f"[validate] {'t':>8s}  {'fidelity':>12s}  {'max|err|':>12s}  {'n_steps':>8s}")

    for t in np.sort(times):
        psi_ex = states_exact[t]
        psi_tr = states_trotter[t]

        fid = abs(np.vdot(psi_ex, psi_tr)) ** 2
        max_err = np.max(np.abs(psi_ex - psi_tr))

        fidelities.append(fid)
        max_errors.append(max_err)

        n_steps = max(1, int(np.ceil(t / dt_used))) if t > 0 else 0
        print(f"  {t:8.2f}  {fid:12.8f}  {max_err:12.2e}  {n_steps:8d}")

    fidelities = np.array(fidelities)
    max_errors = np.array(max_errors)

    print(f"[validate] min fidelity: {fidelities.min():.8f} (at t={times[np.argmin(fidelities)]:.2f})")
    print(f"[validate] max |error|:  {max_errors.max():.2e}")
    print(f"[validate] mean fidelity: {fidelities.mean():.8f}")

    return {
        "fidelities": fidelities,
        "max_errors": max_errors,
        "trotter_dt": dt_used,
        "trotter_order": trotter_order,
        "eigvals": eigvals,
    }


def _process_geometry(args):
    """Process a single geometry — designed for multiprocessing.Pool.

    Args:
        args: tuple of (i_R, R, n_atoms, times, n_shadows, seed,
              use_trotter, trotter_dt, trotter_order)
            Last 3 are optional for backward compat.

    Returns:
        (R, outcomes_for_R, n_qubits, eigvals, elapsed)
    """
    import time as _time

    # Unpack — support both old 6-tuple and new 9-tuple
    if len(args) == 6:
        i_R, R, n_atoms, times, n_shadows, seed = args
        use_trotter, trotter_dt, trotter_order = False, None, 2
    else:
        (i_R, R, n_atoms, times, n_shadows, seed,
         use_trotter, trotter_dt, trotter_order) = args

    rng = np.random.default_rng(seed)
    geom_start = _time.time()

    if use_trotter:
        H_sparse, n_qubits, H_pl = build_hydrogen_chain_hamiltonian(
            n_atoms, R, return_pennylane=True
        )
        psi_0, eigvals = prepare_initial_state(H_sparse, n_qubits, n_electrons=n_atoms)
        states = time_evolve_trotter(
            H_pl, n_qubits, psi_0, times,
            trotter_dt=trotter_dt, trotter_order=trotter_order,
        )
    else:
        H_sparse, n_qubits = build_hydrogen_chain_hamiltonian(n_atoms, R)
        psi_0, eigvals = prepare_initial_state(H_sparse, n_qubits, n_electrons=n_atoms)
        states = time_evolve(H_sparse, psi_0, times)

    outcomes_for_R = {}
    for t in times:
        snapshots = sample_fermionic_shadows_statevector(
            states[t], n_qubits, n_shadows, rng=rng
        )
        outcomes_for_R[t] = snapshots

    elapsed = _time.time() - geom_start
    return (R, outcomes_for_R, n_qubits, eigvals, elapsed)


def generate_chain_data(
    n_atoms,
    R_values,
    times,
    n_shadows,
    rng=None,
    n_workers=None,
    use_trotter=False,
    trotter_dt=None,
    trotter_order=2,
):
    """Generate fermionic shadow data for a hydrogen chain.

    Args:
        n_atoms: number of hydrogen atoms
        R_values: array of bond lengths
        times: array of time values
        n_shadows: shadows per (R, t) pair
        rng: numpy random Generator
        n_workers: number of parallel workers for geometry loop.
            None = serial (safe default), >1 = multiprocessing.
        use_trotter: if True, use Trotterized time evolution
        trotter_dt: Trotter step size. If None, uses heuristic.
        trotter_order: 1 or 2 (default 2)

    Returns:
        outcomes_dict: {(R, t): [[perm, bits], ...]}
        metadata: dict with n_qubits, eigvals per R, etc.
    """
    if rng is None:
        rng = np.random.default_rng()

    import time as _time

    n_geom = len(R_values)
    n_t = len(times)
    total_snapshots = n_geom * n_t * n_shadows
    evol_tag = f"trotter(order={trotter_order}, dt={trotter_dt})" if use_trotter else "exact"
    print(f"[info] H{n_atoms} shadow generation | {n_geom} geometries | {n_t} times | {n_shadows} shadows/pair | {total_snapshots} total | {evol_tag}", flush=True)

    # Each worker gets an independent seed derived from the parent rng.
    # This guarantees: (1) no shared RNG state across workers,
    # (2) deterministic results for a given parent seed, and
    # (3) reproducibility regardless of n_workers.
    geom_seeds = rng.integers(0, 2**63, size=n_geom).tolist()

    outcomes_dict = {}
    metadata = {"n_atoms": n_atoms, "eigvals": {}}
    wall_start = _time.time()

    worker_args = [
        (i_R, R, n_atoms, times, n_shadows, geom_seeds[i_R],
         use_trotter, trotter_dt, trotter_order)
        for i_R, R in enumerate(R_values)
    ]

    if n_workers is not None and n_workers > 1:
        from multiprocessing import Pool

        print(f"[info] parallel mode: {n_workers} workers across {n_geom} geometries", flush=True)
        with Pool(n_workers) as pool:
            # imap preserves input order (unlike imap_unordered), so
            # results[i] always corresponds to R_values[i]. Each worker
            # writes only to its own return value — no shared state.
            results = []
            for i, result in enumerate(pool.imap(_process_geometry, worker_args)):
                R, _, n_qubits, eigvals, elapsed = result
                E_gap = eigvals[1] - eigvals[0] if len(eigvals) > 1 else float("nan")
                print(f"[geom {i + 1:02d}/{n_geom:02d}] R={R:.2f}A | E0={eigvals[0]:.4e} | gap={E_gap:.4e} | time={elapsed:.1f}s", flush=True)
                results.append(result)
    else:
        print(f"[info] serial mode: {n_geom} geometries")
        results = []
        for i_R, args in enumerate(worker_args):
            result = _process_geometry(args)
            R, _, n_qubits, eigvals, elapsed = result
            E_gap = eigvals[1] - eigvals[0] if len(eigvals) > 1 else float("nan")
            print(f"[geom {i_R + 1:02d}/{n_geom:02d}] R={R:.2f}A | E0={eigvals[0]:.4e} | gap={E_gap:.4e} | time={elapsed:.1f}s")
            results.append(result)

    # Collect results
    for R, outcomes_for_R, n_qubits, eigvals, elapsed in results:
        metadata["eigvals"][R] = eigvals
        metadata["n_qubits"] = n_qubits
        for t, snapshots in outcomes_for_R.items():
            outcomes_dict[(R, t)] = snapshots

    wall_elapsed = _time.time() - wall_start
    elapsed_min = wall_elapsed / 60
    n_qubits = metadata["n_qubits"]
    print(f"[summary] H{n_atoms} | {n_geom} geometries | {total_snapshots} snapshots | {n_qubits} qubits | time={elapsed_min:.1f}min")

    return outcomes_dict, metadata


def main():
    parser = argparse.ArgumentParser(description="Generate fermionic shadow data")
    parser.add_argument("--n_atoms", type=int, default=4)
    parser.add_argument("--r_start", type=float, default=0.5)
    parser.add_argument("--r_end", type=float, default=3.0)
    parser.add_argument("--r_step", type=float, default=0.1)
    parser.add_argument("--t_max", type=float, default=10.0)
    parser.add_argument("--n_times", type=int, default=50)
    parser.add_argument("--n_shadows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--use_trotter", action="store_true",
                        help="Use Trotterized time evolution instead of exact")
    parser.add_argument("--trotter_dt", type=float, default=None,
                        help="Trotter step size (default: heuristic 0.5/||H||_1)")
    parser.add_argument("--trotter_order", type=int, default=2, choices=[1, 2],
                        help="Trotter order: 1 (Lie) or 2 (Suzuki, default)")
    parser.add_argument("--validate_trotter", action="store_true",
                        help="Run validation comparing exact vs Trotter at a few geometries")
    args = parser.parse_args()

    if args.validate_trotter:
        R_values = np.round(
            np.arange(args.r_start, args.r_end + args.r_step / 2, args.r_step), 2
        )
        times = np.linspace(0, args.t_max, args.n_times)
        # Validate at 3 geometries: short, mid, long R
        test_Rs = [R_values[0], R_values[len(R_values) // 2], R_values[-1]]
        for R in test_Rs:
            validate_trotter(
                args.n_atoms, R, times,
                trotter_dt=args.trotter_dt,
                trotter_order=args.trotter_order,
            )
            print()
        return

    rng = np.random.default_rng(args.seed)

    R_values = np.round(
        np.arange(args.r_start, args.r_end + args.r_step / 2, args.r_step), 2
    )
    times = np.linspace(0, args.t_max, args.n_times)

    print(f"[info] H{args.n_atoms} standalone data generation")
    print(f"[info] R values: {len(R_values)} in [{R_values[0]:.2f}, {R_values[-1]:.2f}]")
    print(f"[info] times: {len(times)} in [0, {args.t_max}]")
    print(f"[info] shadows per (R, t): {args.n_shadows}")
    print(f"[info] total snapshots: {len(R_values) * len(times) * args.n_shadows}")

    outcomes_dict, metadata = generate_chain_data(
        n_atoms=args.n_atoms,
        R_values=R_values,
        times=times,
        n_shadows=args.n_shadows,
        rng=rng,
        use_trotter=args.use_trotter,
        trotter_dt=args.trotter_dt,
        trotter_order=args.trotter_order,
    )

    # Save as HDF5
    from fermionic_pipeline.data.dataset import FermionicShadowDataset

    n_qubits = metadata["n_qubits"]
    dataset = FermionicShadowDataset.from_outcomes_dict(outcomes_dict, n_qubits)

    if args.output is None:
        output_dir = os.path.join(_REPO_ROOT, "data", "fermionic_shadows")
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"H{args.n_atoms}_shadows.h5")

    dataset.save_hdf5(args.output, split="all")
    print(f"[done] saved {len(dataset)} snapshots -> {args.output}")


if __name__ == "__main__":
    main()
