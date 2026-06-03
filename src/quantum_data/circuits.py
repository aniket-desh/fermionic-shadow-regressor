import numpy as np
import cirq

from openfermion import MolecularData
from openfermion.utils import jordan_wigner
from openfermion.transforms import get_fermion_operator
from openfermionpyscf import run_pyscf


########################################
# 1. PySCF + OpenFermion electronic structure
########################################


def run_rhf_openfermion(
    symbols,
    coords,
    charge=0,
    mult=1,
    basis="sto-3g",
    active_electrons=None,
    active_orbitals=None,
):
    """
    Build molecule in PySCF, run RHF via openfermionpyscf.run_pyscf,
    optionally truncate to active space, and return:
      mol_of   : OpenFermion MolecularData
      n_qubits : number of spin orbitals in active space
      hf_bitstr: np.array of 0/1 occupations in that active space ordering
    """

    geometry = list(zip(symbols, coords))
    spin = mult - 1  # PySCF convention: spin = (#alpha - #beta)

    # Create MolecularData container (OpenFermion style)
    mol_of = MolecularData(
        geometry=geometry, basis=basis, charge=charge, multiplicity=mult
    )

    # Run pyscf backend to fill in integrals / HF solution
    mol_of = run_pyscf(
        mol_of,
        run_scf=True,
        run_mp2=False,
        run_cisd=False,
        run_ccsd=False,
        run_fci=False,
    )

    # Now mol_of has:
    # - one_body_integrals (spatial MO)
    # - two_body_integrals (spatial MO chemist notation)
    # - n_electrons
    # - n_orbitals  (spatial)
    # We'll build an active space if requested.

    n_electrons_total = mol_of.n_electrons
    n_orb_total = mol_of.n_orbitals  # spatial

    if active_orbitals is None:
        n_orb_act = n_orb_total
        first_orb = 0
    else:
        n_orb_act = active_orbitals
        first_orb = 0  # simplest: just take lowest-energy orbitals

    # electrons in active space
    if active_electrons is None:
        n_elec_act = n_electrons_total
    else:
        n_elec_act = active_electrons

    # slice integrals to active window [first_orb:first_orb+n_orb_act]
    sl = slice(first_orb, first_orb + n_orb_act)
    h1 = mol_of.one_body_integrals[sl, sl]
    h2 = mol_of.two_body_integrals[sl, sl, sl, sl]

    # Build a new MolecularData for JUST the active space
    # Note: geometry stays same; we'll just overwrite integrals.
    mol_active = MolecularData(
        geometry=geometry, basis=basis, charge=charge, multiplicity=mult
    )
    mol_active.one_body_integrals = h1
    mol_active.two_body_integrals = h2
    mol_active.n_electrons = n_elec_act
    mol_active.n_orbitals = n_orb_act
    mol_active.canonical_orbitals = True

    # Number of *spin* orbitals = 2 * n_orb_act
    n_qubits = 2 * n_orb_act

    # Hartree–Fock reference occupation in spin-orbital ordering
    # Convention: spin orbitals [0,1,...] => (orb0_alpha, orb0_beta, orb1_alpha, orb1_beta, ...)
    hf_bitstr = np.zeros(n_qubits, dtype=int)
    for so in range(n_elec_act):
        hf_bitstr[so] = 1

    return mol_active, n_qubits, hf_bitstr


########################################
# 2. Build qubit Hamiltonian with JW
########################################


def get_qubit_hamiltonian(mol_active):
    """
    Take MolecularData (with one_body_integrals, two_body_integrals),
    build fermionic Hamiltonian, then JW map -> QubitOperator.
    Returns list of (coeff, pauli_string_dict)
    where pauli_string_dict is {qubit_index: 'X'/'Y'/'Z', ...}.
    """

    # Fermionic Hamiltonian
    fermion_ham = mol_active.get_molecular_hamiltonian()
    fermion_op = get_fermion_operator(fermion_ham)

    # Jordan–Wigner to qubits
    qubit_op = jordan_wigner(fermion_op)

    # qubit_op.terms is a dict: { ((i,'X'), (j,'Z'), ...): coeff }
    pauli_terms = []
    for term, coeff in qubit_op.terms.items():
        # term is a tuple like ((qubit_index, 'X'), (qubit_index2,'Y'), ...)
        # build dense string dict
        pstr = {}
        for q, p in term:
            pstr[q] = p
        pauli_terms.append((coeff, pstr))

    return pauli_terms


########################################
# 3. Circuit synthesis helpers
########################################


def _prepare_hf_ops(hf_bitstr, qubits):
    """X on occupied spin orbitals."""
    ops = []
    for bit, q in zip(hf_bitstr, qubits):
        if bit == 1:
            ops.append(cirq.X(q))
    return ops


def _single_excitation_block(i, j, theta, qubits):
    """
    Very lightweight "single excitation" between spin-orbitals i and j.
    This is *not* the full fermionic number-conserving generator with phases,
    but a simple entangling Ry sandwich that mixes |10> and |01>.
    Good enough to kick you off the HF eigenstate.
    """
    qi, qj = qubits[i], qubits[j]
    return [
        cirq.CNOT(qj, qi),
        cirq.ry(theta)(qi),
        cirq.CNOT(qj, qi),
    ]


def _basis_change_for_pauli(p, q):
    # map Pauli p on qubit q to Z via local Cliffords.
    # returns ops_before, ops_after
    if p == "X":
        return [cirq.H(q)], [cirq.H(q)]
    if p == "Y":
        # S† H before, H S after
        return [cirq.H(q), cirq.S(q) ** -1], [cirq.S(q), cirq.H(q)]
    if p == "Z":
        return [], []
    raise ValueError("bad pauli")


def _exp_minus_i_dt_coeff_P(coeff, pauli_dict, dt, qubits):
    """
    Build exp(-i * coeff * dt * P) for P = ⊗_k P_k.
    pauli_dict: {q_index: 'X'/'Y'/'Z'}
    Strategy:
      1. basis change each involved qubit s.t. P_k -> Z
      2. ladder CNOTs to last qubit
      3. Rz(2*coeff*dt)
      4. uncompute CNOTs
      5. undo basis change
    """
    ops = []

    # Which qubits actually appear?
    involved = sorted(pauli_dict.keys())
    if len(involved) == 0:
        return ops

    # 1. basis change
    basis_befores = {}
    basis_afters = {}
    for q_idx in involved:
        qb = qubits[q_idx]
        p = pauli_dict[q_idx]
        b_before, b_after = _basis_change_for_pauli(p, qb)
        basis_befores[q_idx] = b_before
        basis_afters[q_idx] = b_after
        ops += b_before

    # 2. entangle to last qubit
    target = qubits[involved[-1]]
    for q_idx in involved[:-1]:
        ops.append(cirq.CNOT(qubits[q_idx], target))

    # 3. rotation: exp(-i * theta/2 * Z) with theta = 2*coeff*dt
    theta = (
        2.0 * coeff.real * dt
    )  # assuming coeff ~ real (chem hamiltonian is Hermitian real)
    ops.append(cirq.rz(theta)(target))

    # 4. uncompute
    for q_idx in reversed(involved[:-1]):
        ops.append(cirq.CNOT(qubits[q_idx], target))

    # 5. undo basis change
    for q_idx in reversed(involved):
        ops += basis_afters[q_idx]

    return ops


def _trotter_layer(pauli_terms, qubits, dt):
    """
    One first-order Trotter layer:
    prod_k exp(-i * coeff_k * dt * P_k)
    """
    ops = []
    for coeff, pstr in pauli_terms:
        ops += _exp_minus_i_dt_coeff_P(coeff, pstr, dt, qubits)
    return ops


########################################
# 4. Main API
########################################


def make_time_evolution_circuit(
    symbols,
    coords,
    charge=0,
    mult=1,
    basis="sto-3g",
    active_electrons=None,
    active_orbitals=None,
    theta_excitation=0.2,
    t=1.0,
    k_trotter=2,
):
    """
    Returns:
        circuit: cirq.Circuit
        qubits:  list[cirq.LineQubit]
        pauli_terms: list of (coeff, {q:'X'/'Y'/'Z'})
    """

    # 1. RHF + active space via PySCF/OpenFermion
    mol_active, n_qubits, hf_bitstr = run_rhf_openfermion(
        symbols,
        coords,
        charge=charge,
        mult=mult,
        basis=basis,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )

    # 2. JW qubit Hamiltonian
    pauli_terms = get_qubit_hamiltonian(mol_active)

    # 3. choose a simple single excitation
    #
    # We just pick the first occupied spin orbital i and first virtual j.
    # occupied = [0 .. n_elec-1], virtual = [n_elec .. n_qubits-1]
    n_elec = mol_active.n_electrons
    if n_elec < 1 or n_elec >= n_qubits:
        exc_pair = None
    else:
        exc_pair = (0, n_elec)  # excite from lowest occ to first virt

    # 4. build circuit ops
    qubits = cirq.LineQubit.range(n_qubits)
    ops = []

    # HF reference
    ops += _prepare_hf_ops(hf_bitstr, qubits)

    # small single excitation to move off eigenstate
    if exc_pair is not None:
        i, j = exc_pair
        ops += _single_excitation_block(i, j, theta_excitation, qubits)

    # real-time evolution under H with first-order Trotter, k_trotter steps
    dt = t / k_trotter
    for _ in range(k_trotter):
        ops += _trotter_layer(pauli_terms, qubits, dt)

    circuit = cirq.Circuit(ops)
    return circuit, qubits, pauli_terms


########################################
# example
########################################

if __name__ == "__main__":
    print("Example: time evolution circuit for H2O molecule")
    symbols = ["O", "H", "H"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.0, 0.7586, 0.5859],
        [0.0, -0.7586, 0.5859],
    ]

    print("Building time evolution circuit for H2O...")
    circuit, qubits, H_terms = make_time_evolution_circuit(
        symbols,
        coords,
        charge=0,
        mult=1,
        basis="sto-3g",
        active_electrons=2,
        active_orbitals=2,
        theta_excitation=0.2,
        t=1.0,
        k_trotter=2,
    )

    print("Qubits:", qubits)
    print("Circuit:")
    print(cirq.Circuit(circuit))
    print("First 5 H terms:")
    for coeff, term in H_terms[:5]:
        # pretty print pauli string
        ps = ["I"] * len(qubits)
        for q, p in term.items():
            ps[q] = p
        print(coeff, "".join(ps))
