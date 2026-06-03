import sys
import os

# fermion_shadows_cirq_simulation uses a bare import of optimal_matchgate_circuit,
# need to add the path to sys.path before importing
_FERMION_SIM_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "symmetry_adjusted_classical_shadows_main",
    "fermion_shadows",
    "simulation",
)
if _FERMION_SIM_DIR not in sys.path:
    sys.path.insert(0, _FERMION_SIM_DIR)

from symmetry_adjusted_classical_shadows_main.fermion_shadows.simulation import (  # noqa: E402
    fermion_shadows_cirq_simulation as _ferm,
)
from symmetry_adjusted_classical_shadows_main.pauli_shadows.simulation import (  # noqa: E402
    pauli_shadows_cirq_simulation as _pauli,
)


def pauli_shadows(circuit, qubits, simulator, n_shadows) -> list:
    # random single-qubit cliffords
    # returns list of (pauli_basis, bitstring)
    return _pauli.pauli_shadow_sampling(
        qubits=qubits,
        circuit=circuit,
        simulator=simulator,
        repetitions=n_shadows,
    )


def fermionic_shadows(circuit, qubits, simulator, n_shadows) -> list:
    # random matchgate (majorana permutation)
    # returns list of (permutation, bitstring)
    return _ferm.sample_shadows_full_system(
        qubits=qubits,
        circuit=circuit,
        simulator=simulator,
        repetitions=n_shadows,
    )
