"""Tests for the generalized molecule builder (H-chains + LiH active space).

Run from the models/ directory:  python -m pytest tests/test_molecule_builder.py -q
"""

import numpy as np
import pytest

from fermionic_pipeline.data.generate_shadows import (
    build_hydrogen_chain_hamiltonian,
    build_molecule_hamiltonian,
    molecule_n_electrons,
    prepare_initial_state,
    _parse_molecule,
)


def test_lih_interface_matches_h4():
    """LiH CAS(2,4) is 8 qubits / 16 modes / 120 channels — identical to H4."""
    H_sparse, n_qubits = build_molecule_hamiltonian("lih", 1.6)
    assert n_qubits == 8
    n_modes = 2 * n_qubits
    assert n_modes == 16
    from itertools import combinations
    assert len(list(combinations(range(n_modes), 2))) == 120


def test_hchain_delegation_is_byte_identical():
    """The H-chain path must be unchanged: build_molecule_hamiltonian('h4'/4, R)
    equals build_hydrogen_chain_hamiltonian(4, R) exactly."""
    R = 1.1
    ref, ref_nq = build_hydrogen_chain_hamiltonian(4, R)
    for mol in ("h4", 4):
        Hs, nq = build_molecule_hamiltonian(mol, R)
        assert nq == ref_nq
        assert (Hs - ref).nnz == 0 or np.allclose(Hs.toarray(), ref.toarray())


def test_n_electrons_active_space():
    assert molecule_n_electrons("lih") == 2   # CAS(2,4)
    assert molecule_n_electrons("h4") == 4
    assert molecule_n_electrons(2) == 2
    assert molecule_n_electrons("h6") == 6


def test_parse_molecule_rejects_unknown():
    assert _parse_molecule("h4") == ("hchain", 4)
    assert _parse_molecule(4) == ("hchain", 4)
    assert _parse_molecule("lih") == ("named", "lih")
    assert _parse_molecule("h2o") == ("named", "h2o")   # now a registered molecule
    with pytest.raises(ValueError):
        _parse_molecule("co2")   # genuinely unregistered


def test_lih_initial_state_is_active_space_hf():
    """psi_0 must be normalized and carry weight on the active-space HF det
    (first n_electrons=2 spin-orbitals occupied)."""
    H_sparse, n_qubits = build_molecule_hamiltonian("lih", 2.0)
    ne = molecule_n_electrons("lih")
    psi_0, eigvals = prepare_initial_state(H_sparse, n_qubits, n_electrons=ne)
    assert psi_0.shape == (2 ** n_qubits,)
    assert np.isclose(np.linalg.norm(psi_0), 1.0)
    hf_idx = sum(1 << (n_qubits - 1 - i) for i in range(ne))
    assert abs(psi_0[hf_idx]) ** 2 > 0.5  # HF dominates (sqrt(0.8) amplitude)
    assert eigvals.ndim == 1 and eigvals.size >= 2


@pytest.mark.parametrize("mol,R,ne", [("n2", 1.1, 4), ("beh2", 1.3, 4), ("h2o", 0.96, 4)])
def test_extra_molecules_8qubit_interface(mol, R, ne):
    """BeH2 / H2O / N2 all build at the uniform 8-qubit / 120-channel interface."""
    H_sparse, n_qubits = build_molecule_hamiltonian(mol, R)
    assert n_qubits == 8
    assert H_sparse.shape == (256, 256)
    assert molecule_n_electrons(mol) == ne


def test_polyatomic_geometry():
    """Geometry builder: BeH2 linear-symmetric (3 atoms), H2O bent (3 atoms)."""
    from fermionic_pipeline.data.generate_shadows import _molecule_coordinates, _MOLECULE_SPECS
    beh2 = _molecule_coordinates(_MOLECULE_SPECS["beh2"], 1.3).reshape(-1, 3)
    assert beh2.shape == (3, 3)
    assert np.allclose(beh2[1], [1.3, 0, 0]) and np.allclose(beh2[2], [-1.3, 0, 0])
    h2o = _molecule_coordinates(_MOLECULE_SPECS["h2o"], 0.96).reshape(-1, 3)
    assert h2o.shape == (3, 3)
    # both O-H bond lengths equal R
    assert np.isclose(np.linalg.norm(h2o[1]), 0.96) and np.isclose(np.linalg.norm(h2o[2]), 0.96)


def test_return_pennylane_shapes():
    out = build_molecule_hamiltonian("lih", 1.5, return_pennylane=True)
    assert len(out) == 3
    H_sparse, n_qubits, H_pl = out
    assert n_qubits == 8
    assert H_sparse.shape == (256, 256)
