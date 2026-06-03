"""
Regression dataset: exact signal matrix entries for direct observable prediction.

Computes the exact convergent values of the matchgate shadow estimator by
analytically evaluating E_b[estimator | Q] for each permutation Q, then
averaging over Q's. This gives noise-free D matrix targets that are fully
consistent with the shadow estimator's conventions.

Can generate from scratch (statevector + matchgate) or from an existing
exact conditional HDF5 (reuses the stored p(b|Q,R,t) probabilities).

Optimizations for large geometry grids:
  - Precompute Q decompositions once (Givens factorization is Q-dependent, not state-dependent)
  - Precompute per-Q estimator metadata (target keys, parities, sign factors)
  - Vectorize marginal → estimator accumulation
  - Parallelize across geometries with multiprocessing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time as _time
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Optional, Sequence

import numpy as np
import torch
from scipy.special import comb as binom
from torch.utils.data import Dataset
from tqdm.auto import tqdm

try:
    import h5py
except ModuleNotFoundError:
    h5py = None

from fermionic_pipeline.data.generate_shadows import (
    _apply_1q,
    _apply_majorana_rotation,
    _apply_matchgate,
    _apply_sign_gates,
    _apply_standard_majorana_givens,
    _zpow,
    build_hydrogen_chain_hamiltonian,
    compute_hf_orbital_energies,
    prepare_initial_state,
    time_evolve,
    time_evolve_trotter,
)
from fermionic_pipeline.inference.spectral_analysis import _precompute_diagonal_ops


def _require_h5py():
    if h5py is None:
        raise ImportError("h5py is required for regression dataset storage.")


# ── Observable key utilities ─────────────────────────────────────────

def majorana_2pt_keys(n_modes: int):
    """All degree-2 Majorana operator keys (i, j) with i < j."""
    return list(combinations(range(n_modes), 2))


# ── Precomputed matchgate decomposition ──────────────────────────────

def _precompute_decomposition(Q):
    """Precompute Givens decomposition of Q (expensive, do once per Q).

    Returns (left_rotations, right_rotations, signs) where each rotation
    entry is augmented with pre-resolved Givens sub-decompositions so that
    _apply_decomposition_fast never calls standard_givens_decomposition at
    runtime.
    """
    _VENDOR_DIR = os.path.join(os.path.dirname(__file__), "..", "vendor")
    if os.path.abspath(_VENDOR_DIR) not in sys.path:
        sys.path.insert(0, os.path.abspath(_VENDOR_DIR))
    from optimal_matchgate_circuit import (
        majorana_block_decomposition,
        standard_givens_decomposition,
    )

    left_rotations, right_rotations, signs = majorana_block_decomposition(Q)

    # Pre-resolve the Givens sub-decompositions for every orth_mat so that
    # _apply_decomposition_fast avoids recomputing them on every call.
    def _resolve_rotations(rotations, n_qubits):
        resolved = []
        for p, _, orth_mat in rotations:
            sub_rotations, sub_signs = standard_givens_decomposition(orth_mat)
            resolved.append((p, sub_rotations, sub_signs))
        return resolved

    # n_qubits needed for full_signs array size — infer from signs length
    n_qubits = len(signs) // 2

    resolved_right = _resolve_rotations(right_rotations, n_qubits)
    resolved_left_tail = _resolve_rotations(left_rotations[1:], n_qubits)
    left_zpow = left_rotations[0]  # (p, _, theta)

    return resolved_right, resolved_left_tail, left_zpow, signs, n_qubits


def _apply_decomposition(decomp, psi, n_qubits):
    """Apply a precomputed matchgate decomposition to a statevector.

    Uses pre-resolved Givens sub-decompositions to avoid redundant
    standard_givens_decomposition calls (the key optimization).
    """
    resolved_right, resolved_left_tail, left_zpow, signs, _ = decomp
    psi = psi.copy().astype(complex)

    for p, sub_rotations, sub_signs in resolved_right:
        psi = _apply_resolved_givens(psi, sub_rotations, sub_signs, p, n_qubits)

    psi = _apply_sign_gates(psi, signs, n_qubits)

    p_zp, _, theta = left_zpow
    psi = _apply_1q(psi, _zpow(theta / np.pi), p_zp // 2, n_qubits)

    for p, sub_rotations, sub_signs in resolved_left_tail:
        psi = _apply_resolved_givens(psi, sub_rotations, sub_signs, p, n_qubits)

    return psi


def _apply_resolved_givens(psi, sub_rotations, sub_signs, index_offset, n_qubits):
    """Apply pre-resolved Givens rotations (no standard_givens_decomposition call)."""
    full_signs = np.ones(2 * n_qubits)
    full_signs[index_offset:index_offset + len(sub_signs)] = sub_signs
    psi = _apply_sign_gates(psi, full_signs, n_qubits)

    for r, s, theta in sub_rotations:
        psi = _apply_majorana_rotation(
            psi, r + index_offset, s + index_offset, theta, n_qubits
        )
    return psi


# ── Time-batched primitives (operate on Psi: (D, T) instead of psi: (D,)) ──
# All matchgate operations are linear and act independently on each column,
# so we batch all N_T time points into one Psi matrix and apply each Q once.
# Removes ~N_T factor of Python overhead in the hot loop.

def _apply_1q_batched(Psi, gate, q, n):
    """Apply 2x2 gate to qubit q across all T columns of Psi: (D, T) -> (D, T)."""
    T = Psi.shape[1]
    d0 = 1 << q
    d1 = 1 << (n - q - 1)
    P = Psi.reshape(d0, 2, d1, T)
    P = np.einsum("ij,ajbt->aibt", gate, P)
    return P.reshape(-1, T)


def _apply_2q_batched(Psi, gate, q, n):
    """Apply 4x4 gate to adjacent qubits (q, q+1) across all T columns of Psi."""
    T = Psi.shape[1]
    d0 = 1 << q
    d1 = 1 << (n - q - 2)
    P = Psi.reshape(d0, 4, d1, T)
    P = np.einsum("ij,ajbt->aibt", gate, P)
    return P.reshape(-1, T)


def _apply_majorana_rotation_batched(Psi, p, q, theta, n_qubits):
    """Batched analog of _apply_majorana_rotation; mirrors its case structure."""
    from fermionic_pipeline.data.generate_shadows import _xxpow, _yypow

    if np.isclose(theta, 0.0):
        return Psi
    if p > q:
        p, q = q, p
    d = q - p

    if d == 1:
        if p % 2 == 0:
            i = p // 2
            return _apply_1q_batched(Psi, _zpow(theta / np.pi), i, n_qubits)
        i = (p - 1) // 2
        return _apply_2q_batched(Psi, _xxpow(theta / np.pi), i, n_qubits)
    if d == 2:
        if p % 2 == 0:
            i = p // 2
            Psi = _apply_1q_batched(Psi, _zpow(-0.5), i, n_qubits)
            Psi = _apply_2q_batched(Psi, _xxpow(-theta / np.pi), i, n_qubits)
            return _apply_1q_batched(Psi, _zpow(0.5), i, n_qubits)
        i = (p - 1) // 2
        Psi = _apply_1q_batched(Psi, _zpow(-0.5), i + 1, n_qubits)
        Psi = _apply_2q_batched(Psi, _xxpow(theta / np.pi), i, n_qubits)
        return _apply_1q_batched(Psi, _zpow(0.5), i + 1, n_qubits)
    if d == 3 and p % 2 == 0:
        i = p // 2
        return _apply_2q_batched(Psi, _yypow(-theta / np.pi), i, n_qubits)
    raise ValueError(f"Majorana indices ({p},{q}) do not map onto 2-local gates.")


def _apply_sign_gates_batched(Psi, signs, n_qubits):
    """Batched analog of _apply_sign_gates. Same diagonal-Pauli decomposition,
    applied via _apply_1q_batched."""
    from fermionic_pipeline.data.generate_shadows import _X, _Y, _Z

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
            Psi = _apply_1q_batched(Psi, _Z, p, n_qubits)
        elif pauli == (1, 0):
            Psi = _apply_1q_batched(Psi, _X, p, n_qubits)
        elif pauli == (1, 1):
            Psi = _apply_1q_batched(Psi, _Y, p, n_qubits)
    return Psi


def _apply_resolved_givens_batched(Psi, sub_rotations, sub_signs, index_offset, n_qubits):
    full_signs = np.ones(2 * n_qubits)
    full_signs[index_offset:index_offset + len(sub_signs)] = sub_signs
    Psi = _apply_sign_gates_batched(Psi, full_signs, n_qubits)
    for r, s, theta in sub_rotations:
        Psi = _apply_majorana_rotation_batched(
            Psi, r + index_offset, s + index_offset, theta, n_qubits
        )
    return Psi


def _apply_decomposition_batched(decomp, Psi, n_qubits):
    """Apply matchgate decomposition to Psi: (D, T). Returns rotated (D, T)."""
    resolved_right, resolved_left_tail, left_zpow, signs, _ = decomp
    Psi = Psi.copy().astype(complex)
    for p, sub_rotations, sub_signs in resolved_right:
        Psi = _apply_resolved_givens_batched(Psi, sub_rotations, sub_signs, p, n_qubits)
    Psi = _apply_sign_gates_batched(Psi, signs, n_qubits)
    p_zp, _, theta = left_zpow
    Psi = _apply_1q_batched(Psi, _zpow(theta / np.pi), p_zp // 2, n_qubits)
    for p, sub_rotations, sub_signs in resolved_left_tail:
        Psi = _apply_resolved_givens_batched(Psi, sub_rotations, sub_signs, p, n_qubits)
    return Psi


def _precompute_estimator_metadata(permutations, signs, n_qubits, diag_ops, degrees):
    """Precompute per-Q estimator routing: target keys, parities, sign factors.

    Returns a list (one per Q) of lists (one per diag_op) of:
        (target_key_tuple, combined_sign_and_parity, qubit_indices_array)
    """
    n_modes = 2 * n_qubits
    N_Q = len(permutations)
    metadata = []

    for q_idx in range(N_Q):
        perm = np.asarray(permutations[q_idx], dtype=np.intp)
        sgn = np.asarray(signs[q_idx], dtype=np.float64)
        inv_perm = np.empty(n_modes, dtype=np.intp)
        inv_perm[perm] = np.arange(n_modes)

        q_meta = []
        for diag_op, deg in zip(diag_ops, degrees):
            permuted = perm[diag_op]
            target = tuple(sorted(permuted))

            inv_applied = inv_perm[list(target)]
            inversions = sum(
                1 for i in range(len(inv_applied))
                for j in range(i + 1, len(inv_applied))
                if inv_applied[i] > inv_applied[j]
            )
            parity_sign = (-1) ** inversions
            sign_factor = float(np.prod(sgn[diag_op]))
            combined = sign_factor * parity_sign

            qubit_indices = diag_op[::2] // 2
            q_meta.append((target, combined, qubit_indices))
        metadata.append(q_meta)

    return metadata


def _precompute_vectorized_metadata(estimator_metadata, key_to_idx, K):
    """Convert per-Q metadata into vectorized arrays for fast accumulation.

    Returns per-Q arrays: (target_indices, combined_signs, qubit_idx_pairs)
    so the inner loop in _compute_signal_row_fast becomes pure numpy.
    """
    N_Q = len(estimator_metadata)
    vec_meta = []

    for q_idx in range(N_Q):
        q_meta = estimator_metadata[q_idx]
        target_indices = []
        combined_signs = []
        qubit_idx_list = []

        for target, combined, qubit_indices in q_meta:
            idx = key_to_idx.get(target)
            if idx is not None:
                target_indices.append(idx)
                combined_signs.append(combined)
                qubit_idx_list.append(qubit_indices)

        vec_meta.append((
            np.array(target_indices, dtype=np.intp),
            np.array(combined_signs, dtype=np.float64),
            np.array(qubit_idx_list, dtype=np.intp),  # (n_ops, n_qubit_per_op)
        ))

    return vec_meta


# ── Vectorized signal computation ────────────────────────────────────

def _build_bit_array(n_qubits: int) -> np.ndarray:
    """Precompute bitstring decomposition for marginal computation."""
    dim = 1 << n_qubits
    bits = np.zeros((dim, n_qubits), dtype=np.float64)
    for b in range(dim):
        for p in range(n_qubits):
            bits[b, p] = (b >> (n_qubits - 1 - p)) & 1
    return bits


def compute_signal_from_probs(
    probs: np.ndarray,
    permutation: np.ndarray,
    signs: np.ndarray,
    n_qubits: int,
    bit_array: np.ndarray,
    diag_ops: list,
    degrees: list,
) -> dict:
    """Compute expected shadow estimator contributions for one Q (unoptimized)."""
    n_modes = 2 * n_qubits
    perm = np.asarray(permutation, dtype=np.intp)
    sgn = np.asarray(signs, dtype=np.float64)

    marginals = (1.0 - bit_array).T @ probs
    expected_diag = 2.0 * marginals - 1.0

    inv_perm = np.empty(n_modes, dtype=np.intp)
    inv_perm[perm] = np.arange(n_modes)

    result = {}
    for diag_op, deg in zip(diag_ops, degrees):
        permuted = perm[diag_op]
        target = tuple(sorted(permuted))
        inv_applied = inv_perm[list(target)]
        inversions = sum(
            1 for i in range(len(inv_applied))
            for j in range(i + 1, len(inv_applied))
            if inv_applied[i] > inv_applied[j]
        )
        parity_sign = (-1) ** inversions
        sign_factor = float(np.prod(sgn[diag_op]))
        qubit_indices = diag_op[::2] // 2
        diag_val = float(np.prod(expected_diag[qubit_indices]))
        val = sign_factor * parity_sign * diag_val
        result[target] = result.get(target, 0.0) + val

    return result


def _compute_signal_row_fast(
    psi: np.ndarray,
    n_qubits: int,
    decompositions: list,
    vec_meta: list,
    bit_array: np.ndarray,
    shadow_coeff: float,
    K: int,
) -> np.ndarray:
    """Compute one row of the signal matrix (one time point) using precomputed data.

    This is the hot loop — called n_R * n_T times.
    Uses pre-resolved Givens decompositions and vectorized metadata.
    """
    N_Q = len(decompositions)
    accum = np.zeros(K, dtype=np.float64)

    for q_idx in range(N_Q):
        rotated = _apply_decomposition(decompositions[q_idx], psi, n_qubits)
        probs = np.abs(rotated) ** 2
        probs *= 1.0 / probs.sum()

        # Qubit marginals → expected diagonal elements
        marginals = (1.0 - bit_array).T @ probs
        expected_diag = 2.0 * marginals - 1.0

        # Vectorized accumulation (no Python loop over observables)
        target_indices, combined_signs, qubit_idx_pairs = vec_meta[q_idx]
        diag_vals = np.prod(expected_diag[qubit_idx_pairs], axis=1)
        np.add.at(accum, target_indices, combined_signs * diag_vals)

    accum *= shadow_coeff / N_Q
    return accum


def _compute_signal_block_fast(
    Psi: np.ndarray,
    n_qubits: int,
    decompositions: list,
    vec_meta: list,
    bit_array: np.ndarray,
    shadow_coeff: float,
    K: int,
) -> np.ndarray:
    """Time-batched analog of _compute_signal_row_fast.

    Psi: (D, T) complex matrix of state vectors at all T time points.
    Returns: (T, K) signal matrix.

    Removes the outer per-time loop's Python overhead by applying each Q's
    matchgate decomposition to all T columns at once via _apply_decomposition_batched.
    Mathematically identical to looping over t and calling _compute_signal_row_fast,
    but ~5-50x faster in practice for large T.
    """
    D, T = Psi.shape
    N_Q = len(decompositions)
    accum = np.zeros((T, K), dtype=np.float64)

    one_minus_bits_T = (1.0 - bit_array).T  # (n_qubits, D)

    for q_idx in range(N_Q):
        rotated = _apply_decomposition_batched(decompositions[q_idx], Psi, n_qubits)
        probs = np.abs(rotated) ** 2                                # (D, T)
        probs *= 1.0 / probs.sum(axis=0, keepdims=True)
        marginals = one_minus_bits_T @ probs                        # (n_qubits, T)
        expected_diag = 2.0 * marginals - 1.0                       # (n_qubits, T)

        target_indices, combined_signs, qubit_idx_pairs = vec_meta[q_idx]
        # qubit_idx_pairs: (n_ops, deg), gather then product over deg axis
        diag_vals = np.prod(expected_diag[qubit_idx_pairs, :], axis=1)  # (n_ops, T)
        # Per-time scatter-add into accum[:, target_indices]; use np.add.at
        # over the K-axis to handle duplicate target keys correctly.
        contrib = combined_signs[:, None] * diag_vals  # (n_ops, T)
        np.add.at(accum, (slice(None), target_indices), contrib.T)

    accum *= shadow_coeff / N_Q
    return accum


# ── Per-geometry worker ──────────────────────────────────────────────

def _compute_time_chunk(args):
    """Worker: compute signal rows for a chunk of time points."""
    (psi_list, n_qubits, decompositions, vec_meta, bit_array,
     shadow_coeff, K, t_indices) = args
    rows = np.zeros((len(t_indices), K), dtype=np.float64)
    for local_idx, t_idx in enumerate(t_indices):
        rows[local_idx] = _compute_signal_row_fast(
            psi_list[t_idx], n_qubits, decompositions, vec_meta,
            bit_array, shadow_coeff, K,
        )
    return t_indices, rows


def _process_geometry(args):
    """Worker for multiprocessing: process one geometry."""
    # Support both old 10-tuple and new 13-tuple with Trotter params
    if len(args) == 10:
        (r_idx, R, n_atoms, times, n_qubits, K,
         decompositions, vec_meta, bit_array, shadow_coeff) = args
        use_trotter, trotter_dt, trotter_order = False, None, 2
    else:
        (r_idx, R, n_atoms, times, n_qubits, K,
         decompositions, vec_meta, bit_array, shadow_coeff,
         use_trotter, trotter_dt, trotter_order) = args

    t_start = _time.time()

    if use_trotter:
        H_sparse, nq, H_pl = build_hydrogen_chain_hamiltonian(
            n_atoms, float(R), return_pennylane=True
        )
    else:
        H_sparse, nq = build_hydrogen_chain_hamiltonian(n_atoms, float(R))

    H_dense = H_sparse.toarray().astype(np.complex128)
    eigvals = np.sort(np.linalg.eigvalsh(H_dense).real)

    psi_0, _ = prepare_initial_state(H_sparse, n_qubits, n_electrons=n_atoms)

    if use_trotter:
        state_dict = time_evolve_trotter(
            H_pl, n_qubits, psi_0, times,
            trotter_dt=trotter_dt, trotter_order=trotter_order,
        )
    else:
        state_dict = time_evolve(H_sparse, psi_0, times)

    # Batch all T time points into Psi: (2^n, T) and apply each Q once.
    # Mathematically identical to the per-t loop below, but ~5-50x faster.
    dim = 1 << n_qubits
    Psi = np.empty((dim, len(times)), dtype=np.complex128)
    for t_idx, t in enumerate(times):
        Psi[:, t_idx] = state_dict[t].astype(np.complex128)
    D = _compute_signal_block_fast(
        Psi, n_qubits, decompositions, vec_meta, bit_array, shadow_coeff, K,
    )

    elapsed = _time.time() - t_start
    return r_idx, D, eigvals, elapsed


# ── Config ───────────────────────────────────────────────────────────

@dataclass
class RegressionDatasetConfig:
    n_atoms: int = 4
    r_start: float = 0.5
    r_end: float = 3.0
    r_step: float = 0.05
    r_dense_cutoff: float = 0.0
    r_dense_step: float = 0.01
    r_zoom_lo: Optional[float] = None    # zoom range lower bound (Å); replaces base R-points in [lo, hi]
    r_zoom_hi: Optional[float] = None    # zoom range upper bound (Å)
    r_zoom_step: Optional[float] = None  # finer step inside zoom range
    t_max: float = 100.0
    n_times: int = 500
    n_q: int = 1000
    seed: int = 42
    use_trotter: bool = False
    trotter_dt: Optional[float] = None
    trotter_order: int = 2

    @property
    def R_values(self) -> np.ndarray:
        if self.r_dense_cutoff > self.r_start:
            dense = np.arange(self.r_start, self.r_dense_cutoff, self.r_dense_step)
            coarse = np.arange(self.r_dense_cutoff, self.r_end + self.r_step / 2, self.r_step)
            base = np.concatenate([dense, coarse])
        else:
            base = np.arange(self.r_start, self.r_end + self.r_step / 2, self.r_step)

        if (self.r_zoom_lo is not None and self.r_zoom_hi is not None
                and self.r_zoom_step is not None):
            base = base[(base < self.r_zoom_lo) | (base > self.r_zoom_hi)]
            zoom = np.arange(self.r_zoom_lo, self.r_zoom_hi + self.r_zoom_step / 2, self.r_zoom_step)
            base = np.concatenate([base, zoom])

        return np.round(np.sort(np.unique(base)), 8)

    @property
    def times(self) -> np.ndarray:
        return np.linspace(0.0, self.t_max, self.n_times, dtype=np.float64)


# ── HDF5 handle + PyTorch dataset ────────────────────────────────────

class RegressionDatasetHandle:
    """Metadata wrapper for the regression HDF5 dataset."""

    def __init__(self, path: str):
        _require_h5py()
        self.path = path
        with h5py.File(path, "r") as f:
            self.n_atoms = int(f.attrs["n_atoms"])
            self.n_qubits = int(f.attrs["n_qubits"])
            self.n_modes = int(f.attrs["n_modes"])
            self.n_observables = int(f.attrs["n_observables"])
            self.R_values = f["R_values"][:]
            self.times = f["times"][:]
            self.observable_keys = [tuple(k) for k in f["observable_keys"][:]]
            self.expectations = f["expectations"][:]  # (n_R, n_t, K)
            self.eigvals = f["eigvals"][:]
            if "hf_orbital_energies" in f:
                self.hf_orbital_energies = f["hf_orbital_energies"][:]  # (n_R, n_orb)
            else:
                self.hf_orbital_energies = None
            if "omega_op" in f:
                self.omega_op = f["omega_op"][:]  # (n_R,)
            else:
                self.omega_op = None


class RegressionTorchDataset(Dataset):
    """PyTorch dataset: (R, t) → K observable expectations."""

    def __init__(
        self, handle: RegressionDatasetHandle, r_indices: Optional[Sequence[int]] = None
    ):
        self.handle = handle
        if r_indices is None:
            r_indices = list(range(len(handle.R_values)))
        self.r_indices = np.array(sorted(r_indices), dtype=np.int64)
        self.n_t = len(handle.times)

    def __len__(self):
        return len(self.r_indices) * self.n_t

    @property
    def has_orbital_energies(self):
        return self.handle.hf_orbital_energies is not None

    @property
    def has_omega_op(self):
        return self.handle.omega_op is not None

    def __getitem__(self, idx):
        t_idx = idx % self.n_t
        r_local = idx // self.n_t
        r_idx = int(self.r_indices[r_local])

        rt = np.array(
            [self.handle.R_values[r_idx], self.handle.times[t_idx]], dtype=np.float32
        )
        targets = self.handle.expectations[r_idx, t_idx].astype(np.float32)

        out = {"rt": torch.from_numpy(rt), "targets": torch.from_numpy(targets)}
        if self.has_orbital_energies:
            out["orb_e"] = torch.from_numpy(
                self.handle.hf_orbital_energies[r_idx].astype(np.float32)
            )
        if self.has_omega_op:
            out["omega_op"] = torch.tensor(
                float(self.handle.omega_op[r_idx]), dtype=torch.float32
            )

        # Backward-compat tuple form (no omega_op consumers): preserves the
        # 2- and 3-tuple shapes expected by existing trainer/eval code paths.
        if "omega_op" in out:
            if "orb_e" in out:
                return out["rt"], out["orb_e"], out["omega_op"], out["targets"]
            return out["rt"], out["omega_op"], out["targets"]
        if "orb_e" in out:
            return out["rt"], out["orb_e"], out["targets"]
        return out["rt"], out["targets"]


# ── Generation ───────────────────────────────────────────────────────

def generate_regression_dataset(
    output_path: str,
    config: RegressionDatasetConfig,
    exact_conditional_path: Optional[str] = None,
    n_workers: Optional[int] = None,
) -> str:
    """Generate exact regression targets.

    If exact_conditional_path is given, reuses stored p(b|Q,R,t) from the
    classifier pipeline (fast, no matchgate computation). Otherwise generates
    from scratch using statevector + matchgate with precomputed decompositions
    and optional multiprocessing.
    """
    _require_h5py()
    R_values = config.R_values
    times = config.times
    n_R = len(R_values)

    H_probe, n_qubits = build_hydrogen_chain_hamiltonian(config.n_atoms, float(R_values[0]))
    n_modes = 2 * n_qubits
    k = 1
    all_keys = majorana_2pt_keys(n_modes)
    K = len(all_keys)
    key_to_idx = {key: i for i, key in enumerate(all_keys)}

    diag_ops, degrees = _precompute_diagonal_ops(n_qubits, k)
    shadow_coeff = binom(2 * n_qubits, 2, exact=True) / binom(n_qubits, 1, exact=True)
    bit_array = _build_bit_array(n_qubits)

    # Q library
    from fermionic_pipeline.data.exact_conditional_dataset import (
        sample_signed_permutation_library,
        signed_permutation_to_matrix,
    )

    if exact_conditional_path is not None:
        with h5py.File(exact_conditional_path, "r") as f:
            permutations = f["permutations"][:]
            signs = f["signs"][:]
        N_Q = len(permutations)
        print(f"[info] reusing Q library from {exact_conditional_path} (N_Q={N_Q})")
    else:
        N_Q = config.n_q
        permutations, signs = sample_signed_permutation_library(
            n_modes, N_Q, seed=config.seed
        )
        print(f"[info] sampled fresh Q library (N_Q={N_Q})")

    print(f"[info] {n_R} geometries | {len(times)} times | {N_Q} Q's | K={K} observables")

    # ── Precompute Q decompositions (done ONCE, reused for all states) ──
    use_hdf5 = exact_conditional_path is not None
    decompositions = None
    estimator_metadata = None

    if not use_hdf5:
        print(f"[info] precomputing {N_Q} matchgate decompositions...", end=" ", flush=True)
        t0 = _time.time()
        Q_matrices = [
            signed_permutation_to_matrix(permutations[i], signs[i]).astype(np.float64)
            for i in range(N_Q)
        ]
        decompositions = [_precompute_decomposition(Q) for Q in Q_matrices]
        print(f"done ({_time.time() - t0:.1f}s)")

        print(f"[info] precomputing estimator metadata...", end=" ", flush=True)
        t0 = _time.time()
        estimator_metadata = _precompute_estimator_metadata(
            permutations, signs, n_qubits, diag_ops, degrees
        )
        vec_meta = _precompute_vectorized_metadata(estimator_metadata, key_to_idx, K)
        print(f"done ({_time.time() - t0:.1f}s)")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # ── Precompute HF orbital energies (cheap, one RHF per geometry) ──
    print(f"[info] computing HF orbital energies...", end=" ", flush=True)
    t0 = _time.time()
    all_hf_energies = np.array([
        compute_hf_orbital_energies(config.n_atoms, float(R))
        for R in R_values
    ])
    n_orb = all_hf_energies.shape[1]
    print(f"done ({_time.time() - t0:.1f}s, {n_orb} orbitals per geometry)")

    # ── Generate ──
    if not use_hdf5 and n_workers and n_workers > 1:
        # Parallel across geometries
        from multiprocessing import Pool

        worker_args = [
            (r_idx, R, config.n_atoms, times, n_qubits, K,
             decompositions, vec_meta, bit_array, shadow_coeff,
             config.use_trotter, config.trotter_dt, config.trotter_order)
            for r_idx, R in enumerate(R_values)
        ]

        print(f"[info] parallel mode: {n_workers} workers across {n_R} geometries")
        all_D = [None] * n_R
        all_eigvals = [None] * n_R

        with Pool(n_workers) as pool:
            for r_idx, D, eigvals, elapsed in pool.imap_unordered(
                _process_geometry, worker_args
            ):
                R = R_values[r_idx]
                all_D[r_idx] = D
                all_eigvals[r_idx] = eigvals
                print(
                    f"[geom {r_idx + 1:02d}/{n_R:02d}] R={R:.3f} | "
                    f"time={elapsed:.1f}s",
                    flush=True,
                )

        # Write to HDF5
        with h5py.File(output_path, "w") as f:
            f.attrs["n_atoms"] = config.n_atoms
            f.attrs["n_qubits"] = n_qubits
            f.attrs["n_modes"] = n_modes
            f.attrs["n_observables"] = K
            f.attrs["config_json"] = json.dumps(asdict(config))

            f.create_dataset("R_values", data=R_values)
            f.create_dataset("times", data=times)
            f.create_dataset("observable_keys", data=np.array(all_keys, dtype=np.int32))
            f.create_dataset(
                "expectations",
                data=np.stack(all_D),
                dtype=np.float64,
                chunks=(1, len(times), K),
                compression="lzf",
            )
            f.create_dataset(
                "eigvals",
                data=np.stack(all_eigvals),
                dtype=np.float64,
                compression="lzf",
            )
            f.create_dataset(
                "hf_orbital_energies",
                data=all_hf_energies,
                dtype=np.float64,
            )

    else:
        # Serial (or HDF5-backed)
        shadow_coeffs = {1: shadow_coeff}

        with h5py.File(output_path, "w") as f:
            f.attrs["n_atoms"] = config.n_atoms
            f.attrs["n_qubits"] = n_qubits
            f.attrs["n_modes"] = n_modes
            f.attrs["n_observables"] = K
            f.attrs["config_json"] = json.dumps(asdict(config))

            f.create_dataset("R_values", data=R_values)
            f.create_dataset("times", data=times)
            f.create_dataset("observable_keys", data=np.array(all_keys, dtype=np.int32))
            exp_ds = f.create_dataset(
                "expectations",
                shape=(n_R, len(times), K),
                dtype=np.float64,
                chunks=(1, len(times), K),
                compression="lzf",
            )
            eigvals_ds = f.create_dataset(
                "eigvals",
                shape=(n_R, 1 << n_qubits),
                dtype=np.float64,
                compression="lzf",
            )
            f.create_dataset(
                "hf_orbital_energies",
                data=all_hf_energies,
                dtype=np.float64,
            )

            for r_idx, R in enumerate(tqdm(R_values, desc="Regression dataset")):
                t0 = _time.time()

                if config.use_trotter:
                    H_sparse, nq, H_pl = build_hydrogen_chain_hamiltonian(
                        config.n_atoms, float(R), return_pennylane=True
                    )
                else:
                    H_sparse, nq = build_hydrogen_chain_hamiltonian(config.n_atoms, float(R))

                H_dense = H_sparse.toarray().astype(np.complex128)
                eigvals_ds[r_idx] = np.sort(np.linalg.eigvalsh(H_dense).real)

                psi_0, _ = prepare_initial_state(H_sparse, n_qubits, n_electrons=config.n_atoms)

                if config.use_trotter:
                    state_dict = time_evolve_trotter(
                        H_pl, n_qubits, psi_0, times,
                        trotter_dt=config.trotter_dt,
                        trotter_order=config.trotter_order,
                    )
                else:
                    state_dict = time_evolve(H_sparse, psi_0, times)

                for t_idx in range(len(times)):
                    if use_hdf5:
                        from fermionic_pipeline.data.regression_dataset import (
                            compute_signal_matrix_from_hdf5,
                        )
                        accum = compute_signal_matrix_from_hdf5(
                            exact_conditional_path, r_idx, t_idx,
                            permutations, signs, n_qubits,
                            bit_array, diag_ops, degrees, shadow_coeffs,
                        )
                        row = np.zeros(K, dtype=np.float64)
                        for key, val in accum.items():
                            if key in key_to_idx:
                                row[key_to_idx[key]] = val
                        exp_ds[r_idx, t_idx] = row
                    else:
                        psi_t = state_dict[times[t_idx]].astype(np.complex128)
                        exp_ds[r_idx, t_idx] = _compute_signal_row_fast(
                            psi_t, n_qubits, decompositions, vec_meta,
                            bit_array, shadow_coeff, K,
                        )

                elapsed = _time.time() - t0
                print(
                    f"[geom {r_idx + 1:02d}/{n_R:02d}] R={R:.3f} | time={elapsed:.1f}s",
                    flush=True,
                )

    return output_path


def compute_signal_matrix_from_hdf5(
    h5_path, r_idx, t_idx, permutations, signs, n_qubits,
    bit_array, diag_ops, degrees, shadow_coeffs,
):
    """Compute exact signal for one (R, t) using stored exact conditionals."""
    N_Q = len(permutations)
    accum = {}
    with h5py.File(h5_path, "r") as f:
        for q_idx in range(N_Q):
            probs = f["probabilities"][r_idx, t_idx, q_idx].astype(np.float64)
            contribs = compute_signal_from_probs(
                probs, permutations[q_idx], signs[q_idx], n_qubits,
                bit_array, diag_ops, degrees,
            )
            for key, val in contribs.items():
                accum[key] = accum.get(key, 0.0) + val
    for key in accum:
        j = len(key) // 2
        accum[key] *= shadow_coeffs[j] / N_Q
    return accum


def load_config_defaults(config_path: str) -> RegressionDatasetConfig:
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    dc = cfg["data"]
    return RegressionDatasetConfig(
        n_atoms=dc["n_atoms"],
        r_start=dc["r_start"],
        r_end=dc["r_end"],
        r_step=dc["r_step"],
    )


def main():
    parser = argparse.ArgumentParser(description="Build regression target dataset")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--exact_conditional", type=str, default=None,
                        help="Path to exact conditional HDF5 (reuses Q library + probs)")
    parser.add_argument("--n_atoms", type=int, default=None)
    parser.add_argument("--r_start", type=float, default=None)
    parser.add_argument("--r_end", type=float, default=None)
    parser.add_argument("--r_step", type=float, default=None)
    parser.add_argument("--r_dense_cutoff", type=float, default=None,
                        help="Use r_dense_step below this R, r_step above")
    parser.add_argument("--r_dense_step", type=float, default=None)
    parser.add_argument("--r_zoom_lo", type=float, default=None,
                        help="Zoom range lower bound (Å); replaces base R-points in [lo, hi] with finer grid.")
    parser.add_argument("--r_zoom_hi", type=float, default=None,
                        help="Zoom range upper bound (Å).")
    parser.add_argument("--r_zoom_step", type=float, default=None,
                        help="Finer step inside zoom range.")
    parser.add_argument("--t_max", type=float, default=None)
    parser.add_argument("--n_times", type=int, default=None)
    parser.add_argument("--n_q", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument("--use_trotter", action="store_true",
                        help="Use Trotterized time evolution instead of exact")
    parser.add_argument("--trotter_dt", type=float, default=None,
                        help="Trotter step size (default: heuristic 0.5/||H||_1)")
    parser.add_argument("--trotter_order", type=int, default=2, choices=[1, 2],
                        help="Trotter order: 1 (Lie) or 2 (Suzuki, default)")
    args = parser.parse_args()

    if args.config is not None:
        config = load_config_defaults(args.config)
    else:
        config = RegressionDatasetConfig()

    for field in ["n_atoms", "r_start", "r_end", "r_step", "r_dense_cutoff", "r_dense_step",
                  "r_zoom_lo", "r_zoom_hi", "r_zoom_step", "t_max", "n_times"]:
        value = getattr(args, field)
        if value is not None:
            setattr(config, field, value)
    config.n_q = args.n_q
    config.seed = args.seed
    config.use_trotter = args.use_trotter
    if args.trotter_dt is not None:
        config.trotter_dt = args.trotter_dt
    config.trotter_order = args.trotter_order

    generate_regression_dataset(
        output_path=args.output,
        config=config,
        exact_conditional_path=args.exact_conditional,
        n_workers=args.n_workers,
    )
    print(f"[done] regression dataset -> {args.output}")


if __name__ == "__main__":
    main()
