"""
Exact conditional dataset for fixed signed-permutation libraries.

This module builds a reusable library of signed permutations Q in B(2n),
computes exact conditional probabilities p(b | Q, R, t), and stores the
results in an HDF5 file that can be streamed during training.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

try:
    import h5py
except ModuleNotFoundError:
    h5py = None

from fermionic_pipeline.data.generate_shadows import (
    _apply_1q,
    _apply_matchgate,
    build_hydrogen_chain_hamiltonian,
    prepare_initial_state,
    time_evolve,
)


_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _require_h5py():
    if h5py is None:
        raise ImportError("h5py is required for exact_conditional_dataset storage operations.")


@dataclass
class ExactConditionalConfig:
    n_atoms: int = 4
    r_start: float = 0.5
    r_end: float = 3.0
    r_step: float = 0.5
    t_max: float = 10.0
    n_times: int = 10
    n_q: int = 100
    seed: int = 42
    storage_dtype: str = "float32"

    @property
    def R_values(self) -> np.ndarray:
        return np.round(
            np.arange(self.r_start, self.r_end + self.r_step / 2, self.r_step),
            8,
        )

    @property
    def times(self) -> np.ndarray:
        return np.linspace(0.0, self.t_max, self.n_times, dtype=np.float64)


def sample_signed_permutation_library(n_modes: int, n_q: int, seed: int = 42):
    """Sample a reproducible library of signed permutations in B(2n)."""
    rng = np.random.default_rng(seed)
    permutations = np.zeros((n_q, n_modes), dtype=np.int16)
    signs = np.zeros((n_q, n_modes), dtype=np.int8)

    for i in range(n_q):
        permutations[i] = rng.permutation(n_modes)
        signs[i] = rng.choice(np.array([-1, 1], dtype=np.int8), size=n_modes)

    return permutations, signs


def signed_permutation_to_matrix(permutation, signs):
    """Convert (permutation, signs) into the orthogonal matrix Q.

    The convention matches the existing unsigned code path:
    column j maps to signs[j] * e_{permutation[j]}.
    """
    permutation = np.asarray(permutation, dtype=np.int64)
    signs = np.asarray(signs, dtype=np.float64)
    n_modes = permutation.shape[0]
    Q = np.zeros((n_modes, n_modes), dtype=np.float64)
    Q[permutation, np.arange(n_modes)] = signs
    return Q


def encode_q_features(permutations, signs):
    """Encode (pi, s) into the flattened 4n feature vector used by the classifier."""
    permutations = np.asarray(permutations, dtype=np.float32)
    signs = np.asarray(signs, dtype=np.float32)
    n_modes = permutations.shape[1]
    denom = max(n_modes - 1, 1)
    perm_scaled = 2.0 * (permutations / denom) - 1.0
    return np.concatenate([perm_scaled, signs], axis=1).astype(np.float32)


def bitstrings_from_class_indices(indices: np.ndarray, n_qubits: int) -> np.ndarray:
    """Decode integer basis indices into bitstrings using MSB-first ordering."""
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    shifts = np.arange(n_qubits - 1, -1, -1, dtype=np.int64)
    return ((indices[:, None] >> shifts[None, :]) & 1).astype(np.int8)


def _apply_single_majorana(statevector: np.ndarray, index: int, n_qubits: int) -> np.ndarray:
    """Apply gamma_index to a statevector using Jordan-Wigner conventions."""
    qubit = index // 2
    out = statevector.astype(np.complex128, copy=True)
    for q in range(qubit):
        out = _apply_1q(out, _Z, q, n_qubits)
    out = _apply_1q(out, _X if index % 2 == 0 else _Y, qubit, n_qubits)
    return out


def exact_single_majorana_expectations(statevector: np.ndarray, n_qubits: int) -> np.ndarray:
    """Compute exact <gamma_j> for j = 0, ..., 2n-1."""
    n_modes = 2 * n_qubits
    expectations = np.zeros(n_modes, dtype=np.float64)
    for j in range(n_modes):
        ket = _apply_single_majorana(statevector, j, n_qubits)
        expectations[j] = np.real(np.vdot(statevector, ket))
    return expectations


def _save_q_library(path: str, permutations: np.ndarray, signs: np.ndarray) -> None:
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    np.savez(path, permutations=permutations, signs=signs)


class ExactConditionalDatasetHandle:
    """Small metadata wrapper around the HDF5-backed exact conditional dataset."""

    def __init__(self, path: str):
        _require_h5py()
        self.path = path
        with h5py.File(path, "r") as f:
            self.n_atoms = int(f.attrs["n_atoms"])
            self.n_qubits = int(f.attrs["n_qubits"])
            self.n_modes = int(f.attrs["n_modes"])
            self.n_q = int(f.attrs["n_q"])
            self.n_bitstrings = int(f.attrs["n_bitstrings"])
            self.storage_dtype = str(f["probabilities"].dtype)
            self.R_values = f["R_values"][:]
            self.times = f["times"][:]
            self.permutations = f["permutations"][:]
            self.signs = f["signs"][:]
            self.q_features = f["q_features"][:]
            self.eigvals = f["eigvals"][:]

    def get_probability(self, r_idx: int, t_idx: int, q_idx: int) -> np.ndarray:
        with h5py.File(self.path, "r") as f:
            return f["probabilities"][r_idx, t_idx, q_idx].astype(np.float64)

    def get_single_majorana_targets(self, r_idx: int, t_idx: int) -> np.ndarray:
        with h5py.File(self.path, "r") as f:
            return f["single_majorana_expectations"][r_idx, t_idx].astype(np.float64)

class ExactConditionalTorchDataset(Dataset):
    """Torch dataset that lazily streams exact conditionals from HDF5."""

    def __init__(self, path: str, r_indices: Optional[Sequence[int]] = None):
        _require_h5py()
        self.path = path
        self.handle = ExactConditionalDatasetHandle(path)
        if r_indices is None:
            r_indices = list(range(len(self.handle.R_values)))
        self.r_indices = np.array(sorted(r_indices), dtype=np.int64)
        self.n_t = len(self.handle.times)
        self.n_q = self.handle.n_q
        self._file = None

    def __len__(self):
        return len(self.r_indices) * self.n_t * self.n_q

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def _unravel_index(self, idx: int):
        q_idx = idx % self.n_q
        idx //= self.n_q
        t_idx = idx % self.n_t
        idx //= self.n_t
        r_local = idx
        r_idx = int(self.r_indices[r_local])
        return r_idx, t_idx, q_idx

    def __getitem__(self, idx: int):
        f = self._ensure_open()
        r_idx, t_idx, q_idx = self._unravel_index(idx)
        q_feat = f["q_features"][q_idx].astype(np.float32)
        rt = np.array(
            [f["R_values"][r_idx], f["times"][t_idx]],
            dtype=np.float32,
        )
        probs = f["probabilities"][r_idx, t_idx, q_idx].astype(np.float32)
        obs = f["single_majorana_expectations"][r_idx, t_idx].astype(np.float32)
        ids = np.array([r_idx, t_idx, q_idx], dtype=np.int64)
        return (
            torch.from_numpy(q_feat),
            torch.from_numpy(rt),
            torch.from_numpy(probs),
            torch.from_numpy(obs),
            torch.from_numpy(ids),
        )


def split_r_indices(
    n_r: int,
    test_fraction: float = 0.2,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(test_fraction * n_r)))
    test_idx = np.sort(rng.choice(n_r, size=n_test, replace=False))
    train_idx = np.array(sorted(set(range(n_r)) - set(test_idx.tolist())), dtype=np.int64)
    return train_idx, test_idx


def generate_exact_conditional_dataset(
    output_path: str,
    config: ExactConditionalConfig,
    q_library_path: Optional[str] = None,
) -> str:
    """Generate the exact conditional dataset and stream it to HDF5."""
    _require_h5py()
    R_values = np.asarray(config.R_values, dtype=np.float64)
    times = np.asarray(config.times, dtype=np.float64)

    H_probe, n_qubits = build_hydrogen_chain_hamiltonian(config.n_atoms, float(R_values[0]))
    n_modes = 2 * n_qubits
    n_bitstrings = 1 << n_qubits
    probabilities_dtype = np.float64 if config.storage_dtype == "float64" else np.float32

    permutations, signs = sample_signed_permutation_library(n_modes, config.n_q, seed=config.seed)
    q_features = encode_q_features(permutations, signs)
    q_mats = [
        signed_permutation_to_matrix(permutations[i], signs[i]).astype(np.float64)
        for i in range(config.n_q)
    ]

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if q_library_path is None:
        base, _ = os.path.splitext(output_path)
        q_library_path = f"{base}_q_library.npz"
    _save_q_library(q_library_path, permutations, signs)

    with h5py.File(output_path, "w") as f:
        f.attrs["n_atoms"] = config.n_atoms
        f.attrs["n_qubits"] = n_qubits
        f.attrs["n_modes"] = n_modes
        f.attrs["n_q"] = config.n_q
        f.attrs["n_bitstrings"] = n_bitstrings
        f.attrs["seed"] = config.seed
        f.attrs["storage_dtype"] = config.storage_dtype
        f.attrs["config_json"] = json.dumps(asdict(config))

        f.create_dataset("R_values", data=R_values)
        f.create_dataset("times", data=times)
        f.create_dataset("permutations", data=permutations)
        f.create_dataset("signs", data=signs)
        f.create_dataset("q_features", data=q_features)
        probs_ds = f.create_dataset(
            "probabilities",
            shape=(len(R_values), len(times), config.n_q, n_bitstrings),
            dtype=probabilities_dtype,
            chunks=(1, 1, 1, n_bitstrings),
            compression="lzf",
        )
        obs_ds = f.create_dataset(
            "single_majorana_expectations",
            shape=(len(R_values), len(times), n_modes),
            dtype=np.float64,
            chunks=(1, 1, n_modes),
            compression="lzf",
        )
        eigvals_ds = f.create_dataset(
            "eigvals",
            shape=(len(R_values), n_bitstrings),
            dtype=np.float64,
            compression="lzf",
        )

        outer = tqdm(enumerate(R_values), total=len(R_values), desc="Exact dataset")
        for r_idx, R in outer:
            H_sparse, n_qubits_R = build_hydrogen_chain_hamiltonian(config.n_atoms, float(R))
            if n_qubits_R != n_qubits:
                raise ValueError("n_qubits changed across geometries; expected fixed active space.")
            H_dense = H_sparse.toarray().astype(np.complex128)
            eigvals_ds[r_idx] = np.sort(np.linalg.eigvalsh(H_dense).real)

            psi_0, _ = prepare_initial_state(H_sparse, n_qubits, n_electrons=config.n_atoms)
            states = time_evolve(H_sparse, psi_0, times)

            inner = tqdm(times, leave=False, desc=f"R={R:.2f}")
            for t_idx, t in enumerate(inner):
                psi_t = states[t].astype(np.complex128, copy=False)
                obs_ds[r_idx, t_idx] = exact_single_majorana_expectations(psi_t, n_qubits)

                for q_idx, Q in enumerate(q_mats):
                    rotated = _apply_matchgate(Q, psi_t, n_qubits).astype(np.complex128, copy=False)
                    probs = np.abs(rotated) ** 2
                    probs = probs / probs.sum()
                    probs_ds[r_idx, t_idx, q_idx] = probs.astype(probabilities_dtype)

    return output_path


def load_config_defaults(config_path: str) -> ExactConditionalConfig:
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    dc = cfg["data"]
    return ExactConditionalConfig(
        n_atoms=dc["n_atoms"],
        r_start=dc["r_start"],
        r_end=dc["r_end"],
        r_step=dc["r_step"],
        t_max=dc["t_max"],
        n_times=dc["n_times"],
    )


def main():
    parser = argparse.ArgumentParser(description="Build exact conditional HDF5 dataset")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--q_library_path", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--n_atoms", type=int, default=None)
    parser.add_argument("--r_start", type=float, default=None)
    parser.add_argument("--r_end", type=float, default=None)
    parser.add_argument("--r_step", type=float, default=None)
    parser.add_argument("--t_max", type=float, default=None)
    parser.add_argument("--n_times", type=int, default=None)
    parser.add_argument("--n_q", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--storage_dtype", type=str, default="float32", choices=["float32", "float64"])
    args = parser.parse_args()

    if args.config is not None:
        config = load_config_defaults(args.config)
    else:
        config = ExactConditionalConfig()

    for field in ["n_atoms", "r_start", "r_end", "r_step", "t_max", "n_times"]:
        value = getattr(args, field)
        if value is not None:
            setattr(config, field, value)
    config.n_q = args.n_q
    config.seed = args.seed
    config.storage_dtype = args.storage_dtype

    generate_exact_conditional_dataset(
        output_path=args.output,
        config=config,
        q_library_path=args.q_library_path,
    )
    print(f"[done] exact conditional dataset -> {args.output}")


if __name__ == "__main__":
    main()
