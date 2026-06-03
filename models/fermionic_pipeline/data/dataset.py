"""
PyTorch dataset for fermionic shadow data.

Each sample is a tuple (Q_flat, bitstring, x, t) where:
  - Q_flat: flattened (2n)^2 permutation matrix
  - bitstring: n-bit measurement outcome
  - x: Hamiltonian parameters (e.g. bond length)
  - t: time parameter
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from src.models.transformer_core.utils import make_std_mask

# Token constants (must match film_transformer.py)
PAD_TOKEN = 0
START_TOKEN = 1
TOKEN_SHIFT = 2
N_OUTCOMES = 2  # binary {0, 1}


class FermionicShadowDataset(Dataset):
    """Dataset of fermionic shadow snapshots for training.

    Stores compact signed permutation data (permutation indices + signs)
    and expands to Q_flat on-the-fly. Q is a signed permutation matrix
    from B(2n) with entries in {-1, 0, +1}.
    """

    def __init__(self, permutations, bitstrings, params, times, n_modes):
        """
        Args:
            permutations: (N, 2n) int array — permutation indices
            bitstrings: (N, n) int array — measurement outcomes
            params: (N, dim_x) float array — Hamiltonian parameters
            times: (N,) float array — time values
            n_modes: 2n (number of Majorana modes)
        """
        self.permutations = np.asarray(permutations, dtype=np.int64)
        self.bitstrings = np.asarray(bitstrings, dtype=np.int64)
        self.params = np.asarray(params, dtype=np.float32)
        if self.params.ndim == 1:
            self.params = self.params[:, None]
        self.times = np.asarray(times, dtype=np.float32)
        self.n_modes = n_modes

    def __len__(self):
        return len(self.bitstrings)

    def _perm_to_flat(self, perm):
        """Convert permutation index array to flattened permutation matrix."""
        Q = np.zeros((self.n_modes, self.n_modes), dtype=np.float32)
        Q[np.arange(self.n_modes), perm] = 1.0
        return Q.ravel()

    def __getitem__(self, idx):
        Q_flat = self._perm_to_flat(self.permutations[idx])
        bitstring = self.bitstrings[idx]
        xt = np.concatenate([self.params[idx], [self.times[idx]]])
        return Q_flat, bitstring, xt

    @classmethod
    def from_hdf5(cls, path, split="train"):
        """Load dataset from HDF5 file."""
        import h5py
        with h5py.File(path, "r") as f:
            grp = f[split]
            permutations = grp["permutations"][:]
            bitstrings = grp["bitstrings"][:]
            params = grp["params"][:]
            times = grp["times"][:]
            n_modes = int(grp.attrs["n_modes"])
        return cls(permutations, bitstrings, params, times, n_modes)

    @classmethod
    def from_outcomes_dict(cls, outcomes_dict, n_qubits):
        """Build dataset from outcomes dict.

        Each snapshot is [perm, bits, signs] (signed permutation) or
        [perm, bits] (legacy unsigned, signs default to +1).

        Args:
            outcomes_dict: {(x_val, t_val): [[perm, bits, signs?], ...]}
            n_qubits: number of qubits
        """
        n_modes = 2 * n_qubits
        all_perms = []
        all_bits = []
        all_signs = []
        all_params = []
        all_times = []

        for (x_val, t_val), snapshots in outcomes_dict.items():
            for perm, bits, *rest in snapshots:
                all_perms.append(perm)
                all_bits.append(bits)
                all_params.append([x_val] if np.isscalar(x_val) else list(x_val))
                all_times.append(t_val)

        return cls(
            permutations=np.array(all_perms),
            bitstrings=np.array(all_bits),
            params=np.array(all_params, dtype=np.float32),
            times=np.array(all_times, dtype=np.float32),
            n_modes=n_modes,
        )

    def save_hdf5(self, path, split="train"):
        """Save dataset to HDF5."""
        import h5py
        with h5py.File(path, "a") as f:
            grp = f.require_group(split)
            for key in ["permutations", "bitstrings", "params", "times"]:
                if key in grp:
                    del grp[key]
            grp.create_dataset("permutations", data=self.permutations)
            grp.create_dataset("bitstrings", data=self.bitstrings)
            grp.create_dataset("params", data=self.params)
            grp.create_dataset("times", data=self.times)
            grp.attrs["n_modes"] = self.n_modes


def fermionic_collate_fn(batch):
    """Collate function for DataLoader.

    Returns:
        Q_flat: (B, (2n)^2) float tensor
        tgt: (B, n+1) long tensor — [start, b1+shift, ..., bn+shift]
        tgt_y: (B, n) long tensor — [b1+shift, ..., bn+shift]
        tgt_mask: (B, 1, n) bool tensor
        xt: (B, param_dim + 1) float tensor
    """
    Q_flat_list, bits_list, xt_list = zip(*batch)

    Q_flat = torch.tensor(np.stack(Q_flat_list), dtype=torch.float32)
    xt = torch.tensor(np.stack(xt_list), dtype=torch.float32)

    # Build token sequence: [start, b1+shift, ..., bn+shift]
    bits = np.stack(bits_list).astype(np.int64) + TOKEN_SHIFT
    start_col = np.full((len(bits), 1), START_TOKEN, dtype=np.int64)
    tokens = np.concatenate([start_col, bits], axis=1)
    tokens = torch.from_numpy(tokens).long()

    tgt = tokens[:, :-1]
    tgt_y = tokens[:, 1:]
    tgt_mask = make_std_mask(tgt, PAD_TOKEN)

    return Q_flat, tgt, tgt_y, tgt_mask, xt
