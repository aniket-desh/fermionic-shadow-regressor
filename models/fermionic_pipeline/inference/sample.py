"""
Autoregressive sampling of synthetic fermionic shadow data at unseen geometries.

For a new geometry x*:
  1. Sample fresh Q ~ Uniform(S_{2n}) classically
  2. Sample b ~ p_{theta,phi}(. | Q, x*, t_k) autoregressively from trained model
  3. Form synthetic shadow snapshots (Q, b)
"""

import numpy as np
import torch


def sample_permutation_matrices(n_modes, n_samples, rng=None):
    """Sample random permutation matrices and return as flattened vectors.

    Args:
        n_modes: 2n (number of Majorana modes)
        n_samples: number of permutations to sample
        rng: numpy random Generator

    Returns:
        perms: (n_samples, 2n) int array — permutation indices
        Q_flat: (n_samples, (2n)^2) float32 array — flattened permutation matrices
    """
    if rng is None:
        rng = np.random.default_rng()

    perms = np.zeros((n_samples, n_modes), dtype=np.int64)
    Q_flat = np.zeros((n_samples, n_modes * n_modes), dtype=np.float32)

    for i in range(n_samples):
        p = rng.permutation(n_modes)
        perms[i] = p
        Q = np.zeros((n_modes, n_modes), dtype=np.float32)
        Q[np.arange(n_modes), p] = 1.0
        Q_flat[i] = Q.ravel()

    return perms, Q_flat


def generate_synthetic_shadows(
    model,
    x_star,
    times,
    n_shadows_per_time,
    n_qubits,
    batch_size=1000,
    rng=None,
    device=None,
):
    """Generate synthetic fermionic shadow data at an unseen geometry.

    Args:
        model: trained FiLMConditionalTransformer
        x_star: Hamiltonian parameter(s) — scalar or array
        times: array of time values
        n_shadows_per_time: number of shadow snapshots per time point
        n_qubits: number of qubits
        batch_size: max batch size for generation
        rng: numpy random Generator
        device: torch device

    Returns:
        synthetic_outcomes: dict mapping t -> list of [permutation, bitstring]
    """
    if rng is None:
        rng = np.random.default_rng()
    if device is None:
        device = next(model.parameters()).device

    n_modes = 2 * n_qubits
    x_val = np.atleast_1d(np.asarray(x_star, dtype=np.float32))

    synthetic_outcomes = {}

    for t in times:
        # Sample fresh permutations classically
        perms, Q_flat = sample_permutation_matrices(n_modes, n_shadows_per_time, rng)
        Q_flat_tensor = torch.tensor(Q_flat, dtype=torch.float32)

        # Build (x, t) conditioning for all samples
        xt = np.zeros((n_shadows_per_time, len(x_val) + 1), dtype=np.float32)
        xt[:, : len(x_val)] = x_val
        xt[:, -1] = t
        xt_tensor = torch.tensor(xt, dtype=torch.float32)

        # Autoregressively sample bitstrings
        bitstrings = model.sample(
            Q_flat_tensor,
            xt_tensor,
            n_qubits,
            batch_size=batch_size,
            print_progress=False,
        )

        # Package as (permutation, bitstring) pairs
        snapshots = []
        for i in range(len(bitstrings)):
            snapshots.append([perms[i].tolist(), bitstrings[i].tolist()])

        synthetic_outcomes[t] = snapshots

    return synthetic_outcomes


def generate_synthetic_shadows_batch(
    model,
    x_values,
    times,
    n_shadows_per_time,
    n_qubits,
    batch_size=1000,
    rng=None,
):
    """Generate synthetic shadows for multiple geometries.

    Returns:
        all_outcomes: dict mapping (x, t) -> list of [permutation, bitstring]
    """
    all_outcomes = {}

    for x in x_values:
        outcomes = generate_synthetic_shadows(
            model,
            x,
            times,
            n_shadows_per_time,
            n_qubits,
            batch_size=batch_size,
            rng=rng,
        )
        for t, snapshots in outcomes.items():
            x_scalar = float(x) if np.isscalar(x) else tuple(x)
            all_outcomes[(x_scalar, t)] = snapshots

    return all_outcomes
