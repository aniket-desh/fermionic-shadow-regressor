"""Exact degree-two Majorana expectations for small statevectors."""
from __future__ import annotations

import numpy as np


def broadened_hf_state(n_qubits, n_electrons):
    """Repository-standard 80% HF + 20% first-four singles initial state."""
    hf = sum(1 << (n_qubits - 1 - i) for i in range(n_electrons))
    state = np.zeros(1 << n_qubits, dtype=np.complex128)
    excitations = []
    for occupied in range(n_electrons):
        for virtual in range(n_electrons, n_qubits):
            excitations.append(hf ^ (1 << (n_qubits - 1 - occupied))
                               ^ (1 << (n_qubits - 1 - virtual)))
            if len(excitations) == 4: break
        if len(excitations) == 4: break
    state[hf] = np.sqrt(0.8) if excitations else 1.0
    if excitations:
        state[excitations] = np.sqrt(0.2 / len(excitations))
    return state


def _majorana_action(index, n_qubits):
    q, y = divmod(index, 2)
    basis = np.arange(1 << n_qubits, dtype=np.int64)
    bit = (basis >> (n_qubits - 1 - q)) & 1
    prefix = np.zeros_like(basis)
    for j in range(q):
        prefix ^= (basis >> (n_qubits - 1 - j)) & 1
    phase = (1 - 2 * prefix).astype(np.complex128)
    if y: phase *= 1j * (1 - 2 * bit)
    return basis ^ (1 << (n_qubits - 1 - q)), phase


def exact_majorana_channels(states, keys, n_qubits):
    """Return (T,K) expectations of -i gamma_p gamma_q for states (2**n,T)."""
    actions = [_majorana_action(i, n_qubits) for i in range(2 * n_qubits)]
    out = np.empty((states.shape[1], len(keys)), dtype=np.float64)
    for k, (p, q) in enumerate(keys):
        dst_q, ph_q = actions[q]; dst_p, ph_p = actions[p]
        final, phase = dst_p[dst_q], ph_q * ph_p[dst_q]
        applied = np.empty_like(states)
        applied[final] = phase[:, None] * states
        out[:, k] = np.real(-1j * np.sum(np.conj(states) * applied, axis=0))
    return out


def known_initial_channels(n_qubits, n_electrons, keys):
    state = broadened_hf_state(n_qubits, n_electrons)[:, None]
    return exact_majorana_channels(state, keys, n_qubits)[0]
