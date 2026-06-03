"""
Conditioning modules for the fermionic shadow generative model.

Components:
  - PermutationEmbedding: h_psi(Q) — embeds the matchgate permutation matrix (flat MLP)
  - StructuredPermutationEmbedding: per-element (i, π(i)) embedding for cross-attention
  - HamiltonianTimeEmbedding: g_phi(x, t) — embeds Hamiltonian parameters + time
  - FiLMConditioner: combines h_psi(Q) and g_phi(x,t) via FiLM modulation
"""

import torch
import torch.nn as nn


class PermutationEmbedding(nn.Module):
    """h_psi: vec(Q) -> R^d_model via 2-layer MLP.

    Q in B(2n) is a 2n x 2n signed permutation matrix.
    Input is the flattened matrix vec(Q) in R^{(2n)^2}.
    """

    def __init__(self, input_dim, hidden_dim=128, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, Q_flat):
        # Q_flat: (batch, (2n)^2)
        return self.net(Q_flat)


class StructuredPermutationEmbedding(nn.Module):
    """Per-element embedding of Q for cross-attention.

    Q is a permutation matrix: row i has a single nonzero entry at
    column π(i). Instead of flattening to (2n)^2, we embed each
    (i, π(i)) pair independently and return a sequence of 2n embeddings.

    This produces (batch, 2n, d_model) — one embedding per Majorana mode —
    which serves as key/value for cross-attention with the bitstring tokens.
    Bit b_j's distribution depends on what Q does to the modes near qubit j,
    and cross-attention lets the model learn this position-specific dependence.
    """

    def __init__(self, n_modes, d_model=128):
        super().__init__()
        self.n_modes = n_modes
        # Learned embeddings for source index i and target index π(i)
        self.src_embed = nn.Embedding(n_modes, d_model // 2)
        self.tgt_embed = nn.Embedding(n_modes, d_model // 2)
        # Project concatenated (src, tgt) to d_model
        self.project = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, Q_flat):
        """Extract permutation from Q_flat and embed per-element.

        Args:
            Q_flat: (batch, (2n)^2) flattened permutation matrix

        Returns:
            q_elements: (batch, 2n, d_model) per-element embeddings
        """
        batch_size = Q_flat.shape[0]
        device = Q_flat.device
        N = self.n_modes

        # Reconstruct permutation indices from Q_flat
        Q = Q_flat.view(batch_size, N, N)
        # π(i) = argmax of row i (works for both signed and unsigned)
        perm_indices = Q.abs().argmax(dim=2)  # (batch, 2n)

        # Source indices: 0, 1, ..., 2n-1 (same for all samples)
        src_idx = torch.arange(N, device=device).unsqueeze(0).expand(batch_size, -1)

        src_emb = self.src_embed(src_idx)          # (batch, 2n, d_model/2)
        tgt_emb = self.tgt_embed(perm_indices)     # (batch, 2n, d_model/2)

        combined = torch.cat([src_emb, tgt_emb], dim=-1)  # (batch, 2n, d_model)
        return self.project(combined)  # (batch, 2n, d_model)


class FourierTimeEmbedding(nn.Module):
    """g_phi: (x, t) -> R^d_model via Fourier features on t.

    Replaces raw [x, t] input with [x_normalized, sin(ω₁t), cos(ω₁t), ...]
    before passing through a 2-layer MLP. This gives the network explicit
    access to oscillatory basis functions that ReLU cannot construct.

    A ReLU MLP with H hidden units has at most H linear regions, so it can
    approximate at most H/2 cycles of a sinusoid. For t_max=500 and energy
    gaps up to ~1.5 Eₕ, the signal has ~115 cycles — far more than a
    128-unit MLP can represent. Fourier features solve this by providing
    pre-computed sin/cos at a bank of frequencies; the MLP only needs to
    learn how to weight them.
    """

    def __init__(
        self,
        param_dim=1,
        n_freq=64,
        max_freq=2.0,
        hidden_dim=128,
        d_model=128,
        param_range=(0.5, 3.0),
    ):
        super().__init__()
        self.param_dim = param_dim
        self.param_min = param_range[0]
        self.param_scale = param_range[1] - param_range[0]

        # Fixed frequency bank: linearly spaced in [0, max_freq]
        freqs = torch.linspace(0.0, max_freq, n_freq)
        self.register_buffer("freqs", freqs)

        input_dim = param_dim + 2 * n_freq
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, xt):
        # xt: (batch, param_dim + 1) — last column is t
        x = xt[:, : self.param_dim]  # (batch, param_dim)
        t = xt[:, self.param_dim :]  # (batch, 1)

        # Normalize Hamiltonian parameters to [0, 1]
        x_norm = (x - self.param_min) / max(self.param_scale, 1e-8)

        # Fourier features on t
        phases = t * self.freqs  # (batch, n_freq)
        fourier = torch.cat([x_norm, torch.sin(phases), torch.cos(phases)], dim=-1)
        return self.net(fourier)


class HamiltonianTimeEmbedding(nn.Module):
    """g_phi: (x, t) -> R^d_model via 2-layer MLP.

    input_dim = dim(x) + 1 (e.g. bond length + time = 2 for H chains).
    """

    def __init__(self, input_dim, hidden_dim=128, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, xt):
        # xt: (batch, input_dim)
        return self.net(xt)


class FiLMConditioner(nn.Module):
    """FiLM conditioning: c = gamma(h_Q) * g_xt + beta(h_Q).

    Combines permutation embedding and Hamiltonian/time embedding via
    feature-wise linear modulation. Initialized near identity (gamma -> 1,
    beta -> 0) so that early training behaves like additive conditioning.
    """

    def __init__(self, d_model):
        super().__init__()
        self.gamma = nn.Linear(d_model, d_model)
        self.beta = nn.Linear(d_model, d_model)
        # Initialize near identity: gamma(h_Q) ≈ 1, beta(h_Q) ≈ 0
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, h_Q, g_xt):
        # h_Q: (batch, d_model) from PermutationEmbedding
        # g_xt: (batch, d_model) from HamiltonianTimeEmbedding
        gamma = self.gamma(h_Q)
        beta = self.beta(h_Q)
        return gamma * g_xt + beta
