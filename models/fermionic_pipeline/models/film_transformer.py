"""
FiLM-conditioned autoregressive transformer for fermionic shadow generation.

Extends the existing ConditionalTransformer architecture with:
  - Binary token vocabulary (b_i in {0, 1}) instead of Pauli-6
  - FiLM conditioning that combines permutation Q and Hamiltonian params (x, t)
  - Per-sample Q conditioning (each sample in a batch has its own Q)
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.models.transformer_core.layers import DecoderLayer
from src.models.transformer_core.modules import (
    MultiHeadAttention,
    PositionwiseFeedForward,
    PositionalEncoding,
    Embeddings,
    LayerNorm,
    SublayerConnection,
)
from src.models.transformer_core.models import Decoder, Generator
from src.models.transformer_core.utils import make_std_mask, clones

from fermionic_pipeline.models.conditioning import (
    PermutationEmbedding,
    StructuredPermutationEmbedding,
    HamiltonianTimeEmbedding,
    FourierTimeEmbedding,
    FiLMConditioner,
)

# Binary tokens: pad=0, start=1, shift=2 so {0,1} -> {2,3}
PAD_TOKEN = 0
START_TOKEN = 1
TOKEN_SHIFT = 2
N_OUTCOMES = 2  # binary


def init_film_transformer(
    n_qubits,
    param_dim,
    d_model=128,
    n_layers=4,
    n_heads=4,
    d_ff=512,
    dropout=0.1,
    hidden_dim=128,
    inject_every_layer=False,
    time_embedding="fourier",
    n_freq=64,
    max_freq=2.0,
    param_range=(0.5, 3.0),
):
    """Construct a FiLMConditionalTransformer from hyperparameters.

    Args:
        n_qubits: number of qubits (sequence length = n_qubits)
        param_dim: dimension of Hamiltonian parameters x (e.g. 1 for bond length)
        d_model: transformer model dimension
        n_layers: number of decoder layers
        n_heads: number of attention heads
        d_ff: feedforward hidden dimension
        dropout: dropout rate
        hidden_dim: hidden dimension for conditioning MLPs
        inject_every_layer: if True, inject FiLM conditioning at every decoder layer
        time_embedding: "fourier" (default) or "mlp" (legacy ReLU)
        n_freq: number of Fourier frequencies (for time_embedding="fourier")
        max_freq: maximum frequency in Eₕ (for time_embedding="fourier")
        param_range: (min, max) for normalizing x (for time_embedding="fourier")
    """
    n_modes = 2 * n_qubits
    q_input_dim = n_modes * n_modes  # vec(Q) dimension

    # Conditioning networks
    perm_embed = PermutationEmbedding(
        input_dim=q_input_dim, hidden_dim=hidden_dim, d_model=d_model
    )
    if time_embedding == "fourier":
        ham_embed = FourierTimeEmbedding(
            param_dim=param_dim,
            n_freq=n_freq,
            max_freq=max_freq,
            hidden_dim=hidden_dim,
            d_model=d_model,
            param_range=param_range,
        )
    else:
        ham_embed = HamiltonianTimeEmbedding(
            input_dim=param_dim + 1,
            hidden_dim=hidden_dim,
            d_model=d_model,
        )
    film = FiLMConditioner(d_model)

    # Transformer backbone
    attn = MultiHeadAttention(n_heads, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    vocab_size = N_OUTCOMES + TOKEN_SHIFT  # {pad, start, 0+shift, 1+shift}

    model = FiLMConditionalTransformer(
        perm_embed=perm_embed,
        ham_embed=ham_embed,
        film=film,
        decoder=Decoder(
            DecoderLayer(d_model, deepcopy(attn), deepcopy(ff), dropout),
            n_layers=n_layers,
        ),
        tgt_embed=nn.Sequential(Embeddings(d_model, vocab_size), deepcopy(position)),
        generator=Generator(d_model, vocab_size),
        inject_every_layer=inject_every_layer,
    )

    # Xavier init
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Re-apply FiLM init (Xavier would have overwritten it)
    nn.init.zeros_(model.film.gamma.weight)
    nn.init.ones_(model.film.gamma.bias)
    nn.init.zeros_(model.film.beta.weight)
    nn.init.zeros_(model.film.beta.bias)

    return model


class FiLMConditionalTransformer(nn.Module):
    """Autoregressive transformer with FiLM conditioning on (Q, x, t).

    Each sample in a batch is conditioned on its own permutation Q and
    shared/per-sample Hamiltonian parameters (x, t). The FiLM module
    combines h_psi(Q) and g_phi(x, t) into a single conditioning vector
    that is added to the token embeddings.
    """

    def __init__(
        self,
        perm_embed,
        ham_embed,
        film,
        decoder,
        tgt_embed,
        generator,
        inject_every_layer=False,
    ):
        super().__init__()
        self.perm_embed = perm_embed
        self.ham_embed = ham_embed
        self.film = film
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.inject_every_layer = inject_every_layer

        self.pad_token = PAD_TOKEN
        self.start_token = START_TOKEN
        self.token_shift = TOKEN_SHIFT
        self.n_outcomes = N_OUTCOMES
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device
        return super().to(device)

    def _compute_conditioning(self, Q_flat, xt):
        """Compute FiLM conditioning vector from Q and (x, t)."""
        h_Q = self.perm_embed(Q_flat)  # (batch, d_model)
        g_xt = self.ham_embed(xt)  # (batch, d_model)
        c = self.film(h_Q, g_xt)  # (batch, d_model)
        return c

    def forward(self, tgt, tgt_mask, Q_flat, xt):
        """Forward pass with FiLM conditioning.

        Args:
            tgt: (batch, seq_len) token indices
            tgt_mask: (batch, 1, seq_len) causal mask
            Q_flat: (batch, (2n)^2) flattened permutation matrix
            xt: (batch, param_dim + 1) concatenated [x, t]
        """
        tgt_embed = self.tgt_embed(tgt)  # (batch, seq_len, d_model)
        c = self._compute_conditioning(Q_flat, xt)  # (batch, d_model)

        # Add conditioning to all positions (broadcast over seq_len)
        x = tgt_embed + c.unsqueeze(1)

        if self.inject_every_layer:
            # Inject conditioning residually at every decoder layer
            c_expanded = c.unsqueeze(1)  # (batch, 1, d_model)
            for layer in self.decoder.layers:
                x = layer(x, tgt_mask)
                x = x + c_expanded
            return self.decoder.norm(x)
        else:
            return self.decoder(x, tgt_mask)

    def enumerate_distribution(self, Q_flat, xt, n_qubits, max_enum_batch=4096):
        """Compute p_θ(b | Q, x, t) for ALL 2^n bitstrings. Differentiable.

        Uses tree enumeration: at step i, expand all 2^i prefixes by {0, 1}.
        Returns a (batch, 2^n) probability tensor with gradients.

        Args:
            Q_flat: (batch, (2n)^2) permutation matrices
            xt: (batch, param_dim+1) [x, t]
            n_qubits: sequence length
            max_enum_batch: max sequences per forward pass (memory limit)
        """
        B = Q_flat.shape[0]
        device = Q_flat.device

        # Start with log P = 0 for each sample, prefix = [START]
        log_probs = torch.zeros(B, 1, device=device)  # (B, 1) — one prefix per sample
        prefixes = torch.full((B, 1, 1), self.start_token, dtype=torch.long, device=device)
        # prefixes shape: (B, n_prefixes, seq_len)

        for step in range(n_qubits):
            n_prefix = prefixes.shape[1]
            seq_len = prefixes.shape[2]

            # Flatten (B, n_prefix) into one batch dimension
            flat_B = B * n_prefix
            flat_prefix = prefixes.reshape(flat_B, seq_len)
            flat_Q = Q_flat.unsqueeze(1).expand(-1, n_prefix, -1).reshape(flat_B, -1)
            flat_xt = xt.unsqueeze(1).expand(-1, n_prefix, -1).reshape(flat_B, -1)

            # Forward in chunks if needed
            if flat_B <= max_enum_batch:
                mask = make_std_mask(flat_prefix, pad=self.pad_token)
                out = self.forward(flat_prefix, mask, flat_Q, flat_xt)
                logits = self.generator(out[:, -1])  # (flat_B, vocab)
            else:
                logits_list = []
                for start in range(0, flat_B, max_enum_batch):
                    end = min(start + max_enum_batch, flat_B)
                    chunk_p = flat_prefix[start:end]
                    chunk_Q = flat_Q[start:end]
                    chunk_xt = flat_xt[start:end]
                    mask = make_std_mask(chunk_p, pad=self.pad_token)
                    out = self.forward(chunk_p, mask, chunk_Q, chunk_xt)
                    logits_list.append(self.generator(out[:, -1]))
                logits = torch.cat(logits_list, dim=0)

            # Generator returns log-probs; extract and renormalize over valid bits only
            bit_log_probs = logits[:, self.token_shift : self.token_shift + self.n_outcomes]
            bit_probs = torch.softmax(bit_log_probs, dim=-1)  # (flat_B, 2)
            p0 = bit_probs[:, 0]
            p1 = bit_probs[:, 1]

            # Reshape to (B, n_prefix, 2)
            p0 = p0.reshape(B, n_prefix)
            p1 = p1.reshape(B, n_prefix)

            # Branch: each prefix spawns 2 children
            log_probs = log_probs.unsqueeze(-1).expand(-1, -1, 2)  # (B, n_prefix, 2)
            branch_log = torch.stack([torch.log(p0 + 1e-30), torch.log(p1 + 1e-30)], dim=-1)
            log_probs = (log_probs + branch_log).reshape(B, n_prefix * 2)

            # Extend prefixes with new tokens
            tok0 = torch.full((B, n_prefix, 1), self.token_shift + 0, dtype=torch.long, device=device)
            tok1 = torch.full((B, n_prefix, 1), self.token_shift + 1, dtype=torch.long, device=device)
            ext0 = torch.cat([prefixes, tok0], dim=2)
            ext1 = torch.cat([prefixes, tok1], dim=2)
            prefixes = torch.cat([ext0, ext1], dim=1)  # (B, 2*n_prefix, seq_len+1)

        # log_probs: (B, 2^n) — log P(b | Q, x, t) for each bitstring
        return log_probs.exp()

    @torch.no_grad()
    def sample_batch(self, batch_size, Q_flat, xt, n_qubits):
        """Autoregressively sample bitstrings conditioned on (Q, x, t).

        Args:
            batch_size: number of samples
            Q_flat: (batch_size, (2n)^2) per-sample permutation matrices
            xt: (batch_size, param_dim + 1) per-sample [x, t]
            n_qubits: sequence length
        """
        self.eval()
        Q_flat = Q_flat.to(self.device)
        xt = xt.to(self.device)

        tgt = torch.full(
            (batch_size, 1), self.start_token, dtype=torch.long, device=self.device
        )

        for i in range(n_qubits):
            mask = make_std_mask(tgt, pad=self.pad_token)
            out = self.forward(tgt, mask, Q_flat, xt)
            log_probs = self.generator(out[:, -1])
            probs = torch.exp(log_probs).detach()
            probs = probs / probs.sum(dim=-1, keepdim=True)

            next_token = torch.multinomial(probs, 1, replacement=True)
            tgt = torch.cat([tgt, next_token], dim=1)

        # Remove start token and shift back to {0, 1}
        samples = tgt[:, 1:].cpu().numpy() - self.token_shift

        # Filter invalid samples
        valid = np.all((samples >= 0) & (samples < self.n_outcomes), axis=1)
        return samples[valid]

    def sample(
        self, Q_flat_all, xt_all, n_qubits, batch_size=1000, print_progress=True
    ):
        """Generate samples for multiple (Q, x, t) conditioning inputs.

        Args:
            Q_flat_all: (N, (2n)^2) permutation matrices for all samples
            xt_all: (N, param_dim + 1) [x, t] for all samples
            n_qubits: number of qubits
            batch_size: max samples per batch
        """
        N = Q_flat_all.shape[0]
        all_samples = []

        indices = list(range(0, N, batch_size))
        if print_progress:
            indices = tqdm(indices, desc=f"Generating {N} samples")

        for start in indices:
            end = min(start + batch_size, N)
            Q_batch = Q_flat_all[start:end]
            xt_batch = xt_all[start:end]
            bs = end - start

            samples = self.sample_batch(bs, Q_batch, xt_batch, n_qubits)
            all_samples.append(samples)

        return np.concatenate(all_samples, axis=0)


# ── Cross-attention architecture ──────────────────────────────────────


class CrossAttentionDecoderLayer(nn.Module):
    """Decoder layer with cross-attention to Q elements.

    Architecture per layer:
      1. Causal self-attention over token sequence
      2. Cross-attention: tokens (query) attend to Q elements (key/value)
      3. Feedforward network

    This replaces the FiLM broadcast with position-specific Q information:
    bit b_i can attend to the Q elements most relevant to qubit i.
    """

    def __init__(self, d_model, self_attn, cross_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)

    def forward(self, x, tgt_mask, q_elements):
        """
        Args:
            x: (batch, seq_len, d_model) token embeddings
            tgt_mask: (batch, 1, seq_len) causal mask
            q_elements: (batch, 2n, d_model) per-element Q embeddings
        """
        # 1. Causal self-attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 2. Cross-attention: tokens attend to Q elements (no mask)
        x = self.sublayer[1](x, lambda x: self.cross_attn(x, q_elements, q_elements))
        # 3. Feedforward
        return self.sublayer[2](x, self.feed_forward)


class CrossAttentionDecoder(nn.Module):
    """Stack of CrossAttentionDecoderLayers with final layer norm."""

    def __init__(self, layer, n_layers):
        super().__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.sublayer[0].norm_layer.a_2.shape[0])

    def forward(self, x, tgt_mask, q_elements):
        for layer in self.layers:
            x = layer(x, tgt_mask, q_elements)
        return self.norm(x)


def init_crossattn_transformer(
    n_qubits,
    param_dim,
    d_model=128,
    n_layers=4,
    n_heads=4,
    d_ff=512,
    dropout=0.1,
    hidden_dim=128,
    time_embedding="fourier",
    n_freq=64,
    max_freq=2.0,
    param_range=(0.5, 3.0),
):
    """Construct a CrossAttentionTransformer.

    Uses structured per-element Q embedding + cross-attention instead of
    FiLM broadcast. Hamiltonian params (x,t) are still added globally.

    Args:
        n_qubits: number of qubits (sequence length = n_qubits)
        param_dim: dimension of Hamiltonian parameters x
        d_model: transformer model dimension
        n_layers: number of decoder layers
        n_heads: number of attention heads
        d_ff: feedforward hidden dimension
        dropout: dropout rate
        hidden_dim: hidden dimension for conditioning MLPs
        time_embedding: "fourier" (default) or "mlp" (legacy ReLU)
        n_freq: number of Fourier frequencies
        max_freq: maximum frequency in Eₕ
        param_range: (min, max) for normalizing x
    """
    n_modes = 2 * n_qubits

    # Q embedding: per-element (i, π(i)) → (batch, 2n, d_model)
    q_embed = StructuredPermutationEmbedding(n_modes, d_model)

    # (x, t) embedding: global conditioning
    if time_embedding == "fourier":
        ham_embed = FourierTimeEmbedding(
            param_dim=param_dim,
            n_freq=n_freq,
            max_freq=max_freq,
            hidden_dim=hidden_dim,
            d_model=d_model,
            param_range=param_range,
        )
    else:
        ham_embed = HamiltonianTimeEmbedding(
            input_dim=param_dim + 1, hidden_dim=hidden_dim, d_model=d_model
        )

    # Decoder with cross-attention
    self_attn = MultiHeadAttention(n_heads, d_model, dropout)
    cross_attn = MultiHeadAttention(n_heads, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    vocab_size = N_OUTCOMES + TOKEN_SHIFT

    model = CrossAttentionTransformer(
        q_embed=q_embed,
        ham_embed=ham_embed,
        decoder=CrossAttentionDecoder(
            CrossAttentionDecoderLayer(
                d_model,
                deepcopy(self_attn),
                deepcopy(cross_attn),
                deepcopy(ff),
                dropout,
            ),
            n_layers=n_layers,
        ),
        tgt_embed=nn.Sequential(Embeddings(d_model, vocab_size), deepcopy(position)),
        generator=Generator(d_model, vocab_size),
    )

    # Xavier init
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


class CrossAttentionTransformer(nn.Module):
    """Autoregressive transformer with cross-attention to Q elements.

    Instead of compressing Q into a single vector (FiLM), this model
    embeds each element of Q separately and lets each token position
    attend to the relevant Q elements via cross-attention.

    Hamiltonian parameters (x, t) are still added globally to token
    embeddings (same as FiLM approach for the non-Q conditioning).
    """

    def __init__(self, q_embed, ham_embed, decoder, tgt_embed, generator):
        super().__init__()
        self.q_embed = q_embed
        self.ham_embed = ham_embed
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator

        self.pad_token = PAD_TOKEN
        self.start_token = START_TOKEN
        self.token_shift = TOKEN_SHIFT
        self.n_outcomes = N_OUTCOMES
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(self, tgt, tgt_mask, Q_flat, xt):
        """Forward pass with cross-attention Q conditioning.

        Args:
            tgt: (batch, seq_len) token indices
            tgt_mask: (batch, 1, seq_len) causal mask
            Q_flat: (batch, (2n)^2) flattened permutation matrix
            xt: (batch, param_dim + 1) concatenated [x, t]
        """
        tgt_embed = self.tgt_embed(tgt)  # (batch, seq_len, d_model)

        # Per-element Q embeddings for cross-attention
        q_elements = self.q_embed(Q_flat)  # (batch, 2n, d_model)

        # Global (x, t) conditioning added to all token positions
        g_xt = self.ham_embed(xt)  # (batch, d_model)
        x = tgt_embed + g_xt.unsqueeze(1)

        # Decode with cross-attention to Q elements
        x = self.decoder(x, tgt_mask, q_elements)
        return x

    def enumerate_distribution(self, Q_flat, xt, n_qubits, max_enum_batch=4096):
        """Compute p_θ(b | Q, x, t) for ALL 2^n bitstrings. Differentiable.

        Same as FiLMConditionalTransformer.enumerate_distribution but uses
        the cross-attention forward pass.
        """
        B = Q_flat.shape[0]
        device = Q_flat.device

        log_probs = torch.zeros(B, 1, device=device)
        prefixes = torch.full((B, 1, 1), self.start_token, dtype=torch.long, device=device)

        for step in range(n_qubits):
            n_prefix = prefixes.shape[1]
            seq_len = prefixes.shape[2]

            flat_B = B * n_prefix
            flat_prefix = prefixes.reshape(flat_B, seq_len)
            flat_Q = Q_flat.unsqueeze(1).expand(-1, n_prefix, -1).reshape(flat_B, -1)
            flat_xt = xt.unsqueeze(1).expand(-1, n_prefix, -1).reshape(flat_B, -1)

            if flat_B <= max_enum_batch:
                mask = make_std_mask(flat_prefix, pad=self.pad_token)
                out = self.forward(flat_prefix, mask, flat_Q, flat_xt)
                logits = self.generator(out[:, -1])
            else:
                logits_list = []
                for start in range(0, flat_B, max_enum_batch):
                    end = min(start + max_enum_batch, flat_B)
                    mask = make_std_mask(flat_prefix[start:end], pad=self.pad_token)
                    out = self.forward(flat_prefix[start:end], mask, flat_Q[start:end], flat_xt[start:end])
                    logits_list.append(self.generator(out[:, -1]))
                logits = torch.cat(logits_list, dim=0)

            bit_log_probs = logits[:, self.token_shift : self.token_shift + self.n_outcomes]
            bit_probs = torch.softmax(bit_log_probs, dim=-1)
            p0 = bit_probs[:, 0].reshape(B, n_prefix)
            p1 = bit_probs[:, 1].reshape(B, n_prefix)

            log_probs = log_probs.unsqueeze(-1).expand(-1, -1, 2)
            branch_log = torch.stack([torch.log(p0 + 1e-30), torch.log(p1 + 1e-30)], dim=-1)
            log_probs = (log_probs + branch_log).reshape(B, n_prefix * 2)

            tok0 = torch.full((B, n_prefix, 1), self.token_shift + 0, dtype=torch.long, device=device)
            tok1 = torch.full((B, n_prefix, 1), self.token_shift + 1, dtype=torch.long, device=device)
            prefixes = torch.cat([
                torch.cat([prefixes, tok0], dim=2),
                torch.cat([prefixes, tok1], dim=2),
            ], dim=1)

        return log_probs.exp()

    @torch.no_grad()
    def sample_batch(self, batch_size, Q_flat, xt, n_qubits):
        """Autoregressively sample bitstrings conditioned on (Q, x, t)."""
        self.eval()
        Q_flat = Q_flat.to(self.device)
        xt = xt.to(self.device)

        tgt = torch.full(
            (batch_size, 1), self.start_token, dtype=torch.long, device=self.device
        )

        # Pre-compute Q elements (constant across autoregressive steps)
        q_elements = self.q_embed(Q_flat)
        g_xt = self.ham_embed(xt)

        for i in range(n_qubits):
            mask = make_std_mask(tgt, pad=self.pad_token)
            tgt_embed = self.tgt_embed(tgt)
            x = tgt_embed + g_xt.unsqueeze(1)
            x = self.decoder(x, mask, q_elements)

            log_probs = self.generator(x[:, -1])
            probs = torch.exp(log_probs).detach()
            probs = probs / probs.sum(dim=-1, keepdim=True)

            next_token = torch.multinomial(probs, 1, replacement=True)
            tgt = torch.cat([tgt, next_token], dim=1)

        samples = tgt[:, 1:].cpu().numpy() - self.token_shift
        valid = np.all((samples >= 0) & (samples < self.n_outcomes), axis=1)
        return samples[valid]

    def sample(
        self, Q_flat_all, xt_all, n_qubits, batch_size=1000, print_progress=True
    ):
        """Generate samples for multiple (Q, x, t) conditioning inputs."""
        N = Q_flat_all.shape[0]
        all_samples = []

        indices = list(range(0, N, batch_size))
        if print_progress:
            indices = tqdm(indices, desc=f"Generating {N} samples")

        for start in indices:
            end = min(start + batch_size, N)
            Q_batch = Q_flat_all[start:end]
            xt_batch = xt_all[start:end]
            bs = end - start

            samples = self.sample_batch(bs, Q_batch, xt_batch, n_qubits)
            all_samples.append(samples)

        return np.concatenate(all_samples, axis=0)
