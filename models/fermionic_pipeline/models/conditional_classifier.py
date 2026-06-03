"""
Full conditional classifier for p(b | Q, R, t).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from fermionic_pipeline.data.exact_conditional_dataset import bitstrings_from_class_indices


@dataclass
class ConditionalClassifierConfig:
    n_qubits: int = 8
    d_model: int = 128
    hidden_dim: int = 512
    observable_head: bool = False
    n_observables: int = 16

    @property
    def n_modes(self) -> int:
        return 2 * self.n_qubits

    @property
    def q_input_dim(self) -> int:
        return 4 * self.n_qubits

    @property
    def n_classes(self) -> int:
        return 1 << self.n_qubits

    def to_dict(self):
        return asdict(self)


class _EmbeddingMLP(nn.Module):
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ConditionalClassifier(nn.Module):
    def __init__(self, config: ConditionalClassifierConfig):
        super().__init__()
        self.config = config
        self.q_embed = _EmbeddingMLP(config.q_input_dim, config.d_model)
        self.rt_embed = _EmbeddingMLP(2, config.d_model)

        self.trunk_fc1 = nn.Linear(2 * config.d_model, config.hidden_dim)
        self.trunk_fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_head = nn.Linear(config.hidden_dim, config.n_classes)
        self.observable_head = (
            nn.Linear(config.hidden_dim, config.n_observables)
            if config.observable_head
            else None
        )
        self.activation = nn.ReLU()

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, q_features, rt):
        q_hidden = self.q_embed(q_features)
        rt_hidden = self.rt_embed(rt)
        hidden = torch.cat([q_hidden, rt_hidden], dim=-1)
        hidden = self.activation(self.trunk_fc1(hidden))
        hidden = self.activation(self.trunk_fc2(hidden))
        return hidden

    def forward(self, q_features, rt):
        hidden = self.encode(q_features, rt)
        logits = self.out_head(hidden)
        obs = self.observable_head(hidden) if self.observable_head is not None else None
        return logits, obs

    @torch.no_grad()
    def predict_distribution(self, q_features, rt):
        logits, obs = self.forward(q_features, rt)
        probs = torch.softmax(logits, dim=-1)
        return probs, obs

    @torch.no_grad()
    def sample_bitstrings(self, q_features, rt, n_samples, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        if q_features.ndim == 1:
            q_features = q_features[None, :]
        if rt.ndim == 1:
            rt = rt[None, :]

        if not torch.is_tensor(q_features):
            q_features = torch.as_tensor(q_features, dtype=torch.float32, device=next(self.parameters()).device)
        if not torch.is_tensor(rt):
            rt = torch.as_tensor(rt, dtype=torch.float32, device=next(self.parameters()).device)

        probs, _ = self.predict_distribution(q_features, rt)
        probs_np = probs[0].detach().cpu().numpy()
        class_indices = rng.choice(self.config.n_classes, size=n_samples, p=probs_np)
        return bitstrings_from_class_indices(class_indices, self.config.n_qubits)


def init_conditional_classifier(
    n_qubits: int = 8,
    d_model: int = 128,
    hidden_dim: int = 512,
    observable_head: bool = False,
    n_observables: int = 16,
):
    return ConditionalClassifier(
        ConditionalClassifierConfig(
            n_qubits=n_qubits,
            d_model=d_model,
            hidden_dim=hidden_dim,
            observable_head=observable_head,
            n_observables=n_observables,
        )
    )
