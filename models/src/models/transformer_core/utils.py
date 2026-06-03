"""Shared transformer helpers (masking, layer cloning).

Single source of truth for utilities that were previously duplicated between
``src.models.transformer`` and ``src.models.gctransformer``. Both packages
re-export these names, so existing import paths keep working.

Note: tokenization constants (PAD/START/SHIFT) are NOT here — they are
model-specific and live in ``src.models.transformer.utils`` (the generative
transformer) and ``src.models.gctransformer.sample_structure`` (the
graph-conditioned transformer).
"""

import copy

import torch
import torch.nn as nn


def clones(module: nn.Module, n_clones: int):
    """Produce ``n_clones`` deep copies of ``module`` as an ``nn.ModuleList``."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n_clones)])


def subsequent_mask(size):
    """Mask out subsequent positions (lower-triangular causal mask)."""
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return mask == 0


def make_std_mask(tgt, pad):
    """Create a mask hiding both padding and future positions."""
    tgt_mask = (tgt != pad).unsqueeze(-2)  # noqa
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    return tgt_mask
