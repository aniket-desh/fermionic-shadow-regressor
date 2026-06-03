"""Shared transformer building blocks used by BOTH generative transformer
variants (``src.models.transformer``, ``src.models.gctransformer``) AND the
regression pipeline's FiLM transformer (``fermionic_pipeline.models.film_transformer``).

This package is the single source of truth; the per-model packages re-export
these names so legacy import paths keep working.
"""

from .utils import clones, subsequent_mask, make_std_mask
from .modules import (
    LayerNorm,
    SublayerConnection,
    attention,
    MultiHeadAttention,
    PositionwiseFeedForward,
    Embeddings,
    PositionalEncoding,
    PositionalEncoding2D,
)
from .layers import DecoderLayer
from .models import Decoder, Generator
from .loss import LabelSmoothing

__all__ = [
    "clones",
    "subsequent_mask",
    "make_std_mask",
    "LayerNorm",
    "SublayerConnection",
    "attention",
    "MultiHeadAttention",
    "PositionwiseFeedForward",
    "Embeddings",
    "PositionalEncoding",
    "PositionalEncoding2D",
    "DecoderLayer",
    "Decoder",
    "Generator",
    "LabelSmoothing",
]
