"""Shared decoder layer (single source of truth).

The generative ``transformer`` and the graph-conditioned ``gctransformer`` used
byte-identical decoder layers; this is the canonical version. The forward
signature accepts the mask positionally (``layer(x, mask)``) or by keyword
(``layer(x, tgt_mask=mask)``), so both decoders work unchanged.
"""

import torch.nn as nn

from src.models.transformer_core.utils import clones
from src.models.transformer_core.modules import (
    SublayerConnection,
    MultiHeadAttention,
    PositionwiseFeedForward,
)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        size: int,
        self_attn: MultiHeadAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, tgt_mask):
        x = self.sublayer[0](
            x, lambda x: self.self_attn(query=x, key=x, value=x, mask=tgt_mask)
        )
        x = self.sublayer[1](
            x, lambda x: self.self_attn(query=x, key=x, value=x, mask=tgt_mask)
        )
        return self.sublayer[2](x, self.feed_forward)
