"""Token embedding — maps token ids to dense vectors."""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Lookup table of shape ``(vocab_size, embed_dim)``.

    Llama 3.2 1B/3B tie this weight with the LM head (handled at model level).
    Position is *not* added here — modern dense stacks use RoPE instead.
    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.weight = nn.Embedding(vocab_size, embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Args:
            token_ids: ``(..., seq_len)`` int64
        Returns:
            ``(..., seq_len, embed_dim)``
        """
        return self.weight(token_ids)
