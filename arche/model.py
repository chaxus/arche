"""Full Llama 3.2-style dense decoder: embed → blocks → norm → LM head."""

from __future__ import annotations

import torch
import torch.nn as nn

from arche.block import TransformerBlock
from arche.config import Llama32Config
from arche.norms import RMSNorm


class Llama(nn.Module):
    """Modern dense causal LM matching the Llama 3.2 layout."""

    def __init__(self, cfg: Llama32Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.embed_dim, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Args:
            input_ids: ``(batch, seq)``
        Returns:
            logits ``(batch, seq, vocab_size)``
        """
        x = self.embed_tokens(input_ids)
        for block in self.layers:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)
