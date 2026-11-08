"""Full Llama 3.2-style dense decoder: embed → blocks → norm → LM head."""

from __future__ import annotations

import torch
import torch.nn as nn

from arche.block import TransformerBlock
from arche.cache import KVCache
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

    def forward(
        self,
        input_ids: torch.Tensor,
        caches: list[KVCache | None] | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, list[KVCache]]:
        """Args:
            input_ids: ``(batch, seq)``
            caches: optional per-layer KV caches
            start_pos: absolute position of ``input_ids[:, 0]``
        Returns:
            logits ``(batch, seq, vocab_size)`` and updated caches
        """
        if caches is None:
            caches = [None] * len(self.layers)
        assert len(caches) == len(self.layers)

        x = self.embed_tokens(input_ids)
        new_caches: list[KVCache] = []
        for layer, cache in zip(self.layers, caches, strict=True):
            x, new_cache = layer(x, cache=cache, start_pos=start_pos)
            new_caches.append(new_cache)
        x = self.norm(x)
        return self.lm_head(x), new_caches
