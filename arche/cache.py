"""Per-layer key/value cache for autoregressive decoding."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class KVCache:
    """Cached K/V with shape ``(batch, kv_heads, seq, head_dim)``."""

    k: torch.Tensor
    v: torch.Tensor

    @property
    def seq_len(self) -> int:
        return int(self.k.shape[-2])

    def append(self, k: torch.Tensor, v: torch.Tensor) -> KVCache:
        return KVCache(k=torch.cat([self.k, k], dim=-2), v=torch.cat([self.v, v], dim=-2))
