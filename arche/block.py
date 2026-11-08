"""Transformer block: pre-RMSNorm → attention / MLP + residual."""

from __future__ import annotations

import torch
import torch.nn as nn

from arche.attention import CausalSelfAttention
from arche.config import Llama32Config
from arche.mlp import SwiGLU
from arche.norms import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Llama32Config) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.embed_dim, eps=cfg.norm_eps)
        self.self_attn = CausalSelfAttention(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            rope_theta=cfg.rope_theta,
            max_seq_len=min(cfg.max_seq_len, 8192),
        )
        self.post_attention_layernorm = RMSNorm(cfg.embed_dim, eps=cfg.norm_eps)
        self.mlp = SwiGLU(cfg.embed_dim, cfg.intermediate_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
