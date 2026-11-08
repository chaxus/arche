"""Grouped-Query Attention (GQA) with causal mask — Llama 3.2 style."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from arche.rope import apply_rotary_emb, build_rope_cache


class CausalSelfAttention(nn.Module):
    """Multi-head attention with fewer KV heads than query heads (GQA)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_theta: float = 500_000.0,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.n_rep = num_heads // num_kv_heads

        self.q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, embed_dim, bias=False)

        cos, sin = build_rope_cache(max_seq_len, head_dim, theta=rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: ``(batch, seq, embed_dim)``
        """
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        # (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos = self.rope_cos[:seq_len]
        sin = self.rope_sin[:seq_len]
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Repeat KV heads to match Q heads (GQA).
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        # Causal attention: each position may only see itself and the past.
        attn = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=1.0 / math.sqrt(self.head_dim)
        )
        out = attn.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(out)
