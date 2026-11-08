"""Multi-head Latent Attention (MLA) — DeepSeek-style compressed KV."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from arche.rope import apply_rotary_emb, build_rope_cache


@dataclass
class MLACache:
    """Cached latent KV and RoPE keys — much smaller than full K/V."""

    kv_latent: torch.Tensor  # (batch, seq, kv_lora_rank)
    k_rope: torch.Tensor  # (batch, seq, rope_dim)

    @property
    def seq_len(self) -> int:
        return int(self.kv_latent.shape[1])

    def append(self, kv_latent: torch.Tensor, k_rope: torch.Tensor) -> MLACache:
        return MLACache(
            kv_latent=torch.cat([self.kv_latent, kv_latent], dim=1),
            k_rope=torch.cat([self.k_rope, k_rope], dim=1),
        )


class MultiHeadLatentAttention(nn.Module):
    """Educational MLA with low-rank KV (and optional Q) compression.

    Ideas taken from DeepSeek-V2/V3:
    - compress tokens into a shared KV latent before expanding to heads
    - keep a small RoPE stream decoupled from the latent content path
    - cache the latent + RoPE key instead of full per-head K/V
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        kv_lora_rank: int,
        q_lora_rank: int | None = None,
        rope_dim: int | None = None,
        rope_theta: float = 500_000.0,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.rope_dim = rope_dim or max(16, head_dim // 2)
        assert self.rope_dim % 2 == 0

        if q_lora_rank is None:
            self.q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
            self.q_a = None
            self.q_b = None
        else:
            self.q_proj = None
            self.q_a = nn.Linear(embed_dim, q_lora_rank, bias=False)
            self.q_b = nn.Linear(q_lora_rank, num_heads * head_dim, bias=False)

        self.q_rope_proj = nn.Linear(embed_dim, num_heads * self.rope_dim, bias=False)
        self.kv_a = nn.Linear(embed_dim, kv_lora_rank, bias=False)
        self.k_rope_proj = nn.Linear(embed_dim, self.rope_dim, bias=False)
        self.k_b = nn.Linear(kv_lora_rank, num_heads * head_dim, bias=False)
        self.v_b = nn.Linear(kv_lora_rank, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, embed_dim, bias=False)

        cos, sin = build_rope_cache(max_seq_len, self.rope_dim, theta=rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _project_q(self, x: torch.Tensor) -> torch.Tensor:
        if self.q_proj is not None:
            return self.q_proj(x)
        assert self.q_a is not None and self.q_b is not None
        return self.q_b(self.q_a(x))

    def forward(
        self,
        x: torch.Tensor,
        cache: MLACache | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, MLACache]:
        bsz, seq_len, _ = x.shape

        q = self._project_q(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q_rope = (
            self.q_rope_proj(x)
            .view(bsz, seq_len, self.num_heads, self.rope_dim)
            .transpose(1, 2)
        )

        kv_latent = self.kv_a(x)  # (B, S, R)
        k_rope = self.k_rope_proj(x)  # (B, S, rope_dim)

        end = start_pos + seq_len
        cos = self.rope_cos[start_pos:end]
        sin = self.rope_sin[start_pos:end]
        q_rope = apply_rotary_emb(q_rope, cos, sin)
        k_rope_rot = apply_rotary_emb(k_rope, cos, sin)

        new_cache = (
            MLACache(kv_latent=kv_latent, k_rope=k_rope_rot)
            if cache is None
            else cache.append(kv_latent, k_rope_rot)
        )

        k = (
            self.k_b(new_cache.kv_latent)
            .view(bsz, new_cache.seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_b(new_cache.kv_latent)
            .view(bsz, new_cache.seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k_rope_h = new_cache.k_rope.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        q_full = torch.cat([q, q_rope], dim=-1)
        k_full = torch.cat([k, k_rope_h], dim=-1)

        is_causal = cache is None and seq_len > 1
        attn = F.scaled_dot_product_attention(
            q_full,
            k_full,
            v,
            is_causal=is_causal,
            scale=1.0 / math.sqrt(self.head_dim + self.rope_dim),
        )
        out = attn.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(out), new_cache
