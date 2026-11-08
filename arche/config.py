"""Llama 3.2 dense configs.

Numbers match Meta / Hugging Face Llama 3.2 1B and 3B text models.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Llama32Config:
    name: str
    vocab_size: int
    num_layers: int
    embed_dim: int
    intermediate_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    rope_theta: float
    rope_scale_factor: float
    norm_eps: float
    tie_word_embeddings: bool = True

    @property
    def num_query_groups(self) -> int:
        """How many query heads share one KV head (GQA)."""
        assert self.num_heads % self.num_kv_heads == 0
        return self.num_heads // self.num_kv_heads


def llama32_1b() -> Llama32Config:
    return Llama32Config(
        name="Llama-3.2-1B",
        vocab_size=128_256,
        num_layers=16,
        embed_dim=2048,
        intermediate_dim=8192,
        num_heads=32,
        num_kv_heads=8,
        head_dim=64,
        max_seq_len=131_072,
        rope_theta=500_000.0,
        rope_scale_factor=32.0,
        norm_eps=1e-5,
        tie_word_embeddings=True,
    )


def llama32_3b() -> Llama32Config:
    return Llama32Config(
        name="Llama-3.2-3B",
        vocab_size=128_256,
        num_layers=28,
        embed_dim=3072,
        intermediate_dim=8192,
        num_heads=24,
        num_kv_heads=8,
        head_dim=128,
        max_seq_len=131_072,
        rope_theta=500_000.0,
        rope_scale_factor=32.0,
        norm_eps=1e-5,
        tie_word_embeddings=True,
    )
