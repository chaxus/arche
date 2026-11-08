"""Rotary Position Embedding (RoPE).

Matches the Llama / Hugging Face ``rotate_half`` convention.
Llama 3.2 long-context frequency scaling lands in a later chapter.
"""

from __future__ import annotations

import torch


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float = 500_000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for positions ``0 .. seq_len-1``.

    Returns:
        cos, sin with shape ``(seq_len, head_dim)``.
    """
    assert head_dim % 2 == 0, "RoPE head_dim must be even"
    device = device or torch.device("cpu")
    half = head_dim // 2
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # (seq_len, half)
    emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap halves and negate the new first half (Llama-style)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to ``x`` with shape ``(..., seq_len, head_dim)``."""
    # Broadcast cos/sin over any leading dims (batch, heads, ...).
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)
