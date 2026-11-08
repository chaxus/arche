"""Rotary Position Embedding (RoPE), including Llama 3 / 3.2 frequency scaling."""

from __future__ import annotations

import math

import torch


def llama3_inv_freq(
    head_dim: int,
    theta: float,
    *,
    scale_factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position: int,
    device: torch.device,
) -> torch.Tensor:
    """Inverse frequencies with Llama-3-style wavelength smoothing (HF-compatible)."""
    half = head_dim // 2
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    if scale_factor == 1.0:
        return inv_freq

    low_freq_wavelen = original_max_position / low_freq_factor
    high_freq_wavelen = original_max_position / high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / scale_factor, inv_freq)
    smooth_factor = (original_max_position / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    # Medium band: interpolate between scaled and unscaled (HF formula).
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / scale_factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float = 500_000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    *,
    scale_factor: float = 1.0,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 4.0,
    original_max_position: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for positions ``0 .. seq_len-1``.

    Returns:
        cos, sin with shape ``(seq_len, head_dim)``.
    """
    assert head_dim % 2 == 0, "RoPE head_dim must be even"
    device = device or torch.device("cpu")
    inv_freq = llama3_inv_freq(
        head_dim,
        theta,
        scale_factor=scale_factor,
        low_freq_factor=low_freq_factor,
        high_freq_factor=high_freq_factor,
        original_max_position=original_max_position,
        device=device,
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
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
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)
