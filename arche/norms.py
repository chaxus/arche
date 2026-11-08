"""RMSNorm — default pre-norm in modern Llama / Qwen dense stacks."""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no mean centering, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for stability, cast back to input dtype.
        orig_dtype = x.dtype
        x_f = x.float()
        variance = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_f * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * x_normed).to(orig_dtype)
