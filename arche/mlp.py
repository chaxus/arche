"""SwiGLU feed-forward — default MLP in Llama / Qwen dense stacks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """``down(silu(gate(x)) * up(x))`` with no bias."""

    def __init__(self, embed_dim: int, intermediate_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
