"""Mixture-of-Experts feed-forward — Mixtral / DeepSeek-style sparse MLP."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from arche.mlp import SwiGLU


class MoESwiGLU(nn.Module):
    """Top-k routed SwiGLU experts.

    Each token picks ``top_k`` experts via a linear router, then forms a
    weighted sum of those expert outputs. Dense parameter count grows with
    ``num_experts``; per-token compute stays near ``top_k`` experts.
    """

    def __init__(
        self,
        embed_dim: int,
        intermediate_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(embed_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(embed_dim, intermediate_dim) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            x: ``(batch, seq, embed_dim)``
        Returns:
            output with the same shape, and router probabilities ``(batch, seq, experts)``
        """
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        top_probs, top_idx = probs.topk(self.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        flat = x.reshape(-1, self.embed_dim)
        out = torch.zeros_like(flat)
        flat_idx = top_idx.reshape(-1, self.top_k)
        flat_w = top_probs.reshape(-1, self.top_k)

        for slot in range(self.top_k):
            expert_ids = flat_idx[:, slot]
            weights = flat_w[:, slot].unsqueeze(-1)
            for expert_id, expert in enumerate(self.experts):
                mask = expert_ids == expert_id
                if not bool(mask.any()):
                    continue
                out[mask] = out[mask] + weights[mask] * expert(flat[mask])

        return out.view_as(x), probs
