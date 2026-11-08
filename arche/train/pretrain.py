"""Minimal next-token pretraining loop on toy batches."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from arche.model import Llama


def lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Standard causal LM loss: predict token t+1 from positions 0..T-1."""
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        targets[:, 1:].reshape(-1),
    )


def train_pretrain_steps(
    model: Llama,
    batches: list[torch.Tensor],
    *,
    lr: float = 3e-4,
) -> list[float]:
    """Run a few optimizer steps; returns the loss history."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history: list[float] = []
    for batch in batches:
        opt.zero_grad(set_to_none=True)
        logits, _ = model(batch)
        loss = lm_loss(logits, batch)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    model.eval()
    return history
