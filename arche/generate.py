"""Autoregressive text generation over the hand-written ``Llama``."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from arche.model import Llama


@torch.no_grad()
def generate(
    model: Llama,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: int | None = 50,
    eos_id: int | None = None,
) -> torch.Tensor:
    """Greedy / sampled decoding.

    Args:
        model: arche ``Llama`` in eval mode.
        input_ids: ``(batch, seq)`` prompt ids.
        max_new_tokens: how many tokens to append.
        temperature: ``1.0`` = plain sampling; lower is sharper. ``0`` → greedy.
        top_k: keep only the top-k logits before sampling; ``None`` disables.
        eos_id: optional stop token; generation for a row stops when sampled.
    Returns:
        ``(batch, seq + new)`` token ids including the prompt.
    """
    model.eval()
    out = input_ids
    finished = torch.zeros(out.shape[0], dtype=torch.bool, device=out.device)

    for _ in range(max_new_tokens):
        logits = model(out)[:, -1, :]  # (batch, vocab)
        if temperature <= 0:
            next_ids = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                cutoff = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
            probs = F.softmax(logits, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1)

        if eos_id is not None:
            next_ids = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_ids, eos_id),
                next_ids,
            )
            finished = finished | (next_ids.squeeze(-1) == eos_id)

        out = torch.cat([out, next_ids], dim=1)
        if eos_id is not None and bool(finished.all()):
            break
    return out
