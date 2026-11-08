"""Autoregressive text generation over the hand-written ``Llama``."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from arche.cache import KVCache
from arche.model import Llama


def _sample_next(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        cutoff = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: Llama,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: int | None = 50,
    eos_id: int | None = None,
    use_cache: bool = True,
) -> torch.Tensor:
    """Greedy / sampled decoding.

    Args:
        model: arche ``Llama`` in eval mode.
        input_ids: ``(batch, seq)`` prompt ids.
        max_new_tokens: how many tokens to append.
        temperature: ``1.0`` = plain sampling; lower is sharper. ``0`` → greedy.
        top_k: keep only the top-k logits before sampling; ``None`` disables.
        eos_id: optional stop token; generation for a row stops when sampled.
        use_cache: reuse K/V across steps (prefill once, then decode token-by-token).
    Returns:
        ``(batch, seq + new)`` token ids including the prompt.
    """
    model.eval()
    out = input_ids
    finished = torch.zeros(out.shape[0], dtype=torch.bool, device=out.device)
    caches: list[KVCache | None] | None = [None] * len(model.layers) if use_cache else None

    if use_cache:
        logits, caches = model(out, caches=caches, start_pos=0)
        next_ids = _sample_next(logits[:, -1, :], temperature=temperature, top_k=top_k)
        if eos_id is not None:
            finished = finished | (next_ids.squeeze(-1) == eos_id)
        out = torch.cat([out, next_ids], dim=1)
        start = out.shape[1] - 1
        steps = max_new_tokens - 1
    else:
        start = 0
        steps = max_new_tokens

    for _ in range(steps):
        if use_cache:
            assert caches is not None
            step_ids = out[:, -1:]
            logits, caches = model(step_ids, caches=caches, start_pos=start)
            start += 1
        else:
            logits, _ = model(out, caches=None, start_pos=0)

        next_ids = _sample_next(logits[:, -1, :], temperature=temperature, top_k=top_k)
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
