"""Simplified continuous batching for teaching decode scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from arche.generate import _sample_next
from arche.model import Llama


@dataclass
class DecodeRequest:
    request_id: str
    input_ids: list[int]
    max_new_tokens: int
    temperature: float = 0.0
    top_k: int | None = 50
    eos_id: int | None = None


@dataclass
class _Live:
    request: DecodeRequest
    tokens: list[int]
    generated: int = 0
    done: bool = False


@dataclass
class BatchResult:
    outputs: dict[str, list[int]] = field(default_factory=dict)
    steps: int = 0


@torch.no_grad()
def generate_continuous(
    model: Llama,
    requests: list[DecodeRequest],
    *,
    device: torch.device | None = None,
) -> BatchResult:
    """Run many short generations by stepping active sequences together.

    This is a teaching scheduler: pad active rows each step, sample, retire
    finished sequences. It is not a production paged-attention engine.
    """
    model.eval()
    device = device or next(model.parameters()).device
    live = [_Live(request=r, tokens=list(r.input_ids)) for r in requests]
    result = BatchResult()
    steps = 0

    while any(not item.done for item in live):
        active = [item for item in live if not item.done]
        if not active:
            break
        max_len = max(len(item.tokens) for item in active)
        batch = torch.zeros(len(active), max_len, dtype=torch.long, device=device)
        for row, item in enumerate(active):
            batch[row, : len(item.tokens)] = torch.tensor(item.tokens, device=device)

        logits, _ = model(batch)
        steps += 1
        for row, item in enumerate(active):
            next_id = int(
                _sample_next(
                    logits[row : row + 1, len(item.tokens) - 1, :],
                    temperature=item.request.temperature,
                    top_k=item.request.top_k,
                ).item()
            )
            item.tokens.append(next_id)
            item.generated += 1
            if (
                item.request.eos_id is not None and next_id == item.request.eos_id
            ) or item.generated >= item.request.max_new_tokens:
                item.done = True
                result.outputs[item.request.request_id] = list(item.tokens)

    result.steps = steps
    for item in live:
        result.outputs.setdefault(item.request.request_id, list(item.tokens))
    return result
