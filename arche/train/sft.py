"""Supervised fine-tuning on prompt/response token ids."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from arche.model import Llama


def sft_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_lens: torch.Tensor,
) -> torch.Tensor:
    """Mask the prompt tokens; only supervise the response span."""
    # Shift for next-token prediction.
    shift_logits = logits[:, :-1]
    shift_labels = input_ids[:, 1:]
    bsz, seq, vocab = shift_logits.shape
    losses = F.cross_entropy(
        shift_logits.reshape(-1, vocab),
        shift_labels.reshape(-1),
        reduction="none",
    ).view(bsz, seq)

    mask = torch.zeros_like(shift_labels, dtype=torch.bool)
    for i in range(bsz):
        # Labels at index j predict token j+1; supervise from prompt_len-1 onward.
        start = max(int(prompt_lens[i].item()) - 1, 0)
        mask[i, start:] = True
    return losses[mask].mean()


def train_sft_step(
    model: Llama,
    input_ids: torch.Tensor,
    prompt_lens: torch.Tensor,
    *,
    lr: float = 2e-4,
) -> float:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    opt.zero_grad(set_to_none=True)
    logits, _ = model(input_ids)
    loss = sft_loss(logits, input_ids, prompt_lens)
    loss.backward()
    opt.step()
    model.eval()
    return float(loss.detach())
