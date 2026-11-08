"""Direct Preference Optimization (DPO) — one educational step."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from arche.model import Llama


def _gather_logprobs(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    logprobs = F.log_softmax(logits[:, :-1], dim=-1)
    target = ids[:, 1:]
    return logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1).sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    *,
    beta: float = 0.1,
) -> torch.Tensor:
    """Classic DPO objective (Rafailov et al.)."""
    pi = policy_chosen_logps - policy_rejected_logps
    ref = ref_chosen_logps - ref_rejected_logps
    return -F.logsigmoid(beta * (pi - ref)).mean()


def sequence_logprob(model: Llama, ids: torch.Tensor) -> torch.Tensor:
    logits, _ = model(ids)
    return _gather_logprobs(logits, ids)


def train_dpo_step(
    policy: Llama,
    reference: Llama,
    chosen_ids: torch.Tensor,
    rejected_ids: torch.Tensor,
    *,
    beta: float = 0.1,
    lr: float = 1e-4,
) -> float:
    """One DPO update; ``reference`` is frozen."""
    policy.train()
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=lr)
    opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        ref_c = sequence_logprob(reference, chosen_ids)
        ref_r = sequence_logprob(reference, rejected_ids)
    pol_c = sequence_logprob(policy, chosen_ids)
    pol_r = sequence_logprob(policy, rejected_ids)
    loss = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=beta)
    loss.backward()
    opt.step()
    policy.eval()
    return float(loss.detach())
