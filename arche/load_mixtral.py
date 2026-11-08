"""Map Mixtral-style MoE expert keys into ``MoESwiGLU`` (synthetic-friendly)."""

from __future__ import annotations

import torch

from arche.moe import MoESwiGLU


def mixtral_expert_to_moe_key(key: str) -> str | None:
    """Translate common HF Mixtral MLP expert names to arche MoE names.

    Example HF: ``model.layers.0.block_sparse_moe.experts.3.w1.weight``
    Arche MoE (standalone module): ``experts.3.gate_proj.weight``
    """
    # Standalone MoE module mapping used in tests / primers.
    table = {
        "w1.weight": "gate_proj.weight",
        "w3.weight": "up_proj.weight",
        "w2.weight": "down_proj.weight",
    }
    if "experts." not in key:
        if key.endswith("gate.weight") or key.endswith("router.weight"):
            return "gate.weight"
        return None
    # ...experts.{i}.{w1|w2|w3}.weight
    parts = key.split("experts.")[-1].split(".")
    if len(parts) < 2:
        return None
    idx, leaf = parts[0], ".".join(parts[1:])
    mapped = table.get(leaf)
    if mapped is None:
        return None
    return f"experts.{idx}.{mapped}"


def load_mixtral_shaped_state(moe: MoESwiGLU, hf_like: dict[str, torch.Tensor]) -> None:
    mapped: dict[str, torch.Tensor] = {}
    for src, tensor in hf_like.items():
        dst = mixtral_expert_to_moe_key(src)
        if dst is not None:
            mapped[dst] = tensor
    missing, unexpected = moe.load_state_dict(mapped, strict=False)
    critical = [k for k in missing if k.startswith("experts.") or k == "gate.weight"]
    if critical:
        raise RuntimeError(f"MoE load missing critical keys: {critical}; unexpected={unexpected}")
