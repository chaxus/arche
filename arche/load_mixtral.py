"""Map Mixtral-style MoE expert keys into ``MoESwiGLU`` (synthetic-friendly)."""

from __future__ import annotations

from pathlib import Path

import torch

from arche.moe import MoESwiGLU
from arche.shards import load_sharded_state, write_sharded_state


def mixtral_expert_to_moe_key(key: str) -> str | None:
    """Translate common HF Mixtral MLP expert names to arche MoE names.

    Example HF: ``model.layers.0.block_sparse_moe.experts.3.w1.weight``
    Arche MoE (standalone module): ``experts.3.gate_proj.weight``
    """
    table = {
        "w1.weight": "gate_proj.weight",
        "w3.weight": "up_proj.weight",
        "w2.weight": "down_proj.weight",
    }
    if "experts." not in key:
        if key.endswith("gate.weight") or key.endswith("router.weight"):
            return "gate.weight"
        return None
    parts = key.split("experts.")[-1].split(".")
    if len(parts) < 2:
        return None
    idx, leaf = parts[0], ".".join(parts[1:])
    mapped = table.get(leaf)
    if mapped is None:
        return None
    return f"experts.{idx}.{mapped}"


def moe_to_mixtral_shaped_state(moe: MoESwiGLU, *, layer: int = 0) -> dict[str, torch.Tensor]:
    """Emit HF-like Mixtral keys for a standalone ``MoESwiGLU`` (tests / synthetic shards)."""
    prefix = f"model.layers.{layer}.block_sparse_moe"
    out: dict[str, torch.Tensor] = {
        f"{prefix}.gate.weight": moe.gate.weight.detach().clone(),
    }
    for i, expert in enumerate(moe.experts):
        out[f"{prefix}.experts.{i}.w1.weight"] = expert.gate_proj.weight.detach().clone()
        out[f"{prefix}.experts.{i}.w3.weight"] = expert.up_proj.weight.detach().clone()
        out[f"{prefix}.experts.{i}.w2.weight"] = expert.down_proj.weight.detach().clone()
    return out


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


def write_mixtral_shards(
    directory: str | Path,
    moe: MoESwiGLU,
    *,
    layer: int = 0,
    shard_size: int = 2,
) -> Path:
    """Write Mixtral-shaped tensors as an index + ``.pt`` shards."""
    return write_sharded_state(
        directory,
        moe_to_mixtral_shaped_state(moe, layer=layer),
        shard_size=shard_size,
        prefix="mixtral-moe",
    )


def load_mixtral_shards_into_moe(directory: str | Path, moe: MoESwiGLU) -> None:
    """Load Mixtral-shaped shards from ``directory`` into ``moe``."""
    load_mixtral_shaped_state(moe, load_sharded_state(directory))
