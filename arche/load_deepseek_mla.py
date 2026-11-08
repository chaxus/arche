"""Map DeepSeek-style MLA keys into ``MultiHeadLatentAttention``."""

from __future__ import annotations

from pathlib import Path

import torch

from arche.mla import MultiHeadLatentAttention
from arche.shards import load_sharded_state, write_sharded_state

# Standalone teaching names (one layer's self_attn) → arche MLA module attrs.
_LEAF_MAP = {
    "q_proj.weight": "q_proj.weight",
    "q_a_proj.weight": "q_a.weight",
    "q_b_proj.weight": "q_b.weight",
    "q_rope_proj.weight": "q_rope_proj.weight",
    "kv_a_proj.weight": "kv_a.weight",
    "k_rope_proj.weight": "k_rope_proj.weight",
    "k_b_proj.weight": "k_b.weight",
    "v_b_proj.weight": "v_b.weight",
    "o_proj.weight": "o_proj.weight",
}


def deepseek_mla_to_arche_key(key: str) -> str | None:
    """Map ``...self_attn.<leaf>`` (or bare leaf) onto ``MultiHeadLatentAttention`` names."""
    leaf = key.split("self_attn.")[-1] if "self_attn." in key else key
    return _LEAF_MAP.get(leaf)


def mla_to_deepseek_shaped_state(
    mla: MultiHeadLatentAttention,
    *,
    layer: int = 0,
) -> dict[str, torch.Tensor]:
    """Emit DeepSeek-shaped keys for a standalone MLA module."""
    prefix = f"model.layers.{layer}.self_attn"
    sd = mla.state_dict()
    out: dict[str, torch.Tensor] = {}
    reverse = {v: k for k, v in _LEAF_MAP.items()}
    for arche_key, tensor in sd.items():
        leaf = reverse.get(arche_key)
        if leaf is None:
            continue
        out[f"{prefix}.{leaf}"] = tensor.detach().clone()
    return out


def load_deepseek_shaped_state(
    mla: MultiHeadLatentAttention,
    hf_like: dict[str, torch.Tensor],
) -> None:
    mapped: dict[str, torch.Tensor] = {}
    for src, tensor in hf_like.items():
        dst = deepseek_mla_to_arche_key(src)
        if dst is not None and dst in mla.state_dict():
            mapped[dst] = tensor
    missing, unexpected = mla.load_state_dict(mapped, strict=False)
    critical = [k for k in missing if k.endswith(".weight")]
    if critical:
        raise RuntimeError(f"MLA load missing critical keys: {critical}; unexpected={unexpected}")


def write_deepseek_mla_shards(
    directory: str | Path,
    mla: MultiHeadLatentAttention,
    *,
    layer: int = 0,
    shard_size: int = 2,
) -> Path:
    return write_sharded_state(
        directory,
        mla_to_deepseek_shaped_state(mla, layer=layer),
        shard_size=shard_size,
        prefix="deepseek-mla",
    )


def load_deepseek_mla_shards(directory: str | Path, mla: MultiHeadLatentAttention) -> None:
    load_deepseek_shaped_state(mla, load_sharded_state(directory))
