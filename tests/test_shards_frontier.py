from pathlib import Path

import torch

from arche.load_deepseek_mla import (
    deepseek_mla_to_arche_key,
    load_deepseek_mla_shards,
    write_deepseek_mla_shards,
)
from arche.load_mixtral import load_mixtral_shards_into_moe, write_mixtral_shards
from arche.mla import MultiHeadLatentAttention
from arche.moe import MoESwiGLU
from arche.shards import load_sharded_state, write_sharded_state


def test_sharded_state_roundtrip(tmp_path: Path):
    state = {"a.weight": torch.randn(2, 2), "b.weight": torch.randn(3), "c.weight": torch.randn(1, 4)}
    write_sharded_state(tmp_path, state, shard_size=2)
    loaded = load_sharded_state(tmp_path)
    assert set(loaded) == set(state)
    for key, tensor in state.items():
        assert torch.equal(loaded[key], tensor)


def test_mixtral_sharded_allclose(tmp_path: Path):
    moe = MoESwiGLU(16, 32, num_experts=2, top_k=1)
    write_mixtral_shards(tmp_path, moe, shard_size=2)
    other = MoESwiGLU(16, 32, num_experts=2, top_k=1)
    with torch.no_grad():
        for p in other.parameters():
            p.zero_()
    load_mixtral_shards_into_moe(tmp_path, other)
    x = torch.randn(1, 3, 16)
    with torch.no_grad():
        a, _ = moe(x)
        b, _ = other(x)
    assert torch.allclose(a, b, atol=1e-5)


def test_deepseek_mla_key_and_shards(tmp_path: Path):
    assert (
        deepseek_mla_to_arche_key("model.layers.0.self_attn.kv_a_proj.weight") == "kv_a.weight"
    )
    mla = MultiHeadLatentAttention(32, 2, 8, kv_lora_rank=8, rope_dim=8, max_seq_len=16)
    write_deepseek_mla_shards(tmp_path, mla, shard_size=2)
    other = MultiHeadLatentAttention(32, 2, 8, kv_lora_rank=8, rope_dim=8, max_seq_len=16)
    with torch.no_grad():
        for p in other.parameters():
            p.zero_()
    load_deepseek_mla_shards(tmp_path, other)
    x = torch.randn(1, 4, 32)
    with torch.no_grad():
        a, _ = mla(x)
        b, _ = other(x)
    assert torch.allclose(a, b, atol=1e-5)
