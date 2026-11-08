import torch

from arche.checkpoint import load_checkpoint, save_checkpoint
from arche.load_mixtral import load_mixtral_shaped_state, mixtral_expert_to_moe_key
from arche.mla import MultiHeadLatentAttention
from arche.moe import MoESwiGLU


def test_mixtral_key_mapping():
    assert (
        mixtral_expert_to_moe_key("model.layers.0.block_sparse_moe.experts.2.w1.weight")
        == "experts.2.gate_proj.weight"
    )
    assert mixtral_expert_to_moe_key("model.layers.0.block_sparse_moe.gate.weight") == "gate.weight"


def test_moe_mixtral_shaped_load_and_roundtrip(tmp_path):
    moe = MoESwiGLU(16, 32, num_experts=2, top_k=1)
    hf_like = {
        "model.layers.0.block_sparse_moe.gate.weight": moe.gate.weight.detach().clone(),
        "model.layers.0.block_sparse_moe.experts.0.w1.weight": moe.experts[0].gate_proj.weight.detach().clone(),
        "model.layers.0.block_sparse_moe.experts.0.w3.weight": moe.experts[0].up_proj.weight.detach().clone(),
        "model.layers.0.block_sparse_moe.experts.0.w2.weight": moe.experts[0].down_proj.weight.detach().clone(),
        "model.layers.0.block_sparse_moe.experts.1.w1.weight": moe.experts[1].gate_proj.weight.detach().clone(),
        "model.layers.0.block_sparse_moe.experts.1.w3.weight": moe.experts[1].up_proj.weight.detach().clone(),
        "model.layers.0.block_sparse_moe.experts.1.w2.weight": moe.experts[1].down_proj.weight.detach().clone(),
    }
    other = MoESwiGLU(16, 32, num_experts=2, top_k=1)
    with torch.no_grad():
        for p in other.parameters():
            p.zero_()
    load_mixtral_shaped_state(other, hf_like)
    x = torch.randn(1, 3, 16)
    with torch.no_grad():
        a, _ = moe(x)
        b, _ = other(x)
    assert torch.allclose(a, b, atol=1e-5)

    path = tmp_path / "moe.pt"
    save_checkpoint(path, moe, meta={"kind": "moe"})
    again = MoESwiGLU(16, 32, num_experts=2, top_k=1)
    meta = load_checkpoint(path, again)
    assert meta["kind"] == "moe"


def test_mla_checkpoint_roundtrip(tmp_path):
    mla = MultiHeadLatentAttention(32, num_heads=2, head_dim=8, kv_lora_rank=8, rope_dim=8, max_seq_len=16)
    path = tmp_path / "mla.pt"
    save_checkpoint(path, mla, meta={"kind": "mla"})
    other = MultiHeadLatentAttention(32, num_heads=2, head_dim=8, kv_lora_rank=8, rope_dim=8, max_seq_len=16)
    load_checkpoint(path, other)
    x = torch.randn(1, 4, 32)
    with torch.no_grad():
        a, _ = mla(x)
        b, _ = other(x)
    assert torch.allclose(a, b, atol=1e-5)
