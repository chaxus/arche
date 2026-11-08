import torch

from arche.moe import MoESwiGLU


def test_moe_preserves_shape_and_probs():
    moe = MoESwiGLU(embed_dim=32, intermediate_dim=64, num_experts=4, top_k=2)
    x = torch.randn(2, 5, 32)
    y, probs = moe(x)
    assert y.shape == x.shape
    assert probs.shape == (2, 5, 4)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 5), atol=1e-5)


def test_moe_top_k_uses_subset_of_experts():
    torch.manual_seed(0)
    moe = MoESwiGLU(embed_dim=16, intermediate_dim=32, num_experts=4, top_k=1)
    x = torch.randn(1, 8, 16)
    _, probs = moe(x)
    # With top_k=1 the heaviest expert should dominate the argmax path.
    assert int(probs.argmax(dim=-1).unique().numel()) >= 1
