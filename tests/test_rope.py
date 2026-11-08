import torch

from arche.rope import apply_rotary_emb, build_rope_cache, rotate_half


def test_rope_cache_shapes():
    cos, sin = build_rope_cache(seq_len=8, head_dim=64, theta=500_000.0)
    assert cos.shape == (8, 64)
    assert sin.shape == (8, 64)


def test_rotate_half_roundtrip_norm():
    x = torch.randn(2, 4, 64)
    # rotate_half is orthogonal up to sign pattern; norm preserved
    assert torch.allclose(rotate_half(x).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_apply_rope_preserves_vector_norm():
    torch.manual_seed(0)
    seq_len, head_dim = 5, 64
    x = torch.randn(2, 3, seq_len, head_dim)
    cos, sin = build_rope_cache(seq_len, head_dim)
    y = apply_rotary_emb(x, cos, sin)
    assert y.shape == x.shape
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5)
