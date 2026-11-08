import torch

from arche.attention import CausalSelfAttention


def test_gqa_preserves_shape():
    attn = CausalSelfAttention(64, num_heads=4, num_kv_heads=2, head_dim=16, max_seq_len=32)
    x = torch.randn(2, 7, 64)
    out, cache = attn(x)
    assert out.shape == x.shape
    assert cache.seq_len == 7
