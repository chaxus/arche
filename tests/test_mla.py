import torch

from arche.mla import MLACache, MultiHeadLatentAttention


def test_mla_preserves_shape_and_builds_latent_cache():
    mla = MultiHeadLatentAttention(
        embed_dim=64,
        num_heads=4,
        head_dim=12,
        kv_lora_rank=16,
        q_lora_rank=32,
        rope_dim=8,
        max_seq_len=64,
    )
    x = torch.randn(2, 6, 64)
    y, cache = mla(x)
    assert y.shape == x.shape
    assert isinstance(cache, MLACache)
    assert cache.kv_latent.shape == (2, 6, 16)
    assert cache.k_rope.shape == (2, 6, 8)


def test_mla_cache_decode_step_matches_full_prefix_slice():
    torch.manual_seed(0)
    mla = MultiHeadLatentAttention(
        embed_dim=32,
        num_heads=2,
        head_dim=8,
        kv_lora_rank=8,
        q_lora_rank=None,
        rope_dim=8,
        max_seq_len=32,
    ).eval()
    x = torch.randn(1, 5, 32)
    full, _ = mla(x)
    caches = None
    prefill, caches = mla(x[:, :4], cache=caches, start_pos=0)
    step, _ = mla(x[:, 4:], cache=caches, start_pos=4)
    assert torch.allclose(prefill[:, -1], full[:, 3], atol=1e-5, rtol=1e-4)
    assert torch.allclose(step[:, -1], full[:, 4], atol=1e-5, rtol=1e-4)


def test_mla_latent_cache_is_smaller_than_full_kv():
    num_heads, head_dim, kv_lora_rank, rope_dim = 8, 16, 16, 8
    seq = 32
    full_kv_elems = 2 * num_heads * seq * head_dim
    latent_elems = seq * kv_lora_rank + seq * rope_dim
    assert latent_elems < full_kv_elems
