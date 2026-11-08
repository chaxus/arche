import torch

from arche.block import TransformerBlock
from arche.config import Llama32Config


def test_block_preserves_shape():
    cfg = Llama32Config(
        name="tiny",
        vocab_size=128,
        num_layers=1,
        embed_dim=64,
        intermediate_dim=128,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        max_seq_len=32,
        rope_theta=10_000.0,
        rope_scale_factor=1.0,
        norm_eps=1e-5,
    )
    block = TransformerBlock(cfg)
    x = torch.randn(1, 6, 64)
    assert block(x).shape == x.shape
