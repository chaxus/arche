import torch

from arche.config import Llama32Config
from arche.model import Llama


def _tiny_cfg(*, layers: int = 2) -> Llama32Config:
    return Llama32Config(
        name="tiny",
        vocab_size=128,
        num_layers=layers,
        embed_dim=64,
        intermediate_dim=128,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        max_seq_len=32,
        rope_theta=10_000.0,
        rope_scale_factor=1.0,
        norm_eps=1e-5,
        tie_word_embeddings=True,
    )


def test_llama_logits_shape_and_tying():
    cfg = _tiny_cfg()
    model = Llama(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 5))
    logits = model(ids)
    assert logits.shape == (2, 5, cfg.vocab_size)
    assert model.lm_head.weight is model.embed_tokens.weight


def test_llama_stack_depth():
    cfg = _tiny_cfg(layers=3)
    model = Llama(cfg)
    assert len(model.layers) == 3
