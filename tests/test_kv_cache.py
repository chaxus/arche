import torch

from arche.cache import KVCache
from arche.config import Llama32Config
from arche.generate import generate
from arche.model import Llama


def _tiny() -> Llama:
    cfg = Llama32Config(
        name="tiny",
        vocab_size=64,
        num_layers=2,
        embed_dim=32,
        intermediate_dim=64,
        num_heads=2,
        num_kv_heads=1,
        head_dim=16,
        max_seq_len=64,
        rope_theta=10_000.0,
        rope_scale_factor=1.0,
        norm_eps=1e-5,
    )
    return Llama(cfg)


def test_cache_matches_full_forward_logits():
    torch.manual_seed(0)
    model = _tiny().eval()
    ids = torch.randint(0, 64, (1, 5))

    full_logits, _ = model(ids)
    caches: list[KVCache | None] = [None, None]
    prefill_logits, caches = model(ids[:, :4], caches=caches, start_pos=0)
    step_logits, _ = model(ids[:, 4:], caches=caches, start_pos=4)

    assert torch.allclose(prefill_logits[:, -1, :], full_logits[:, 3, :], atol=1e-5, rtol=1e-4)
    assert torch.allclose(step_logits[:, -1, :], full_logits[:, 4, :], atol=1e-5, rtol=1e-4)


def test_generate_cache_matches_full_recompute_greedy():
    torch.manual_seed(0)
    model = _tiny().eval()
    prompt = torch.randint(0, 64, (1, 3))
    cached = generate(model, prompt.clone(), max_new_tokens=4, temperature=0.0, use_cache=True)
    full = generate(model, prompt.clone(), max_new_tokens=4, temperature=0.0, use_cache=False)
    assert torch.equal(cached, full)
