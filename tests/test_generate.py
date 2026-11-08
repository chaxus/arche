import torch

from arche.config import Llama32Config
from arche.generate import generate
from arche.model import Llama


def test_generate_appends_tokens():
    cfg = Llama32Config(
        name="tiny",
        vocab_size=64,
        num_layers=1,
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
    model = Llama(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (1, 3))
    out = generate(model, prompt, max_new_tokens=5, temperature=0.0)
    assert out.shape == (1, 8)
    assert torch.equal(out[:, :3], prompt)


def test_generate_respects_eos():
    cfg = Llama32Config(
        name="tiny",
        vocab_size=32,
        num_layers=1,
        embed_dim=16,
        intermediate_dim=32,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=32,
        rope_theta=10_000.0,
        rope_scale_factor=1.0,
        norm_eps=1e-5,
    )
    model = Llama(cfg)

    # Force the next-token distribution via a hook on lm_head output by
    # replacing forward temporarily — simpler: monkeypatch model forward.
    calls = {"n": 0}

    def fake_forward(input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq = input_ids.shape
        logits = torch.zeros(bsz, seq, cfg.vocab_size)
        # Always prefer eos id 7 at the last position.
        logits[:, -1, 7] = 10.0
        calls["n"] += 1
        return logits

    model.forward = fake_forward  # type: ignore[method-assign]
    prompt = torch.tensor([[1, 2]])
    out = generate(model, prompt, max_new_tokens=10, temperature=0.0, eos_id=7)
    assert out.shape[1] == 3  # prompt + one eos then stop
    assert int(out[0, -1]) == 7
