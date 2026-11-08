import torch

from arche.batching import DecodeRequest, generate_continuous
from arche.chat import apply_chat_template, encode_chat
from arche.config import Llama32Config
from arche.device import describe_device, pick_device
from arche.model import Llama
from arche.quantize import quantize_llama_int8


def _tiny() -> Llama:
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
    return Llama(cfg)


def test_pick_device_cpu():
    assert pick_device("cpu").type == "cpu"
    assert "cpu" in describe_device(torch.device("cpu"))


def test_int8_quant_runs_and_keeps_shape():
    model = _tiny().eval()
    q = quantize_llama_int8(model)
    ids = torch.randint(0, 64, (1, 4))
    with torch.no_grad():
        a, _ = model(ids)
        b, _ = q(ids)
    assert a.shape == b.shape


def test_chat_template_fallback():
    class Tok:
        def encode(self, text: str) -> list[int]:
            return [ord(c) % 50 for c in text[:8]]

    messages = [{"role": "user", "content": "hi"}]
    text = apply_chat_template(Tok(), messages)
    assert "user" in text and "hi" in text
    assert encode_chat(Tok(), messages)


def test_continuous_batching_finishes_requests():
    model = _tiny().eval()
    reqs = [
        DecodeRequest("a", [1, 2, 3], max_new_tokens=2, temperature=0.0),
        DecodeRequest("b", [4, 5], max_new_tokens=3, temperature=0.0),
    ]
    out = generate_continuous(model, reqs)
    assert set(out.outputs) == {"a", "b"}
    assert len(out.outputs["a"]) == 5
    assert len(out.outputs["b"]) == 5
    assert out.steps >= 2
