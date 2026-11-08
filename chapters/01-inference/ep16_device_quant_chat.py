"""ep16 — Device selection, INT8 weights, chat templates.

Run:
    python chapters/01-inference/ep16_device_quant_chat.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from arche.chat import apply_chat_template
from arche.config import Llama32Config
from arche.device import describe_device, pick_device
from arche.model import Llama
from arche.quantize import quantize_llama_int8


def main() -> None:
    device = pick_device()
    print(f"device: {describe_device(device)}")
    cfg = Llama32Config(
        name="demo",
        vocab_size=128,
        num_layers=1,
        embed_dim=64,
        intermediate_dim=128,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        max_seq_len=64,
        rope_theta=10_000.0,
        rope_scale_factor=1.0,
        norm_eps=1e-5,
    )
    model = Llama(cfg).to(device).eval()
    qmodel = quantize_llama_int8(model)
    ids = torch.randint(0, 128, (1, 4), device=device)
    with torch.no_grad():
        logits, _ = qmodel(ids)
    print(f"int8 logits shape: {tuple(logits.shape)}")

    class _Tok:
        chat_template = None

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return apply_chat_template(self, messages, add_generation_prompt=add_generation_prompt)

        def encode(self, text: str):
            return list(range(min(8, len(text))))

    text = apply_chat_template(
        _Tok(),
        [{"role": "user", "content": "Explain RoPE in one line."}],
    )
    print("chat preview:")
    print(text)


if __name__ == "__main__":
    main()
