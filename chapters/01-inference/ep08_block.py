"""ep08 — Transformer block (pre-norm residual sandwich).

Run:
    python chapters/01-inference/ep08_block.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from arche import TransformerBlock, llama32_1b
from arche.config import Llama32Config


def main() -> None:
    base = llama32_1b()
    # CPU-friendly shrink of the 1B layout (keep GQA ratio 4:1).
    cfg = Llama32Config(
        name="demo-block",
        vocab_size=base.vocab_size,
        num_layers=1,
        embed_dim=256,
        intermediate_dim=512,
        num_heads=8,
        num_kv_heads=2,
        head_dim=32,
        max_seq_len=128,
        rope_theta=base.rope_theta,
        rope_scale_factor=base.rope_scale_factor,
        norm_eps=base.norm_eps,
    )
    block = TransformerBlock(cfg)
    x = torch.randn(1, 5, cfg.embed_dim)
    y = block(x)

    print("TransformerBlock")
    print("  flow: x → RMSNorm → Attn → +x → RMSNorm → SwiGLU → +x")
    print(f"  input  shape = {tuple(x.shape)}")
    print(f"  output shape = {tuple(y.shape)}")
    print(f"  residual preserves shape so blocks can stack ({base.num_layers}× in Llama 3.2 1B)")


if __name__ == "__main__":
    main()
