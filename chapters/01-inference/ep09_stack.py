"""ep09 — Stack blocks + final norm + LM head.

Run:
    python chapters/01-inference/ep09_stack.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from arche import llama32_1b
from arche.config import Llama32Config
from arche.model import Llama


def main() -> None:
    base = llama32_1b()
    cfg = Llama32Config(
        name="demo-stack",
        vocab_size=256,
        num_layers=2,
        embed_dim=128,
        intermediate_dim=256,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        max_seq_len=64,
        rope_theta=base.rope_theta,
        rope_scale_factor=base.rope_scale_factor,
        norm_eps=base.norm_eps,
        tie_word_embeddings=True,
    )
    model = Llama(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    logits, caches = model(ids)

    n_params = sum(p.numel() for p in model.parameters())
    print("Llama stack")
    print(f"  real 1B layers = {base.num_layers}, embed_dim = {base.embed_dim}")
    print(f"  this demo      = {cfg.num_layers} layers, dim {cfg.embed_dim}")
    print(f"  input_ids      = {tuple(ids.shape)}")
    print(f"  logits         = {tuple(logits.shape)}  # (batch, seq, vocab)")
    print(f"  caches         = {len(caches)} layers, seq={caches[0].seq_len}")
    print(f"  tied embed/lm  = {model.lm_head.weight is model.embed_tokens.weight}")
    print(f"  parameters     = {n_params:,}")
    print()
    print("Flow: embed → [block × N] → RMSNorm → lm_head")


if __name__ == "__main__":
    main()
