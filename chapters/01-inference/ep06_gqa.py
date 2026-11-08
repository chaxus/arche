"""ep06 — Grouped-Query Attention (causal + RoPE).

Run:
    python chapters/01-inference/ep06_gqa.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from arche import CausalSelfAttention, llama32_1b


def main() -> None:
    cfg = llama32_1b()
    # Shrink dims so the chapter runs fast on CPU while keeping GQA ratios.
    num_heads, num_kv_heads, head_dim = 8, 2, 32
    embed_dim = num_heads * head_dim
    attn = CausalSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rope_theta=cfg.rope_theta,
        max_seq_len=64,
    )
    x = torch.randn(1, 4, embed_dim)
    y, cache = attn(x)

    print("CausalSelfAttention (GQA)")
    print(f"  Llama 3.2 1B: heads={cfg.num_heads}, kv_heads={cfg.num_kv_heads}, head_dim={cfg.head_dim}")
    print(f"  this demo:    heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}")
    print(f"  n_rep (Q per KV head) = {num_heads // num_kv_heads}")
    print(f"  input  shape = {tuple(x.shape)}")
    print(f"  output shape = {tuple(y.shape)}")
    print(f"  cache seq    = {cache.seq_len}")
    print()
    print("Why GQA: fewer KV heads → smaller KV cache at decode time, quality close to full MHA.")


if __name__ == "__main__":
    main()
