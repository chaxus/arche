"""ep15 — Multi-head Latent Attention (compressed KV).

Run:
    python chapters/01-inference/ep15_mla.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from arche.mla import MultiHeadLatentAttention


def main() -> None:
    mla = MultiHeadLatentAttention(
        embed_dim=64,
        num_heads=4,
        head_dim=12,
        kv_lora_rank=16,
        q_lora_rank=32,
        rope_dim=8,
        max_seq_len=64,
    )
    x = torch.randn(1, 6, 64)
    y, cache = mla(x)

    full_kv = 2 * mla.num_heads * cache.seq_len * mla.head_dim
    latent = cache.kv_latent.numel() + cache.k_rope.numel()
    print("MultiHeadLatentAttention (MLA)")
    print(f"  input / output     = {tuple(x.shape)} → {tuple(y.shape)}")
    print(f"  kv_lora_rank       = {mla.kv_lora_rank}")
    print(f"  rope_dim           = {mla.rope_dim}")
    print(f"  cache kv_latent    = {tuple(cache.kv_latent.shape)}")
    print(f"  cache k_rope       = {tuple(cache.k_rope.shape)}")
    print(f"  elems full K+V     ≈ {full_kv:,}")
    print(f"  elems MLA cache    = {latent:,}  ({100 * latent / full_kv:.1f}% of full KV)")
    print()
    print("DeepSeek-style MLA caches a low-rank latent instead of every head's K/V — that is the decode-memory win.")


if __name__ == "__main__":
    main()
