"""ep03 — Token embedding (no absolute position table).

Run:
    python chapters/01-inference/ep03_embedding.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from arche import TokenEmbedding, llama32_1b


def main() -> None:
    cfg = llama32_1b()
    # Tiny stand-in vocab so the demo runs without downloading weights.
    demo_vocab = 32
    emb = TokenEmbedding(demo_vocab, cfg.embed_dim)
    # Pretend token ids for a 3-token phrase.
    ids = torch.tensor([[3, 7, 11]])
    x = emb(ids)

    print("TokenEmbedding")
    print(f"  real Llama vocab_size = {cfg.vocab_size}")
    print(f"  demo vocab_size       = {demo_vocab}")
    print(f"  embed_dim             = {cfg.embed_dim}")
    print(f"  ids shape             = {tuple(ids.shape)}")
    print(f"  output shape          = {tuple(x.shape)}  # (batch, seq, dim)")
    print()
    print("Modern trunk note:")
    print("  Position is injected later by RoPE on Q/K, not by adding a PE table here.")


if __name__ == "__main__":
    main()
