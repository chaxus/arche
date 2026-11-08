"""ep14 — Mixture of Experts (sparse SwiGLU).

Run:
    python chapters/01-inference/ep14_moe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from arche.moe import MoESwiGLU
from arche.mlp import SwiGLU


def main() -> None:
    embed_dim, intermediate_dim = 64, 128
    moe = MoESwiGLU(embed_dim, intermediate_dim, num_experts=4, top_k=2)
    dense = SwiGLU(embed_dim, intermediate_dim)
    x = torch.randn(1, 5, embed_dim)
    y, probs = moe(x)

    moe_params = sum(p.numel() for p in moe.parameters())
    dense_params = sum(p.numel() for p in dense.parameters())
    print("MoESwiGLU")
    print(f"  experts / top_k     = {moe.num_experts} / {moe.top_k}")
    print(f"  input / output      = {tuple(x.shape)} → {tuple(y.shape)}")
    print(f"  router probs shape  = {tuple(probs.shape)}")
    print(f"  MoE parameters      = {moe_params:,}")
    print(f"  one dense SwiGLU    = {dense_params:,}")
    print(f"  active experts/tok  ≈ {moe.top_k} (sparse compute)")
    print()
    print("Frontier models (Mixtral, DeepSeek) scale capacity with many experts while keeping per-token FLOPs near top_k.")


if __name__ == "__main__":
    main()
