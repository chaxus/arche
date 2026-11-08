"""ep07 — SwiGLU MLP.

Run:
    python chapters/01-inference/ep07_swiglu.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from arche import SwiGLU, llama32_1b


def main() -> None:
    cfg = llama32_1b()
    mlp = SwiGLU(cfg.embed_dim, cfg.intermediate_dim)
    x = torch.randn(1, 3, cfg.embed_dim)
    y = mlp(x)

    print("SwiGLU")
    print(f"  embed_dim        = {cfg.embed_dim}")
    print(f"  intermediate_dim = {cfg.intermediate_dim}")
    print(f"  input  shape     = {tuple(x.shape)}")
    print(f"  output shape     = {tuple(y.shape)}")
    n_params = sum(p.numel() for p in mlp.parameters())
    print(f"  parameters       = {n_params:,}")
    print()
    print("Formula: down(silu(gate(x)) * up(x))  — three bias-free linears.")


if __name__ == "__main__":
    main()
