"""ep04 — RMSNorm (modern pre-norm).

Run:
    python chapters/01-inference/ep04_rmsnorm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from arche import llama32_1b
from arche.norms import RMSNorm


def main() -> None:
    cfg = llama32_1b()
    torch.manual_seed(0)

    # Tiny teaching shape: batch=1, seq=3 ("猫 / 吃 / 鱼" length), dim=embed_dim
    # Full 2048-d is fine on CPU for a single forward.
    x = torch.randn(1, 3, cfg.embed_dim)
    norm = RMSNorm(cfg.embed_dim, eps=cfg.norm_eps)

    y = norm(x)
    # RMS along last dim should be ~1 before weight scale (weight starts as ones).
    rms = y.float().pow(2).mean(dim=-1).sqrt()

    print("RMSNorm")
    print(f"  input  shape: {tuple(x.shape)}")
    print(f"  output shape: {tuple(y.shape)}")
    print(f"  weight shape: {tuple(norm.weight.shape)}")
    print(f"  per-token RMS after norm (expect ~1): {rms.squeeze(0).tolist()}")
    print()
    print("Why RMSNorm (not LayerNorm) on the modern trunk:")
    print("  - no mean centering → cheaper, stable enough in practice")
    print("  - Llama / Qwen dense stacks use it before attention and MLP")


if __name__ == "__main__":
    main()
