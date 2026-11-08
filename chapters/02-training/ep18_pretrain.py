"""ep18 — Tiny pretraining loop.

Run:
    python chapters/02-training/ep18_pretrain.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from arche.config import Llama32Config
from arche.model import Llama
from arche.train.pretrain import train_pretrain_steps


def main() -> None:
    torch.manual_seed(0)
    cfg = Llama32Config(
        name="toy",
        vocab_size=64,
        num_layers=1,
        embed_dim=64,
        intermediate_dim=128,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        max_seq_len=32,
        rope_theta=10_000.0,
        rope_scale_factor=1.0,
        norm_eps=1e-5,
    )
    model = Llama(cfg)
    batches = [torch.randint(0, 64, (4, 16)) for _ in range(6)]
    history = train_pretrain_steps(model, batches, lr=1e-2)
    print("pretrain")
    print(f"  steps = {len(history)}")
    print(f"  loss  = {history[0]:.4f} → {history[-1]:.4f}")


if __name__ == "__main__":
    main()
