"""ep19 — SFT then one DPO preference step.

Run:
    python chapters/02-training/ep19_sft_dpo.py
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
from arche.train.dpo import train_dpo_step
from arche.train.sft import train_sft_step


def main() -> None:
    torch.manual_seed(0)
    cfg = Llama32Config(
        name="toy",
        vocab_size=48,
        num_layers=1,
        embed_dim=32,
        intermediate_dim=64,
        num_heads=2,
        num_kv_heads=1,
        head_dim=16,
        max_seq_len=32,
        rope_theta=10_000.0,
        rope_scale_factor=1.0,
        norm_eps=1e-5,
    )
    policy = Llama(cfg)
    ids = torch.randint(0, 48, (2, 12))
    prompt_lens = torch.tensor([4, 5])
    sft = train_sft_step(policy, ids, prompt_lens, lr=1e-2)
    print(f"SFT loss: {sft:.4f}")

    reference = Llama(cfg)
    reference.load_state_dict(policy.state_dict())
    chosen = torch.randint(0, 48, (2, 8))
    rejected = torch.randint(0, 48, (2, 8))
    dpo = train_dpo_step(policy, reference, chosen, rejected, lr=1e-3)
    print(f"DPO loss: {dpo:.4f}")


if __name__ == "__main__":
    main()
