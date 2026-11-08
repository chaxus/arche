"""ep22 — Sharded Mixtral MoE + DeepSeek MLA loaders (synthetic shards).

Run:
    python chapters/02-training/ep22_sharded_frontier.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from arche.load_deepseek_mla import load_deepseek_mla_shards, write_deepseek_mla_shards
from arche.load_mixtral import load_mixtral_shards_into_moe, write_mixtral_shards
from arche.mla import MultiHeadLatentAttention
from arche.moe import MoESwiGLU


def main() -> None:
    moe = MoESwiGLU(16, 32, num_experts=2, top_k=1)
    mla = MultiHeadLatentAttention(32, 2, 8, kv_lora_rank=8, rope_dim=8, max_seq_len=16)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        moe_dir = root / "mixtral"
        mla_dir = root / "deepseek"
        write_mixtral_shards(moe_dir, moe, shard_size=2)
        write_deepseek_mla_shards(mla_dir, mla, shard_size=2)
        print(f"wrote Mixtral shards under {moe_dir.name}/ ({len(list(moe_dir.glob('*.pt')))} files)")
        print(f"wrote DeepSeek MLA shards under {mla_dir.name}/ ({len(list(mla_dir.glob('*.pt')))} files)")

        moe2 = MoESwiGLU(16, 32, num_experts=2, top_k=1)
        mla2 = MultiHeadLatentAttention(32, 2, 8, kv_lora_rank=8, rope_dim=8, max_seq_len=16)
        with torch.no_grad():
            for p in list(moe2.parameters()) + list(mla2.parameters()):
                p.zero_()
        load_mixtral_shards_into_moe(moe_dir, moe2)
        load_deepseek_mla_shards(mla_dir, mla2)

        x_moe = torch.randn(1, 3, 16)
        x_mla = torch.randn(1, 4, 32)
        with torch.no_grad():
            a, _ = moe(x_moe)
            b, _ = moe2(x_moe)
            c, _ = mla(x_mla)
            d, _ = mla2(x_mla)
        assert torch.allclose(a, b, atol=1e-5)
        assert torch.allclose(c, d, atol=1e-5)
        print("sharded Mixtral MoE load: allclose ok")
        print("sharded DeepSeek MLA load: allclose ok")

    print()
    print("Real gated Mixtral/DeepSeek repos use safetensors indexes; this chapter teaches the same shape.")


if __name__ == "__main__":
    main()
