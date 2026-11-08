"""ep20 — MoE/MLA checkpoint roundtrip + Mixtral-shaped key map.

Run:
    python chapters/02-training/ep20_frontier_ckpt.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from arche.checkpoint import load_checkpoint, save_checkpoint
from arche.load_mixtral import load_mixtral_shaped_state
from arche.mla import MultiHeadLatentAttention
from arche.moe import MoESwiGLU


def main() -> None:
    moe = MoESwiGLU(16, 32, num_experts=2, top_k=1)
    hf_like = {
        "model.layers.0.block_sparse_moe.gate.weight": moe.gate.weight.detach().clone(),
        "model.layers.0.block_sparse_moe.experts.0.w1.weight": moe.experts[0].gate_proj.weight.detach().clone(),
        "model.layers.0.block_sparse_moe.experts.0.w3.weight": moe.experts[0].up_proj.weight.detach().clone(),
        "model.layers.0.block_sparse_moe.experts.0.w2.weight": moe.experts[0].down_proj.weight.detach().clone(),
        "model.layers.0.block_sparse_moe.experts.1.w1.weight": moe.experts[1].gate_proj.weight.detach().clone(),
        "model.layers.0.block_sparse_moe.experts.1.w3.weight": moe.experts[1].up_proj.weight.detach().clone(),
        "model.layers.0.block_sparse_moe.experts.1.w2.weight": moe.experts[1].down_proj.weight.detach().clone(),
    }
    loaded = MoESwiGLU(16, 32, num_experts=2, top_k=1)
    load_mixtral_shaped_state(loaded, hf_like)
    print("Mixtral-shaped MoE keys loaded into arche.MoESwiGLU")

    mla = MultiHeadLatentAttention(32, 2, 8, kv_lora_rank=8, rope_dim=8, max_seq_len=16)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mla.pt"
        save_checkpoint(path, mla, meta={"kind": "mla"})
        again = MultiHeadLatentAttention(32, 2, 8, kv_lora_rank=8, rope_dim=8, max_seq_len=16)
        meta = load_checkpoint(path, again)
        print(f"MLA checkpoint roundtrip ok meta={meta}")

    print()
    print("Full DeepSeek/Mixtral production loaders still need shard layouts + custom kernels;")
    print("this chapter locks the teaching interfaces and key maps you extend next.")


if __name__ == "__main__":
    main()
