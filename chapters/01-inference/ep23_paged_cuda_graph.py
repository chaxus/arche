"""ep23 — Paged KV pool + optional CUDA graph decode step.

Run:
    python chapters/01-inference/ep23_paged_cuda_graph.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from arche.config import Llama32Config
from arche.cuda_graphs import cuda_available, generate_with_optional_graph
from arche.model import Llama
from arche.paged_kv import BlockTable, PagedKVPool, append_sequence, gather_kv


def main() -> None:
    pool = PagedKVPool.create(num_pages=4, page_size=2, num_kv_heads=1, head_dim=4)
    table = BlockTable()
    k = torch.randn(1, 1, 3, 4)
    v = torch.randn(1, 1, 3, 4)
    append_sequence(pool, table, k, v)
    gathered = gather_kv(pool, table)
    assert gathered.seq_len == 3
    assert torch.allclose(gathered.k, k)
    assert torch.allclose(gathered.v, v)
    print(f"paged KV: length={table.length} pages={table.pages} free={len(pool.free)}")

    cfg = Llama32Config(
        name="tiny",
        vocab_size=64,
        num_layers=1,
        embed_dim=32,
        intermediate_dim=64,
        num_heads=2,
        num_kv_heads=1,
        head_dim=16,
        max_seq_len=64,
        rope_theta=10_000.0,
        rope_scale_factor=1.0,
        norm_eps=1e-5,
    )
    model = Llama(cfg).eval()
    ids = torch.randint(0, 64, (1, 4))
    out = generate_with_optional_graph(model, ids, max_new_tokens=3, use_cuda_graph=True)
    print(f"generate_with_optional_graph → shape {tuple(out.shape)} cuda={cuda_available()}")
    print()
    print("Paged attention gathers contiguous KV for SDPA; CUDA graphs capture one fixed decode step.")
    print("Production engines fuse both with slot mapping — this chapter separates the two ideas.")


if __name__ == "__main__":
    main()
