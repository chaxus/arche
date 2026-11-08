"""ep17 — Simplified continuous batching scheduler.

Run:
    python chapters/01-inference/ep17_continuous_batching.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arche.batching import DecodeRequest, generate_continuous
from arche.config import Llama32Config
from arche.model import Llama


def main() -> None:
    cfg = Llama32Config(
        name="demo",
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
    reqs = [
        DecodeRequest("job-a", [1, 2, 3], max_new_tokens=2, temperature=0.0),
        DecodeRequest("job-b", [7, 8], max_new_tokens=4, temperature=0.0),
        DecodeRequest("job-c", [9], max_new_tokens=1, temperature=0.0),
    ]
    out = generate_continuous(model, reqs)
    print("continuous batching")
    print(f"  steps     = {out.steps}")
    for rid, tokens in out.outputs.items():
        print(f"  {rid}: len={len(tokens)} ids={tokens}")
    print()
    print("Production engines add paged KV and CUDA graphs; this chapter only teaches the scheduler shape.")


if __name__ == "__main__":
    main()
