"""ep12 — KV cache: prefill once, decode token-by-token.

Run:
    python chapters/01-inference/ep12_kv_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from arche.cache import KVCache
from arche.config import Llama32Config
from arche.generate import generate
from arche.model import Llama


def main() -> None:
    cfg = Llama32Config(
        name="demo-cache",
        vocab_size=128,
        num_layers=2,
        embed_dim=64,
        intermediate_dim=128,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        max_seq_len=64,
        rope_theta=10_000.0,
        rope_scale_factor=1.0,
        norm_eps=1e-5,
    )
    torch.manual_seed(0)
    model = Llama(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 6))

    caches: list[KVCache | None] = [None] * cfg.num_layers
    logits, caches = model(prompt, caches=caches, start_pos=0)
    print("KV cache")
    print(f"  prefill ids     = {tuple(prompt.shape)}")
    print(f"  prefill logits  = {tuple(logits.shape)}")
    print(f"  cache seq/layer = {[c.seq_len for c in caches]}")

    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    step_logits, caches = model(next_id, caches=caches, start_pos=prompt.shape[1])
    print(f"  decode step     = {tuple(next_id.shape)} → logits {tuple(step_logits.shape)}")
    print(f"  cache after     = {[c.seq_len for c in caches]}")

    out = generate(model, prompt, max_new_tokens=4, temperature=0.0, use_cache=True)
    print(f"  generate(+cache)= {tuple(out.shape)}")
    print()
    print("Without a cache, every step re-runs the full prefix. With a cache, only the new token is projected.")


if __name__ == "__main__":
    main()
