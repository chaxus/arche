"""ep01 — Overview: modern dense trunk, not GPT-2 archaeology.

If you are new: activate `.venv`, `cd` to the repo root, then run this file.
You should see config numbers (layers, heads, dim)—not a chatbot.

Run:
    python chapters/01-inference/ep01_overview.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arche import llama32_1b, llama32_3b


def main() -> None:
    cfg = llama32_1b()
    print("arche — ἀρχή (first principle)")
    print("Target architecture: Llama 3.2 dense decoder")
    print()
    print(f"Config: {cfg.name}")
    print(f"  vocab_size         = {cfg.vocab_size}")
    print(f"  num_layers         = {cfg.num_layers}")
    print(f"  embed_dim          = {cfg.embed_dim}")
    print(f"  intermediate_dim   = {cfg.intermediate_dim}")
    print(f"  num_heads          = {cfg.num_heads}")
    print(f"  num_kv_heads       = {cfg.num_kv_heads}  (GQA group size {cfg.num_query_groups})")
    print(f"  head_dim           = {cfg.head_dim}")
    print(f"  rope_theta         = {cfg.rope_theta:g}")
    print(f"  rope_scale_factor   = {cfg.rope_scale_factor:g}  (long-context; later chapter)")
    print(f"  max_seq_len        = {cfg.max_seq_len}")
    print()
    print("Curriculum:")
    for step in (
        "tokenizer → embedding → pretrain (objective/data/optim/scale)",
        "RMSNorm → RoPE → GQA → SwiGLU → block → stack → weights → generate → HF",
        "SFT/DPO → KV → RoPE scaling → serve → MoE → MLA → shards",
    ):
        print(f"  - {step}")
    print()
    big = llama32_3b()
    print(f"Same lineage, larger sibling: {big.name} ({big.num_layers} layers, dim {big.embed_dim})")



if __name__ == "__main__":
    main()
