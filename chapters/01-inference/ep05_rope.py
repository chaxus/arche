"""ep05 — RoPE (rotary position embedding).

Run:
    python chapters/01-inference/ep05_rope.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from arche import llama32_1b
from arche.rope import apply_rotary_emb, build_rope_cache


def main() -> None:
    cfg = llama32_1b()
    torch.manual_seed(0)

    seq_len = 4
    # One query head worth of features: (batch, heads, seq, head_dim)
    q = torch.randn(1, 1, seq_len, cfg.head_dim)
    cos, sin = build_rope_cache(
        seq_len=seq_len,
        head_dim=cfg.head_dim,
        theta=cfg.rope_theta,
        dtype=q.dtype,
    )
    q_rot = apply_rotary_emb(q, cos, sin)

    print("RoPE")
    print(f"  head_dim   = {cfg.head_dim}")
    print(f"  rope_theta = {cfg.rope_theta:g}")
    print(f"  q shape    = {tuple(q.shape)}")
    print(f"  cos/sin    = {tuple(cos.shape)}")
    print(f"  rotated q  = {tuple(q_rot.shape)}")
    print()
    # Relative-position sanity: rotating two positions changes their dot product
    # in a position-dependent way (absolute PE cannot do this as cleanly).
    dots_before = torch.einsum("bhsd,bhtd->bhst", q, q)[0, 0]
    dots_after = torch.einsum("bhsd,bhtd->bhst", q_rot, q_rot)[0, 0]
    print("self-dot matrix before RoPE (seq x seq):")
    print(dots_before)
    print("self-dot matrix after RoPE:")
    print(dots_after)
    print()
    print("Next: GQA attention will call RoPE on Q and K before the matmul.")
    print(f"Long-context scale_factor={cfg.rope_scale_factor:g} is deferred to a later chapter.")


if __name__ == "__main__":
    main()
