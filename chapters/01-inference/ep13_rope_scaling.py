"""ep13 — Llama 3.2 RoPE frequency scaling for long context.

Run:
    python chapters/01-inference/ep13_rope_scaling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arche import llama32_1b
from arche.rope import llama3_inv_freq


def main() -> None:
    cfg = llama32_1b()
    plain = llama3_inv_freq(
        cfg.head_dim,
        cfg.rope_theta,
        scale_factor=1.0,
        low_freq_factor=cfg.rope_low_freq_factor,
        high_freq_factor=cfg.rope_high_freq_factor,
        original_max_position=cfg.rope_original_max_position,
        device=__import__("torch").device("cpu"),
    )
    scaled = llama3_inv_freq(
        cfg.head_dim,
        cfg.rope_theta,
        scale_factor=cfg.rope_scale_factor,
        low_freq_factor=cfg.rope_low_freq_factor,
        high_freq_factor=cfg.rope_high_freq_factor,
        original_max_position=cfg.rope_original_max_position,
        device=__import__("torch").device("cpu"),
    )

    print("Llama 3.2 RoPE scaling")
    print(f"  rope_theta              = {cfg.rope_theta:g}")
    print(f"  scale_factor             = {cfg.rope_scale_factor:g}")
    print(f"  original_max_position   = {cfg.rope_original_max_position}")
    print(f"  max_seq_len (config)    = {cfg.max_seq_len}")
    print(f"  inv_freq[0] plain/scaled= {float(plain[0]):.6e} / {float(scaled[0]):.6e}")
    print(f"  inv_freq[-1] plain/scaled={float(plain[-1]):.6e} / {float(scaled[-1]):.6e}")
    print()
    print("Long wavelengths are divided by factor; short ones stay put; the middle band is smoothed.")
    print("This is what lets Llama 3.2 claim 128k context from an 8k pretrain window.")


if __name__ == "__main__":
    main()
