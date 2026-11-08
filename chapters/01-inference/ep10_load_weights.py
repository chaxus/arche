"""ep10 — Load official Llama 3.2 weights into the hand-written model.

Run:
    huggingface-cli login
    python chapters/01-inference/ep10_load_weights.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from arche.load_weights import DEFAULT_MODEL_ID, try_from_pretrained


def main() -> None:
    print(f"Loading {DEFAULT_MODEL_ID} into arche.Llama …")
    try:
        model = try_from_pretrained(DEFAULT_MODEL_ID, device="cpu", dtype=torch.float32)
    except RuntimeError as exc:
        print(exc)
        sys.exit(1)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  layers     = {model.cfg.num_layers}")
    print(f"  embed_dim  = {model.cfg.embed_dim}")
    print(f"  parameters = {n_params:,}")
    print(f"  tied head  = {model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()}")

    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        logits, _ = model(ids)
    print(f"  smoke logits shape = {tuple(logits.shape)}")
    print()
    print("Next: sample tokens from these logits (ep11).")


if __name__ == "__main__":
    main()
