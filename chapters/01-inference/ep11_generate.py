"""ep11 — Sample text from the hand-written Llama + official weights.

Run:
    huggingface-cli login
    python chapters/01-inference/ep11_generate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from arche.generate import generate
from arche.load_weights import DEFAULT_MODEL_ID, try_from_pretrained
from arche.tokenizer import load_tokenizer


def main() -> None:
    prompt = "The future of artificial intelligence is"
    print(f"Model: {DEFAULT_MODEL_ID}")
    print(f"Prompt: {prompt!r}")
    try:
        tok = load_tokenizer(DEFAULT_MODEL_ID)
        model = try_from_pretrained(DEFAULT_MODEL_ID, device="cpu", dtype=torch.float32)
    except Exception as exc:  # noqa: BLE001
        print(exc)
        print("Accept the HF license, `huggingface-cli login`, then retry.")
        sys.exit(1)

    ids = torch.tensor([tok.encode(prompt)])
    out = generate(model, ids, max_new_tokens=40, temperature=0.8, top_k=50)
    text = tok.decode(out[0].tolist())
    print()
    print(text)


if __name__ == "__main__":
    main()
