"""ep02 — Tokenizer (official Llama 3.2 BPE).

Run:
    huggingface-cli login   # after accepting the model license
    python chapters/01-inference/ep02_tokenizer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arche.tokenizer import DEFAULT_TOKENIZER_ID, load_tokenizer


def main() -> None:
    text = "The cat sat on the mat."
    print(f"Model tokenizer: {DEFAULT_TOKENIZER_ID}")
    print(f"Example text: {text!r}")
    print()
    try:
        tok = load_tokenizer()
    except Exception as exc:  # noqa: BLE001 — teaching script: show the real blocker
        print("Could not load the official tokenizer.")
        print(f"  {type(exc).__name__}: {exc}")
        print()
        print("Fix:")
        print("  1. Accept the license on the Hugging Face model page")
        print("  2. huggingface-cli login")
        print("  3. Optional: export HF_ENDPOINT=https://hf-mirror.com")
        sys.exit(1)

    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"token ids ({len(ids)}): {ids}")
    print(f"decoded: {decoded!r}")
    print()
    print("Special tokens (subset):")
    for name in ("bos_token", "eos_token", "pad_token"):
        value = getattr(tok, name, None)
        token_id = getattr(tok, f"{name}_id", None)
        print(f"  {name:10} = {value!r:20} id={token_id}")
    print()
    print("Note: we use the official tokenizer; later chapters focus on the dense trunk.")


if __name__ == "__main__":
    main()
