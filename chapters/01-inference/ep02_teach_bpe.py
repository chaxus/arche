"""ep02 companion — train a tiny BPE table (no Hugging Face).

Run:
    python chapters/01-inference/ep02_teach_bpe.py

Shows: how a tokenizer *table* is generated, encode/decode, fertility.
Official Llama table is still ep02_tokenizer.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arche.teach_bpe import decode, encode, fertility, train_bpe


def main() -> None:
    # Small, repetitive corpus so merges are visible — not a real multilingual mix.
    corpus = [
        "the cat sat on the mat",
        "the cat sat",
        "the dog sat on the mat",
        "a cat and a dog",
        "on the mat",
    ]
    # Keep merges modest so the demo sentence stays multi-token (tiny corpora over-merge easily).
    bpe = train_bpe(corpus, num_merges=12)
    text = "the cat sat on the mat"
    text_b = "the dog ran"
    ids = encode(bpe, text)
    back = decode(bpe, ids)
    ids_b = encode(bpe, text_b)

    print("Teach BPE (byte-level, tiny corpus)")
    print(f"  base bytes     = 256")
    print(f"  merges         = {len(bpe.merges)}")
    print(f"  vocab_size     = {bpe.vocab_size}")
    print(f"  text           = {text!r}")
    print(f"  token ids      = {ids}")
    print(f"  n_tokens       = {len(ids)}")
    print(f"  decoded        = {back!r}")
    print(f"  fertility≈     = {fertility(bpe, text):.2f}  tokens / whitespace word")
    print(f"  OOV-ish text   = {text_b!r} → {len(ids_b)} tokens {ids_b}")
    print()
    print("What “good” means here (teaching):")
    print("  - encode→decode round-trips on in-corpus text")
    print("  - text close to the train corpus compresses more (lower fertility)")
    print("  - unseen words fall back toward bytes → more tokens")
    print("  - this table ≠ Llama; never mix with Llama weights")


if __name__ == "__main__":
    main()
