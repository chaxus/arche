"""Tiny byte-level BPE for teaching — train a vocab, then encode/decode.

Not Llama's tokenizer. Same *idea* as production BPE: start from bytes,
merge the most frequent adjacent pair until you hit a target vocab size.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass
class TeachBPE:
    """piece id ↔ bytes, plus ordered merges used at encode time."""

    id_to_piece: list[bytes]
    merges: list[tuple[bytes, bytes]]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_piece)

    def piece_to_id(self) -> dict[bytes, int]:
        return {p: i for i, p in enumerate(self.id_to_piece)}


def _pair_key(a: bytes, b: bytes) -> tuple[bytes, bytes]:
    return (a, b)


def train_bpe(texts: list[str], num_merges: int = 20) -> TeachBPE:
    """Train on UTF-8 bytes. Base vocab = 256 bytes; then ``num_merges`` merges."""
    id_to_piece: list[bytes] = [bytes([i]) for i in range(256)]
    piece_to_id = {p: i for i, p in enumerate(id_to_piece)}
    merges: list[tuple[bytes, bytes]] = []

    corpus: list[list[bytes]] = [[bytes([b]) for b in t.encode("utf-8")] for t in texts]

    for _ in range(num_merges):
        counts: Counter[tuple[bytes, bytes]] = Counter()
        for seq in corpus:
            for i in range(len(seq) - 1):
                counts[_pair_key(seq[i], seq[i + 1])] += 1
        if not counts:
            break
        (a, b), _freq = counts.most_common(1)[0]
        merged = a + b
        if merged not in piece_to_id:
            piece_to_id[merged] = len(id_to_piece)
            id_to_piece.append(merged)
        merges.append((a, b))

        new_corpus: list[list[bytes]] = []
        for seq in corpus:
            out: list[bytes] = []
            i = 0
            while i < len(seq):
                if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            new_corpus.append(out)
        corpus = new_corpus

    return TeachBPE(id_to_piece=id_to_piece, merges=merges)


def encode(bpe: TeachBPE, text: str) -> list[int]:
    parts: list[bytes] = [bytes([b]) for b in text.encode("utf-8")]
    for a, b in bpe.merges:
        merged = a + b
        next_parts: list[bytes] = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and parts[i] == a and parts[i + 1] == b:
                next_parts.append(merged)
                i += 2
            else:
                next_parts.append(parts[i])
                i += 1
        parts = next_parts
    p2i = bpe.piece_to_id()
    return [p2i[p] for p in parts]


def decode(bpe: TeachBPE, ids: list[int]) -> str:
    raw = b"".join(bpe.id_to_piece[i] for i in ids)
    return raw.decode("utf-8", errors="replace")


def fertility(bpe: TeachBPE, text: str) -> float:
    """Tokens per whitespace-split word (rough teaching metric; lower = denser)."""
    words = text.split()
    if not words:
        return 0.0
    return len(encode(bpe, text)) / len(words)
