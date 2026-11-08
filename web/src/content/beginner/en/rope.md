## Idea

Attention compares queries and keys. Without position, “A then B” and “B then A” look the same.

**RoPE** (Rotary Position Embedding) rotates each Q/K vector by an angle that depends on its **position**. Relative distance then shows up naturally in the dot product—no big learned position table glued onto embeddings.

Llama 3.2 uses a large base frequency (`rope_theta`) so long contexts stay usable; later we will see **wavelength scaling** for even longer prompts.

On this page: drag the interactive picture. In code: build cos/sin caches, `rotate_half`, then `q' = q·cos + rotate(q)·sin` (same for k).

## Run

```bash
python chapters/01-inference/ep05_rope.py
```

Expect a short demo of applying RoPE and checking shapes.

## Next

**GQA** (`?c=gqa`): attention with fewer KV heads.
