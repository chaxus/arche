## Idea

Token ids are still just integers. The model needs **numbers it can multiply**—a vector per id.

**Embedding** is a lookup table of shape `(vocab_size, dim)`: row `i` is the vector for token id `i`. One forward = gather those rows. No “understanding” happens here yet; it is the first learnable map from symbols to geometry.

Important for Llama 3.2–style models:

- We do **not** add a separate absolute position table on top of the embedding. **Position comes later via RoPE** on Q/K.
- Often the same table is **tied** with the final LM head (same weights for “id → vector” and “vector → vocab scores”).

The lab uses a tiny fake vocab so it runs without downloading weights; shapes match the real idea: `(batch, seq) → (batch, seq, dim)`.

## Run

```bash
python chapters/01-inference/ep03_embedding.py
```

Expect shapes for ids and output, plus a note that position is RoPE’s job.

## Next

**RMSNorm** (`?c=rmsnorm`): stabilize vectors before the heavy layers.
