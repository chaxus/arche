## Idea

`TokenEmbedding`: id → vector `(vocab, dim)`. No absolute position table here—**RoPE** adds position later. Llama 3.2 ties this weight with the LM head.

## Run

```bash
python chapters/01-inference/ep03_embedding.py
```

## Next

**RMSNorm** (`?c=rmsnorm`).
