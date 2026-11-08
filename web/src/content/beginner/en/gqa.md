## Idea

**GQA**: fewer KV heads than query heads (1B: 32 Q / 8 KV). Shrinks decode KV cache. See the interactive diagram.

Flow: project → RoPE → repeat KV → causal SDPA → out proj.

## Run

```bash
python chapters/01-inference/ep06_gqa.py
```

## Next

**SwiGLU** (`?c=swiglu`).
