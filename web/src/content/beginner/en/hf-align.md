## Idea

With gated weights: compare arche vs HF logits (argmax + max-abs). Offline → chapter exits cleanly; CI keeps synthetic mapper test.

## Run

```bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
```

## Next

**Sharded MoE/MLA** (`?c=sharded-frontier`).
