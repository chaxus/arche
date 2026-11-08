## Idea

Loop: take last logits → (optional temperature / top-k) → sample or greedy → append → stop on EOS/length.

## Run

```bash
python chapters/01-inference/ep11_generate.py
```

## Next

**KV cache** (`?c=kv-cache`).
