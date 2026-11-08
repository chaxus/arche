## Idea

`from_pretrained` builds our graph and copies HF Llama 3.2 tensors via a key map. Needs gated HF login.

## Run

```bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
```

## Next

**Generate** (`?c=generate`).
