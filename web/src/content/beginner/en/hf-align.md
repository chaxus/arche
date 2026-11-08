## Idea

### 1. Why

Trust but verify the dense trunk wiring.

### 2. What

Same short prompt through arche vs HF Llama 3.2; compare argmax and max-abs logit diff. No weights → clean exit.

### 3–6. What we do

Chapter script + CI synthetic mapper test.

## Run

```bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
```

## Fix

Missing weights/login → skip/hints, not necessarily a logic bug.

## Next

**Sharded MoE/MLA** (`?c=sharded-frontier`).
