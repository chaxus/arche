## Idea

Trust but verify: after `from_pretrained`, run the **same short prompt** through arche and Hugging Face Llama 3.2; compare **argmax** and **max absolute logit difference**.

If gated weights are missing, the chapter exits cleanly (CI still has a synthetic mapper test). This is your “did we wire the dense trunk correctly?” gate.

## Run

```bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
```

Expect a small diff report when weights are available.

## Fix

No weights / no login → clean skip or Fix text from the script; not a logic failure of arche itself.

## Next

**Sharded MoE/MLA** (`?c=sharded-frontier`): how big checkpoints are split on disk.
