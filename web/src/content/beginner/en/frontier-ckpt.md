## Idea

Frontier modules (MoE, MLA) need the same discipline as dense: **save and load** without silent shape bugs.

This chapter round-trips teaching MoE/MLA checkpoints and shows how **Mixtral-style** expert keys (`w1/w2/w3`) map into our module names—so reading real open checkpoints is less mysterious.

## Run

```bash
python chapters/02-training/ep20_frontier_ckpt.py
```

Expect successful save→load and a key-map demonstration.

## Next

**HF align** (`?c=hf-align`): prove dense logits match Hugging Face when weights exist.
