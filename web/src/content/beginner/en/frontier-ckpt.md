## Idea

### 1. Why

MoE/MLA modules still need save/load without silent shape bugs.

### 2. What

Round-trip teaching checkpoints; map Mixtral-style `w1/w2/w3` expert keys into our names.

### 3–6. What we do

Primer for reading open MoE checkpoints—not a full trainer.

## Run

```bash
python chapters/02-training/ep20_frontier_ckpt.py
```

## Next

**HF align** (`?c=hf-align`).
