## Idea

This chapter **loads** an open Llama 3.2 checkpoint into your handwritten stack: safetensors / HF names land in arche’s `state_dict` through a **key map**. Random weights mean a correct graph with nonsense text—official floats turn the same forward into a language model right away. You already saw how pretrain *writes* those floats; here you **borrow** a finished run so generate and HF align work without your own cluster. Accept the license, download, map keys, pair the **official tokenizer**, and sanity-check shapes.

### Entry point

[`from_pretrained`](https://github.com/chaxus/arche/blob/main/arche/load_weights.py) builds `Llama(cfg)` and copies HF tensors into matching names.

| Path | Role |
| --- | --- |
| **Load (this chapter)** | Example checkpoint → speak immediately |
| **Pretrain (earlier)** | How those floats get written from zero |
| **HF align (next-next)** | Proof map + formulas match HF logits |

### Mapping pitfalls

| Risk | Effect |
| --- | --- |
| Wrong key / transpose | Silent garbage or crash |
| Wrong tokenizer | Ids point at wrong embed rows |
| Dtype / device slip | Slow or NaN |
| Partial load | “Runs” but broken layers |

### Checklist

| Step | Do |
| --- | --- |
| 1 | `huggingface-cli login` if gated |
| 2 | Resolve model id (course default Llama 3.2 ~1B) |
| 3 | Config → empty `Llama` → copy weights |
| 4 | Always use the **same** tokenizer as the checkpoint |

### Check yourself

| Must hold | No missing/unexpected keys; shapes match |
| Must hold | Tokenizer vocabulary size = embed rows |
| Worth aiming for | Short generate looks like language; HF align later |

### Tuning the whole model

| Lever | Impact |
| --- | --- |
| Fix key map | Unlocks all downstream demos |
| Resume your own ckpt | Same API, your floats |
| Quantized loads | Later serve chapters |

### This course

```bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
```

## Run

```bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
```

## Fix

License, login, mirror—same class of issues as tokenizer download.

## Next

**Text generation** (`/generate/`): loop sample on logits.
