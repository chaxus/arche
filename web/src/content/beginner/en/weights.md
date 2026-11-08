## Idea

**What:** **Load** an open Llama 3.2 checkpoint into your handwritten stack via a **key map** (safetensors / HF → `state_dict`).

**Why:** Random weights = correct graph, nonsense text. Official floats make the same forward a language model *now*. You already saw how pretrain *writes* floats; this chapter **borrows** a finished run so generate/align work without a cluster.

**How:** Accept license → download → map keys → pair the **official tokenizer** → sanity-check shapes.

### 1. What — entry

[`from_pretrained`](https://github.com/chaxus/arche/blob/main/arche/load_weights.py) builds `Llama(cfg)` and copies HF tensors into matching names.

| Path | Role |
| --- | --- |
| **Load (this chapter)** | Example checkpoint → speak immediately |
| **Pretrain (earlier)** | How those floats get written from zero |
| **HF align (next-next)** | Proof map + formulas match HF logits |

### 2. Why — map carefully

| Risk | Effect |
| --- | --- |
| Wrong key / transpose | Silent garbage or crash |
| Wrong tokenizer | Ids point at wrong embed rows |
| Dtype / device slip | Slow or NaN |
| Partial load | “Runs” but broken layers |

### 3. How — checklist

| Step | Do |
| --- | --- |
| 1 | `huggingface-cli login` if gated |
| 2 | Resolve model id (course default Llama 3.2 ~1B) |
| 3 | Config → empty `Llama` → copy weights |
| 4 | Always use the **same** tokenizer as the checkpoint |

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | No missing/unexpected keys; shapes match |
| **Hard** | Tokenizer vocabulary size = embed rows |
| **Soft** | Short generate looks like language; HF align later |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| Fix key map | Unlocks all downstream demos |
| Resume your own ckpt | Same API, your floats |
| Quantized loads | Later serve chapters |

### 6. This course

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
