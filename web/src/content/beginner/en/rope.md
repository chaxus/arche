## Idea

### 1. What it is for

**RoPE** rotates Q and K by an angle from each **position**. Relative distance shows up in the attention dot product—no big PE table on embeddings.

### 2. Why the LLM pipeline needs it

Self-attention alone is order-blind. Language is not. Embedding answers “which token”; RoPE answers “which place.” Modern dense decoders write position into Q/K.

### 3. Where it comes from · how you build it

| Stage | What happens |
| --- | --- |
| **Spec** | Choose `rope_theta`; cache cos/sin per position |
| **Forward** | `rotate_half` on Q/K, combine with cos/sin |
| **Train/infer** | Same rotation; longer context may need scaling (later chapter) |

Drag the page viz to see angles change with position.

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Q/K shape unchanged | Phase only |
| **Hard** | Relative position enters scores | Swap positions → scores change |
| **Hard** | Match checkpoint contract | `theta` / scaling must match weights |
| **Soft** | Stable inside train length | Full-model quality is the judge |
| **Soft** | Extra-long context | See RoPE scaling chapter |

### 5. What labs ship · what we do

Llama 3 / 3.2: RoPE + large `rope_theta`.  
`arche/rope.py`; lab applies RoPE and checks shapes.

## Run

```bash
python chapters/01-inference/ep05_rope.py
```

## Next

**GQA** (`?c=gqa`): fewer KV heads.
