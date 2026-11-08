## Idea

**What:** **RoPE scaling** — adjust rotary frequencies (Llama 3-style wavelength smoothing) so positions beyond the original train window stay usable.

**Why:** Short-context pretrain + naive long indices → rotations wrap badly; quality collapses far out. Long-context products need a **recipe extension**, not “just set max_seq bigger.”

**How:** Match checkpoint `theta` + scale factors; optionally continue-train on long data; eval short and long.

### 1. What — code hook

[`llama3_inv_freq`](https://github.com/chaxus/arche/blob/main/arche/rope.py) / `build_rope_cache(..., scale_factor=..., low_freq_factor=..., high_freq_factor=..., original_max_position=...)`.

When `scale_factor == 1.0`, plain RoPE. Otherwise wavelength bands are scaled/smoothed the HF Llama-3 way—**same math the 128k-class configs advertise**.

### 2. Why — not free length

| Illusion | Reality |
| --- | --- |
| Stretch indices only | High-freq bands misbehave |
| Bigger window free | KV memory ↑ with length |
| Scaling = long knowledge | Still need data / continued train for hard long tasks |

### 3. How — from zero

| Step | Do |
| --- | --- |
| 1 | Read official rope scaling fields from the ckpt config |
| 2 | Build cos/sin with those factors |
| 3 | Keep short-context sanity checks |
| 4 | Long-bench / needle tests before claiming 128k |

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | Factors match the checkpoint |
| **Soft** | Near context still stable; far context usable |
| **Soft** | Long-bench acceptable for the product bar |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| Wrong scaling | Breaks long (and maybe short) |
| YaRN / NTK / … | Other open recipes—don’t mix under Llama weights |
| Longer windows | Always pay in KV |

### 6. This course

```bash
python chapters/01-inference/ep13_rope_scaling.py
```

## Run

```bash
python chapters/01-inference/ep13_rope_scaling.py
```

## Next

**Device · INT8 · chat** (`/device-quant-chat/`).
