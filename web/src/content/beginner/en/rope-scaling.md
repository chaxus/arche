## Idea

**RoPE scaling** adjusts rotary frequencies (Llama 3-style wavelength smoothing) so positions beyond the original training window stay usable. Short-context pretrain plus naive long indices makes rotations wrap badly and quality collapses far out. Long-context products need a **recipe extension**, not “just set max_seq bigger.” Match checkpoint `theta` and scale factors, optionally continue-train on long data, and eval both short and long.

### Code hook

[`llama3_inv_freq`](https://github.com/chaxus/arche/blob/main/arche/rope.py) / `build_rope_cache(..., scale_factor=..., low_freq_factor=..., high_freq_factor=..., original_max_position=...)`.

When `scale_factor == 1.0`, plain RoPE. Otherwise wavelength bands are scaled/smoothed the HF Llama-3 way—**same math the 128k-class configs advertise**.

### Length is not free

| Illusion | Reality |
| --- | --- |
| Stretch indices only | High-freq bands misbehave |
| Bigger window free | KV memory ↑ with length |
| Scaling = long knowledge | Still need data / continued train for hard long tasks |

### Apply the recipe

| Step | Do |
| --- | --- |
| 1 | Read official rope scaling fields from the ckpt config |
| 2 | Build cos/sin with those factors |
| 3 | Keep short-context sanity checks |
| 4 | Long-bench / needle tests before claiming 128k |

### Check yourself

| Must hold | Factors match the checkpoint |
| Worth aiming for | Near context still stable; far context usable |
| Worth aiming for | Long-bench acceptable for the product bar |

### Whole-model impact

| Lever | Impact |
| --- | --- |
| Wrong scaling | Breaks long (and maybe short) |
| YaRN / NTK / … | Other open recipes—don’t mix under Llama weights |
| Longer windows | Always pay in KV |

### This course

```bash
python chapters/01-inference/ep13_rope_scaling.py
```

## Run

```bash
python chapters/01-inference/ep13_rope_scaling.py
```

## Next

**Device · INT8 · chat** (`/device-quant-chat/`).
