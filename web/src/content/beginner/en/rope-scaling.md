## Idea

### 1. Why scale RoPE

Angles were tuned for a training length; much longer prompts can behave oddly.

### 2. What Llama-style scaling does

Keep large `rope_theta`; smoothly rescale **long wavelengths** (factor / low / high)—HF-compatible.

### 3. Landscape

| Method | Plain idea |
| --- | --- |
| NTK / YaRN / … | Other long-context RoPE recipes |
| **Llama 3 wavelength smoothing** | What we mirror here |
| Bigger context train | Data + length, not only formulas |

### 4. What labs ship

Llama 3.2 long-context schedule in the official configs.

### 5. What “good” means

- Short-range stays familiar  
- Long-range stretches instead of wrapping badly  

### 6. What we do

Scaling helpers in `arche/rope.py` + chapter script.

## Run

```bash
python chapters/01-inference/ep13_rope_scaling.py
```

## Next

**MoE** (`?c=moe`).
