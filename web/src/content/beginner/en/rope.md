## Idea

### 1. Why position

Without position, “A then B” and “B then A” look the same to attention.

### 2. What RoPE does

**RoPE** rotates each Q/K by an angle from its **position**. Relative distance shows up in the dot product—no big PE table on embeddings.

On this page: drag the interactive picture. In code: cos/sin cache, `rotate_half`, then `q' = q·cos + rotate(q)·sin`.

### 3. Alternatives

| Approach | Plain idea | Today |
| --- | --- | --- |
| Absolute learned PE | Add a vector per index | Older stacks |
| Sinusoidal PE | Fixed sin/cos by position | Transformer original |
| **RoPE** | Rotate Q/K | **Mainstream decoders** |
| ALiBi / other biases | Distance bias in scores | Some long-context lines |

### 4. What labs ship

Llama 3 / 3.2: large `rope_theta`; later chapters add **wavelength scaling** for longer context.

### 5. What “good” means

- Relative positions matter in attention  
- Shapes of Q/K unchanged  
- Long context may need scaling (next specialty chapter)  

### 6. What we do

`arche/rope.py`. Lab applies RoPE and checks shapes.

## Run

```bash
python chapters/01-inference/ep05_rope.py
```

## Next

**GQA** (`?c=gqa`): fewer KV heads.
