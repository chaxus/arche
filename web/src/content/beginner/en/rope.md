## Idea

From-zero decoders: embedding says “which token”; **RoPE** (or other PE) writes **position** into attention.

### 1. What it is for

Rotate Q/K by position so relative distance enters the dot product—no absolute PE table on embeds.

### 2. From zero: how you build it

Pick `rope_theta`, cache cos/sin, `rotate_half` + combine on Q/K; scale later for long context.

### 3. What “good” means

Shape unchanged; relative position affects scores; match checkpoint `theta`/scaling.

### 4. Improve · whole-model impact

Theta/scaling ↔ long context; absolute PE→RoPE is modern default; don’t mix PE families with foreign weights.

### 5. Industry · open models

Llama/Qwen/DeepSeek: RoPE. Classic Transformer: sinusoidal. Some long-context lines: ALiBi/YaRN/NTK.

### 6. This course

```bash
python chapters/01-inference/ep05_rope.py
```

## Run

```bash
python chapters/01-inference/ep05_rope.py
```

## Next

**GQA** (`/gqa/`): choose head counts from zero to save KV cache.
