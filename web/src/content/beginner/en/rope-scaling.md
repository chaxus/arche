## Idea

After short-context from-zero training, products want 128K-class windows—**RoPE scaling** is a recipe extension, not casually stretching indices.

### 1. What it is for

Llama 3-style wavelength smoothing (large theta + factors) so long positions don’t wrap badly.

### 2. From zero: how you build it

Train at base length → enable checkpoint-matched scaling and/or continue long-data train → eval long context.

### 3. What “good” means

Match checkpoint config; near stable; far usable; long-bench OK.

### 4. Improve · whole-model impact

Wrong scaling breaks long (and maybe short) context; bigger windows raise KV cost; scaling ≠ free long-range knowledge.

### 5. Industry · open models

Llama 3/3.2 official long-context; Qwen/DeepSeek own recipes; YaRN/NTK in open long-context work.

### 6. This course

```bash
python chapters/01-inference/ep13_rope_scaling.py
```

## Run

```bash
python chapters/01-inference/ep13_rope_scaling.py
```

## Next


**Device · INT8 · chat** (`?c=device-quant-chat`).
