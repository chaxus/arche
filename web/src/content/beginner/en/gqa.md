## Idea

From-zero architecture: attention layout sets **decode VRAM** vs quality.

### 1. What it is for

**GQA**: more Q heads than KV heads; groups share KV; still causal.

### 2. From zero: how you build it

Choose divisible `n_q`/`n_kv`, broadcast KV, train; cache stores `n_kv`. Llama 3.2 1B: **32Q/8KV**.

### 3. What “good” means

Valid groups; causal; match weight heads; smaller cache than MHA without quality collapse.

### 4. Improve · whole-model impact

MHA richest/costliest; GQA default open midsize; MQA extreme; MLA further compression. Head-count change ⇒ retrain.

### 5. Industry · open models

Llama 3/Qwen: GQA. DeepSeek-V3: MLA path. Classic GPT-2 teaching: MHA.

### 6. This course

```bash
python chapters/01-inference/ep06_gqa.py
```

## Run

```bash
python chapters/01-inference/ep06_gqa.py
```

## Next

**SwiGLU** (`?c=swiglu`): wire the FFN from zero.
