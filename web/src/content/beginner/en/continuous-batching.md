## Idea

From zero single-user loops → industry **multi-user shared stepping**: continuous batching is the serving schedule shape.

### 1. What it is for

Share a step among jobs; retire finished; admit new. Teaching ≠ full vLLM.

### 2. From zero: how you build it

Active set → step only actives → retire; add paged KV later for production shape.

### 3. What “good” means

Retire finished; no cross-talk; fuller GPU / steadier tail latency.

### 4. Improve · whole-model impact

Better schedule ⇒ more throughput on same hardware; coupled to KV layout, not weights.

### 5. Industry · open models

vLLM continuous batch + paged attention; TGI/TRT-LLM/SGLang same idea.

### 6. This course

```bash
python chapters/01-inference/ep17_continuous_batching.py
```

## Run

```bash
python chapters/01-inference/ep17_continuous_batching.py
```

## Next

**Pretrain** (`?c=pretrain`).
