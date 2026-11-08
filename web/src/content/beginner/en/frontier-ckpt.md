## Idea

From-zero MoE/MLA must **speak checkpoint names/shapes** of open weights—or frontier structure stays on paper.

### 1. What it is for

Round-trip teaching ckpts; map Mixtral-like `w1/w2/w3` into our modules.

### 2. From zero: how you build it

Name map ↔ open names; save/load; fail loudly on missing keys.

### 3. What “good” means

Round-trip align; experts covered; hooks real hubs.

### 4. Improve · whole-model impact

Bad maps ⇒ silent random experts; consistent naming cuts debt.

### 5. Industry · open models

HF shards + documented keys; Mixtral/DeepSeek naming; converters are standard.

### 6. This course

```bash
python chapters/02-training/ep20_frontier_ckpt.py
```

## Run

```bash
python chapters/02-training/ep20_frontier_ckpt.py
```

## Next


**Sharded MoE / MLA** (`?c=sharded-frontier`).
