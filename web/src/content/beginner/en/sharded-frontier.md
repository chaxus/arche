## Idea

### 1. Why

Huge models ship as `index.json` + many shard files.

### 2. What

Synthetic index + shards for Mixtral-shaped MoE / DeepSeek-shaped MLA; reload and `allclose`.

### 3–6. What we do

Packing/unpacking pattern without downloading tens of GB.

## Run

```bash
python chapters/02-training/ep22_sharded_frontier.py
```

## Next

**Paged KV · CUDA graphs** (`?c=paged-cuda-graph`).
