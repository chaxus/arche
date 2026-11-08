## Idea

Synthetic `index.json` + `.pt` shards → reload Mixtral-shaped MoE and DeepSeek-shaped MLA (allclose).

## Run

```bash
python chapters/02-training/ep22_sharded_frontier.py
```

## Next

**Paged KV · CUDA graphs** (`?c=paged-cuda-graph`).
