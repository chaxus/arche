## In one sentence
Write `index.json` + shard files, then reload Mixtral-shaped MoE and DeepSeek-shaped MLA with allclose checks.

## Before you start
- Frontier checkpoint chapter. Time: ~15 minutes.

## The idea (plain language)
Real HF repos split tensors across files listed by an index. We practice the same shape with synthetic `.pt` shards—no multi-gig download required.

## In the code
- `arche/shards.py`, `load_mixtral.py`, `load_deepseek_mla.py`
- `chapters/02-training/ep22_sharded_frontier.py`

## Do this now
```bash
python chapters/02-training/ep22_sharded_frontier.py
```

## If something breaks
- Temp directory permissions → run from a normal user workspace.

## Don't worry about yet
Safetensors mmap edge cases in every host.

## Next
**Paged KV · CUDA graphs** (`?c=paged-cuda-graph`).
