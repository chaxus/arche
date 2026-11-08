## Idea

Huge models rarely ship as one file. Hugging Face-style layouts use an **`index.json`** plus many **`.safetensors` / `.pt` shards**.

We build a **synthetic** index + shards for Mixtral-shaped MoE and DeepSeek-shaped MLA, reload them, and `allclose`—so you see the packing/unpacking pattern without downloading tens of GB.

## Run

```bash
python chapters/02-training/ep22_sharded_frontier.py
```

Expect shard write/read and matching tensors.

## Next

**Paged KV · CUDA graphs** (`?c=paged-cuda-graph`): serving-memory and capture ideas.
