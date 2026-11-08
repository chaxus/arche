## Idea

Open frontier weights usually arrive as **`index.json` + shards**, not one file. If you can’t assemble shards, you can’t load Llama/Qwen/DeepSeek/Mixtral-class hubs; bad concat silently corrupts tensors.

Use an HF-style index, partition keys, write shard files, and aggregate on load—toy size, same protocol.

### API

[`write_sharded_state` / `load_sharded_state`](https://github.com/chaxus/arche/blob/main/arche/shards.py):

```python
write_sharded_state(directory, state_dict, shard_size=...)
state = load_sharded_state(directory)  # rebuild full dict via index
```

Synthetic pack/unpack plus `allclose` teaches the protocol **without** multi-tens-of-GB downloads.

### Why hubs shard

| Single file | Sharded |
| --- | --- |
| Simple demos | Hub default for big models |
| One download stream | Parallel shard fetch |
| Peak RAM = full model | Still careful—assemble wisely |

### From zero

| Step | Do |
| --- | --- |
| 1 | Emit index mapping tensor name → shard file |
| 2 | Split by size/count policy |
| 3 | Load: read index, pull only needed shards |
| 4 | Fail clearly on missing pieces |

### Check yourself

- Index matches shards; reload aligns.
- Missing shard/key errors are loud.
- Shard policy balances download vs peak RAM.

### Levers

| Lever | Impact |
| --- | --- |
| Shard policy | Parallelism and memory peaks |
| Bad concat | Silent bad weights |
| safetensors + index | Industry dominant pair |

### This course

```bash
python chapters/02-training/ep22_sharded_frontier.py
```

Path complete—revisit weak chapters, or re-run HF align with weights.

## Run

```bash
python chapters/02-training/ep22_sharded_frontier.py
```

## Next

Path complete—revisit weak chapters, or re-run HF align with weights.
