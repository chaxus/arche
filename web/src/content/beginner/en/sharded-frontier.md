## Idea

**What:** Open frontier weights arrive as **`index.json` + shards**, not one file. Learn to pack/unpack that layout.

**Why:** Can’t assemble shards ⇒ can’t load Llama/Qwen/DeepSeek/Mixtral-class hubs. Bad concat ⇒ silent wrong tensors.

**How:** HF-style index; partition keys; write shard files; aggregate on load—toy size, same protocol.

### 1. What — API

[`write_sharded_state` / `load_sharded_state`](https://github.com/chaxus/arche/blob/main/arche/shards.py):

```python
write_sharded_state(directory, state_dict, shard_size=...)
state = load_sharded_state(directory)  # rebuild full dict via index
```

Synthetic pack/unpack + `allclose` teaches the protocol **without** multi-tens-of-GB downloads.

### 2. Why — scale reality

| Single file | Sharded |
| --- | --- |
| Simple demos | Hub default for big models |
| One download stream | Parallel shard fetch |
| Peak RAM = full model | Still careful—assemble wisely |

### 3. How — from zero

| Step | Do |
| --- | --- |
| 1 | Emit index mapping tensor name → shard file |
| 2 | Split by size/count policy |
| 3 | Load: read index, pull only needed shards |
| 4 | Fail clearly on missing pieces |

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | Index matches shards; reload aligns |
| **Hard** | Missing shard/key errors are loud |
| **Soft** | Shard policy balances download vs peak RAM |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| Shard policy | Parallelism and memory peaks |
| Bad concat | Silent bad weights |
| safetensors + index | Industry dominant pair |

### 6. This course

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
