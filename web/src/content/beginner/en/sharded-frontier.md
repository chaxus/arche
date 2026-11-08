## Idea

From-zero at scale: weights are **index.json + shards**. Can’t assemble shards ⇒ can’t take open frontier weights.

### 1. What it is for

Synthetic pack/unpack + `allclose` without multi-tens-of-GB downloads.

### 2. From zero: how you build it

HF-style index; shard tensors; aggregate on load.

### 3. What “good” means

Index matches shards; reload aligns; missing pieces fail clearly.

### 4. Improve · whole-model impact

Shard policy affects download parallelism and peak RAM; bad concat ⇒ silent bad weights.

### 5. Industry · open models

HF Hub sharding standard; Llama/Qwen/DeepSeek/Mixtral; safetensors+index dominant.

### 6. This course

```bash
python chapters/02-training/ep22_sharded_frontier.py
```

## Run

```bash
python chapters/02-training/ep22_sharded_frontier.py
```

## Next


Path complete—revisit weak chapters, or re-run HF align with weights.
