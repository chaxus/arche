## Idea

### 1. What it is for

Huge checkpoints often ship as `index.json` + many shards. This lesson uses **synthetic** index+shards (Mixtral-like MoE / DeepSeek-like MLA) to practice pack/unpack and `allclose` after load—no multi-tens-of-GB download.

### 2. Why the LLM pipeline needs it

Single files don’t scale. Can’t read shards ⇒ can’t use frontier open weights. Synthetic data trains **index + concat**, not download stamina.

### 3. Where it comes from · how you build it

Build synthetic shards → load via index → numeric check.

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Index matches shards | Missing/wrong keys fail clearly |
| **Hard** | Reload `allclose` | Concat correct |
| **Soft** | Real hub formats | Product engineering |

### 5. What we do

This episode’s script.

## Run

```bash
python chapters/02-training/ep22_sharded_frontier.py
```

## Next

**Paged KV · CUDA Graph** (`?c=paged-cuda-graph`).
