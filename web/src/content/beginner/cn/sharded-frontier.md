## 要点

从零训到大模型：权重是 **index.json + 分片**。不会拼分片就接不住开放前沿权重。

### 1. 有什么作用

合成分片练习 pack/unpack，装回 `allclose`——免下几十 GB。

### 2. 从零开始：怎么做

按 HF 习惯写 index；按张量切分保存；加载时按 index 聚合。

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | index 与分片一致；装回对齐；缺片报错清楚 |
| **软** | 兼容真仓库格式 |

### 4. 如何改进 · 对整模影响

分片策略影响下载并行与内存峰值；错拼 = 静默坏权。

### 5. 业界 · 开源

HF Hub 分片是标准；Llama/Qwen/DeepSeek/Mixtral 均如此。`safetensors` + index 是主格式。

### 6. 本课

```bash
python chapters/02-training/ep22_sharded_frontier.py
```

## 动手

```bash
python chapters/02-training/ep22_sharded_frontier.py
```

## 下一章

**Paged KV · CUDA Graph**（`?c=paged-cuda-graph`）。
