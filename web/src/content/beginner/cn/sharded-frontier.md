## 要点

### 1. 为什么

超大模型用 `index.json` + 许多分片文件。

### 2. 做什么

合成 index + 分片（Mixtral 形 MoE / DeepSeek 形 MLA）；装回并 `allclose`。

### 3–6. 本课怎么做

打包/拆包形状，不必真下几十 GB。

## 动手

```bash
python chapters/02-training/ep22_sharded_frontier.py
```

## 下一章

**Paged KV · CUDA Graph**（`?c=paged-cuda-graph`）。
