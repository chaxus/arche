## 要点

### 1. 为什么要缓存

朴素解码每步都对整段 prompt 重算 K/V。

### 2. KV 缓存做什么

**Prefill** 一次 → 存每层 K/V。每次 **decode** 只追加一步再对缓存做注意力。见本页条带图。

### 3. 相关想法

| 想法 | 作用 |
| --- | --- |
| 连续缓存 | 沿时间 `cat`（本课教学缓存） |
| Paged KV | 块表 + gather（后面引物） |
| GQA/MLA | 缩小要存的东西 |

### 4. 业界实际

生产解码器都有 KV 缓存；分页/布局各引擎不同。

### 5. 怎样算对

- 与全量重算结果一致（差在数值）  
- 算量随新 token 涨，而不是每步整段 prompt  

### 6. 本课怎么做

`arche/cache.py` 的 `KVCache`。

## 动手

```bash
python chapters/01-inference/ep12_kv_cache.py
```

## 下一章

**RoPE 缩放**（`?c=rope-scaling`）。
