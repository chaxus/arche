## 要点

### 1. 为什么要 MLA

GQA 已减 KV 头；解码显存仍可能占大头。

### 2. MLA 做什么

缓存**低秩 latent**（+ 小段 RoPE 流），而不是每头完整 K/V；需要时再展开。

### 3. 版图

| 想法 | 作用 |
| --- | --- |
| GQA | 更少 KV 头 |
| **MLA** | latent KV（DeepSeek 思路） |
| Paged KV | 显存怎么切块 |

### 4. 业界实际

教学形状——不是比特级 DeepSeek 权重。

### 5. 怎样算对

- latent 缓存比满 KV 小  
- 前向能跑通  

### 6. 本课怎么做

`arche/mla.py` 引物。

## 动手

```bash
python chapters/01-inference/ep15_mla.py
```

## 下一章

**设备 · INT8 · 对话**（`?c=device-quant-chat`）。
