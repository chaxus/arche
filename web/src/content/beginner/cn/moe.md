## 要点

### 1. 为什么要 MoE

每个 token 都跑同一套 dense MLP 简单，但容量要人人付算力。

### 2. MoE 做什么

很多专家 MLP（这里是 SwiGLU）+ **路由器**：每 token 只跑 **top-k** → 稀疏计算。

### 3. 版图

| 想法 | 备注 |
| --- | --- |
| Dense FFN | 始终全开 |
| **MoE（Mixtral 风）** | top-k 专家 |
| 共享专家等变体 | 产品各异 |

### 4. 业界实际

本课是教学 MoE 块——不是完整 Mixtral 生产栈。

### 5. 怎样算对

- 路由到合法专家  
- 负载别极端倾斜（真训练很在意；这里只尝 API）  

### 6. 本课怎么做

`arche/moe.py` 引物。

## 动手

```bash
python chapters/01-inference/ep14_moe.py
```

## 下一章

**MLA**（`?c=mla`）。
