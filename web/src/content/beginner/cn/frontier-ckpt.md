## 要点

### 1. 为什么

MoE/MLA 也要能存取，不能静默形状错乱。

### 2. 做什么

往返教学检查点；把 Mixtral 风 `w1/w2/w3` 映射到我们的模块名。

### 3–6. 本课怎么做

读开放 MoE 检查点的引物——不是完整训练器。

## 动手

```bash
python chapters/02-training/ep20_frontier_ckpt.py
```

## 下一章

**HF 对齐**（`?c=hf-align`）。
