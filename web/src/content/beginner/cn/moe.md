## 要点

Dense MLP 对每个 token 跑**同一套**前馈。**混合专家（MoE）**准备很多专家 MLP（这里是 SwiGLU 专家），再用**路由器**给每个 token 只选 **top-k** 个专家来算。

总参数可以很大，但每个 token 只付少数专家的算力——**稀疏计算**（Mixtral 思路）。本章是教学用 MoE 块，不是完整生产 Mixtral。

## 动手

```bash
python chapters/01-inference/ep14_moe.py
```

预期：一次小 MoE 前向的路由 / 形状。

## 下一章

**MLA**（`?c=mla`）：压缩解码时要缓存的东西。
