## 一句话
MoE 每个 token 只跑 top-k 个专家 MLP——算力稀疏，参数可以很多。

## 开始之前
- 理解 SwiGLU。约 15 分钟。

## 用人话讲
**路由器**给专家打分；每个 token 选 `top_k` 个再加权混合。
这是 Mixtral 类堆叠背后的稀疏 FFN 思路。

## 对应代码
- `arche/moe.py`
- `chapters/01-inference/ep14_moe.py`

## 现在就做
```bash
python chapters/01-inference/ep14_moe.py
```

## 如果翻车了
- `top_k` 不能大于专家数。

## 先别深挖
负载均衡损失与专家并行。

## 下一章
**MLA**（`?c=mla`）。
