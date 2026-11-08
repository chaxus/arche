## 要点

两个服务端思路（教学体量）：

1. **Paged KV** — K/V 放进固定大小的**块** + **块表**，注意力时再 gather（vLLM 启发的形状）。避免「一条超长序列必须整段连续缓存」的故事。  
2. **CUDA Graph** — 有 CUDA 时**捕获**固定 shape 计算，减少启动开销；在 KV 写入安全变为原地之前，解码仍可 eager。

实验展示机制，不是可替换生产引擎。

## 动手

```bash
python chapters/01-inference/ep23_paged_cuda_graph.py
```

预期：分页 gather；仅在有 CUDA 时做 graph 捕获。

## 下一章

路径走完。可回看薄弱章，或有权重时跑 HF 对齐。
