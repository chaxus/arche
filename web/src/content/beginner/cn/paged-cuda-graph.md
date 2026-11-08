## 要点

**Paged KV**：块表 + gather。**CUDA graph**：有 CUDA 时捕获固定 shape 计算（KV 原地写之前解码仍 eager）。

## 动手

```bash
python chapters/01-inference/ep23_paged_cuda_graph.py
```

## 下一章

路径走完。可回看薄弱章，或有权重时跑 HF 对齐。
