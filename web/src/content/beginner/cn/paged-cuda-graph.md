## 要点

### 1. 为什么

服务端需要比玩具循环更聪明的显存与更低的启动开销。

### 2. 做什么

1. **Paged KV** — 定长块 + 块表 + gather  
2. **CUDA Graph** — 有 CUDA 时捕获固定 shape 计算  

教学机制——不是可替换的 vLLM。

### 3–6. 本课怎么做

`arche/paged_kv.py`（及 graph）+ 本章脚本。

## 动手

```bash
python chapters/01-inference/ep23_paged_cuda_graph.py
```

## 下一章

路径走完——可回看薄弱章，或有权重时跑 HF 对齐。
