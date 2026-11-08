## 要点

Part 3 收尾引物：**分页 KV** 管显存碎片，**CUDA Graph** 砍固定形状启动税——对接 vLLM 一类引擎。下一 Part 进入 MoE/MLA 等前沿结构。

### 1. 有什么作用

1. Paged KV：定长块 + 块表 + gather  
2. CUDA Graph：捕获固定 shape 计算  

教学 ≠ 可替换 vLLM。

### 2. 从零开始：怎么做

先连续批与逻辑 KV → 引入块表分配/回收 → 可选 Graph 捕获 decode 步。

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | gather 与逻辑缓存一致；块表不越界 |
| **软** | 更少碎片、更少启动；吞吐↑ |

### 4. 如何改进 · 对整模影响

页大小、预分配策略改吞吐与浪费。Graph 要求形状稳；动态 batch 要小心重捕获。不改模型权重，改服务上限。

### 5. 业界 · 开源

vLLM paged attention 标杆；TensorRT-LLM/SGLang 有图与分页变体；与 Llama/Qwen/DeepSeek 权重搭配使用。

### 6. 本课

```bash
python chapters/01-inference/ep23_paged_cuda_graph.py
```

路径走完：可回看薄弱章，或有权重时跑 HF 对齐。

## 动手

```bash
python chapters/01-inference/ep23_paged_cuda_graph.py
```

## 下一章


**MoE**（`/cn/moe/`）：进入前沿结构。
