## 一句话
用块表把 KV 放进固定 page；在 CUDA 上演示固定 shape 的 graph 捕获。

## 开始之前
- 连续批处理与 KV cache。约 18 分钟。

## 用人话讲
**Paged KV** 分配 page，再 gather 成连续 cache——教学版 vLLM 思路。
**CUDA graphs** 捕获固定 shape 计算；在 KV 变成页内原地写之前，Llama 解码仍 eager。

## 对应代码
- `arche/paged_kv.py`、`cuda_graphs.py`
- `chapters/01-inference/ep23_paged_cuda_graph.py`

## 现在就做
```bash
python chapters/01-inference/ep23_paged_cuda_graph.py
```

## 如果翻车了
- 无 CUDA：跳过 graph；paged KV 仍可在 CPU 跑。

## 先别深挖
重写整个 vLLM——课程到这里是为了叫出生产零件的名字。

## 下一章
路径走完了。可回看薄弱章，或在有权重时跑真正的 HF 对齐。
