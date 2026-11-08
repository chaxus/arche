## 要点

**是什么（What）：** **连续批处理**——服务端**调度**形状：多用户共享解码 step；完成的退役；新请求中途准入。

**为什么（Why）：** 单用户 generate 在短序列早结束后浪费 GPU。工业引擎（vLLM、TGI、SGLang…）只对**活跃**请求一起迈步，把设备喂饱。

**怎么做（How）：** 活跃集 → pad/步进 → 采样 → EOS/长度退役 → 重复。教学调度器 ≠ 完整 paged-attention 引擎。

### 1. What — 环

[`generate_continuous`](https://github.com/chaxus/arche/blob/main/arche/batching.py)：

```python
while any(not item.done for item in live):
    active = [item for item in live if not item.done]
    # pad 活跃行 → model(batch) → 逐行采样 → 退役完成者
```

每个请求自带 temperature / top-k / 最大长度。不改权重——纯调度。

### 2. Why — 吞吐

| 单环 | 连续批 |
| --- | --- |
| 一人独占 step | 多人共享 step |
| 短请求结束后空转 | 退役 + 准入保持 GPU 满 |
| 简单 | 生产要小心 KV 布局 |

### 3. How — 从零

| 步骤 | 做什么 |
| --- | --- |
| 1 | 表示每个任务（`DecodeRequest`） |
| 2 | 每步只对**活跃**行组 tensor |
| 3 | EOS / 长度时退役 |
| 4 | 随后用 paged KV 让显存匹配调度 |

### 4. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 完成的会停；行间不串话 |
| **软** | 长短混合流量下 tokens/s 更高 |
| **软** | 负载下尾延迟更稳 |

### 5. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 更好调度 | 同权重更高吞吐 |
| 耦合 KV 布局 | 生产要 paging / block |
| 忽略批处理 | GPU 白白闲着 |

### 6. 本课

```bash
python chapters/01-inference/ep17_continuous_batching.py
```

## 动手

```bash
python chapters/01-inference/ep17_continuous_batching.py
```

## 下一章

**Paged KV · CUDA Graph**（`/cn/paged-cuda-graph/`）。
