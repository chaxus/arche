## 要点

**连续批处理**是服务端 **调度** 形状：多用户共享解码 step，完成的退役、新请求中途准入。单用户 generate 在短序列早结束后会空转 GPU。vLLM、TGI、SGLang 等引擎只对 **活跃** 请求一起迈步，把设备喂饱。活跃集 → pad → 采样 → EOS/长度退役 → 重复——教学调度器不是完整 paged-attention 引擎。

### 调度环

[`generate_continuous`](https://github.com/chaxus/arche/blob/main/arche/batching.py)：

```python
while any(not item.done for item in live):
    active = [item for item in live if not item.done]
    # pad 活跃行 → model(batch) → 逐行采样 → 退役完成者
```

每个请求自带 temperature / top-k / 最大长度。不改权重——纯调度。

### 吞吐对比

| 单环 | 连续批 |
| --- | --- |
| 一人独占 step | 多人共享 step |
| 短请求结束后空转 | 退役 + 准入保持 GPU 满 |
| 简单 | 生产要小心 KV 布局 |

### 搭玩具调度器

| 步骤 | 做什么 |
| --- | --- |
| 1 | 表示每个任务（`DecodeRequest`） |
| 2 | 每步只对**活跃**行组 tensor |
| 3 | EOS / 长度时退役 |
| 4 | 随后用 paged KV 让显存匹配调度 |

### 自己核对

| 必须成立 | 完成的会停；行间不串话 |
| 尽量做到 | 长短混合流量下 tokens/s 更高 |
| 尽量做到 | 负载下尾延迟更稳 |

### 服务天花板

| 杠杆 | 影响 |
| --- | --- |
| 更好调度 | 同权重更高吞吐 |
| 耦合 KV 布局 | 生产要 paging / block |
| 忽略批处理 | GPU 白白闲着 |

### 本课

```bash
python chapters/01-inference/ep17_continuous_batching.py
```

## 动手

```bash
python chapters/01-inference/ep17_continuous_batching.py
```

## 下一章

**Paged KV · CUDA Graph**（`/cn/paged-cuda-graph/`）。
