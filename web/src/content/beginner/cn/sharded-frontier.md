## 要点

**是什么（What）：** 开源前沿权重以 **`index.json` + 分片** 到来，不是单文件。学会按这套布局打包/还原。

**为什么（Why）：** 组不齐分片 ⇒ 加载不了 Llama/Qwen/DeepSeek/Mixtral 级 hub。拼接错 ⇒ 张量静默错误。

**怎么做（How）：** HF 风格 index；按键分区；写分片文件；加载时聚合——玩具体积，同一协议。

### 1. What — API

[`write_sharded_state` / `load_sharded_state`](https://github.com/chaxus/arche/blob/main/arche/shards.py)：

```python
write_sharded_state(directory, state_dict, shard_size=...)
state = load_sharded_state(directory)  # 经 index 重建完整 dict
```

合成 pack/unpack + `allclose` 教协议，**不必**下几十 GB。

### 2. Why — 规模现实

| 单文件 | 分片 |
| --- | --- |
| 演示简单 | 大模型 hub 默认 |
| 一条下载流 | 可并行拉分片 |
| 峰值 RAM = 全模 | 仍要小心组装 |

### 3. How — 从零

| 步骤 | 做什么 |
| --- | --- |
| 1 | 写出 张量名 → 分片文件 的 index |
| 2 | 按大小/数量策略切开 |
| 3 | 加载：读 index，只拉需要的分片 |
| 4 | 缺件时失败要清楚 |

### 4. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | index 与分片一致；重载对齐 |
| **硬** | 缺分片/缺键错误响亮 |
| **软** | 分片策略在下载并行与峰值 RAM 间折中 |

### 5. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 分片策略 | 并行度与内存峰值 |
| 拼接错 | 静默坏权重 |
| safetensors + index | 业界主流组合 |

### 6. 本课

```bash
python chapters/02-training/ep22_sharded_frontier.py
```

整条路径完成——可回看弱章，或带着权重重跑 HF 对齐。

## 动手

```bash
python chapters/02-training/ep22_sharded_frontier.py
```

## 下一章

路径完成——回看弱章，或带着权重重跑 HF 对齐。
