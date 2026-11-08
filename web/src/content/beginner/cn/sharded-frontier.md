## 要点

开源前沿权重多半是 **`index.json` + 分片**，不是单文件。组不齐分片就加载不了 Llama/Qwen/DeepSeek/Mixtral 级 hub；拼接错会静默弄坏张量。

用 HF 风格 index，按键分区，写分片，加载时聚合——玩具体积，同一协议。

### API

[`write_sharded_state` / `load_sharded_state`](https://github.com/chaxus/arche/blob/main/arche/shards.py)：

```python
write_sharded_state(directory, state_dict, shard_size=...)
state = load_sharded_state(directory)  # 经 index 重建完整 dict
```

合成 pack/unpack 加 `allclose` 教协议，**不必**下几十 GB。

### 为什么要分片

| 单文件 | 分片 |
| --- | --- |
| 演示简单 | 大模型 hub 默认 |
| 一条下载流 | 可并行拉分片 |
| 峰值 RAM = 全模 | 仍要小心组装 |

### 从零

| 步骤 | 做什么 |
| --- | --- |
| 1 | 写出 张量名 → 分片文件 的 index |
| 2 | 按大小/数量策略切开 |
| 3 | 加载：读 index，只拉需要的分片 |
| 4 | 缺件时失败要清楚 |

### 自己核对

- index 与分片一致；重载对齐。
- 缺分片/缺键错误响亮。
- 分片策略在下载并行与峰值 RAM 间折中。

### 杠杆

| 杠杆 | 影响 |
| --- | --- |
| 分片策略 | 并行度与内存峰值 |
| 拼接错 | 静默坏权重 |
| safetensors + index | 业界主流组合 |

### 本课

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
