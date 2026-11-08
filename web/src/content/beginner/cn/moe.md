## 要点

**混合专家（MoE）**用多个 expert 加 router 替换稠密 SwiGLU；每个 token 只跑 **top-k**。参数可以很大，每 token 算力仍接近稠密。DeepSeek-V3 开源故事大体是 MoE 管容量、MLA 管 KV；Mixtral 是另一条开源 MoE。本课先走稠密 Llama 3.2，是为了你会读 FFN 再看稀疏地图。

选定 `E` 与 `top_k`，接好 experts 和 gate，认真训练时加负载均衡，并想好 expert 在检查点里怎么落盘。

### 前向

[`MoESwiGLU`](https://github.com/chaxus/arche/blob/main/arche/moe.py)：

```python
logits = self.gate(x)
probs = F.softmax(logits, dim=-1)
top_probs, top_idx = probs.topk(self.top_k, dim=-1)
# 对选中的 SwiGLU expert 加权求和
```

解码器脊柱不变；只换 MLP 槽位。

### 稠密 vs 稀疏

| | 稠密 SwiGLU | MoE |
| --- | --- | --- |
| 参数 | 一个 FFN | `E` × FFN |
| 每 token FLOPs | 完整 FFN | ≈ `top_k` 个 expert |
| 失败模式 | — | 无均衡时 expert 坍塌 |

### 从零

| 步骤 | 做什么 |
| --- | --- |
| 1 | 保留注意力 / norm；MLP → MoE |
| 2 | 选 `E`、`top_k`（如 Mixtral 类 top-2） |
| 3 | 认真训练时加负载均衡损失 |
| 4 | 规划 expert 并行 / 分片加载 |

### 自己核对

- 路由合法；合并权重合理。
- expert 负载大致均衡；质量/容量折中说得通。
- checkpoint 映射覆盖全部 expert。

### 杠杆

| 杠杆 | 影响 |
| --- | --- |
| 更大 `E` / `k` | 容量 vs 算力 |
| 均衡损失 | 对抗坍塌 |
| Expert 并行 | 多卡工程成本 |

### 业界地图

| 线路 | 角色 |
| --- | --- |
| **DeepSeek-V3** | 大 MoE + MLA（下一章） |
| **Mixtral** | Top-2 SwiGLU experts |
| **Llama 3.2 ~1B（本课）** | 先稠密 |

### 本课

```bash
python chapters/01-inference/ep14_moe.py
```

## 动手

```bash
python chapters/01-inference/ep14_moe.py
```

## 下一章

**MLA**（`/cn/mla/`）：按 DeepSeek 方式压缩 KV。
