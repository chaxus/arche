## 要点

**是什么（What）：** **混合专家（MoE）**——用**多个 expert + router**替换稠密 SwiGLU；每个 token 只跑 **top-k**。

**为什么（Why）：** 参数容量可以很大，但**每 token**算力仍接近稠密。**DeepSeek-V3** 开源故事大体是 **MoE（容量）+ MLA（KV）**；Mixtral 是另一条开源 MoE。本课 Llama 3.2 例子故意先走**稠密**——会读稠密 FFN 后再看这张地图。

**怎么做（How）：** 选 `E` / `top_k`；experts + gate；真训时加负载均衡；稀疏推理；checkpoint 里分片 experts。

### 1. What — 前向

[`MoESwiGLU`](https://github.com/chaxus/arche/blob/main/arche/moe.py)：

```python
logits = self.gate(x)
probs = F.softmax(logits, dim=-1)
top_probs, top_idx = probs.topk(self.top_k, dim=-1)
# 对选中的 SwiGLU expert 加权求和
```

解码器脊柱不变；只换 MLP 槽位。

### 2. Why — 稠密 vs 稀疏

| | 稠密 SwiGLU | MoE |
| --- | --- | --- |
| 参数 | 一个 FFN | `E` × FFN |
| 每 token FLOPs | 完整 FFN | ≈ `top_k` 个 expert |
| 失败模式 | — | 无均衡时 expert 坍塌 |

### 3. How — 从零

| 步骤 | 做什么 |
| --- | --- |
| 1 | 保留注意力 / norm；MLP → MoE |
| 2 | 选 `E`、`top_k`（如 Mixtral 类 top-2） |
| 3 | 认真训练时加负载均衡损失 |
| 4 | 规划 expert 并行 / 分片加载 |

### 4. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 路由合法；合并权重合理 |
| **软** | expert 负载均衡；质量/容量折中 OK |
| **软** | checkpoint 映射覆盖全部 expert |

### 5. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 更大 `E` / `k` | 容量 vs 算力 |
| 均衡损失 | 对抗坍塌 |
| Expert 并行 | 多卡工程成本 |

### 6. 业界地图

| 线路 | 角色 |
| --- | --- |
| **DeepSeek-V3** | 大 MoE + MLA（下一章） |
| **Mixtral** | Top-2 SwiGLU experts |
| **Llama 3.2 ~1B（本课）** | 先稠密 |

### 7. 本课

```bash
python chapters/01-inference/ep14_moe.py
```

## 动手

```bash
python chapters/01-inference/ep14_moe.py
```

## 下一章

**MLA**（`/cn/mla/`）：按 DeepSeek 方式压缩 KV。
