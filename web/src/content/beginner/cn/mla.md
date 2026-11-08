## 要点

**多头潜注意力（MLA）**把 KV 压成低秩**潜变量**，外加一小段 RoPE 流，而不是缓存完整每头 K/V。GQA 之后长上下文 cache 仍可能爆；DeepSeek-V2/V3 在这里推 MLA，Llama/Qwen 稠密主线仍是 GQA。和 MoE 合读：一个管容量，一个管 KV。

本章是**形状入门**，不是 bit 级 DeepSeek 复刻。选潜变量秩，投影，缓存 latent + RoPE key，注意力时再展开。

### Cache 形状

[`MLACache` / `MultiHeadLatentAttention`](https://github.com/chaxus/arche/blob/main/arche/mla.py)：

```python
@dataclass
class MLACache:
    kv_latent: torch.Tensor  # (batch, seq, kv_lora_rank)
    k_rope: torch.Tensor     # (batch, seq, rope_dim)
```

当 rank ≪ heads×dim 时，远小于 `(batch, kv_heads, seq, head_dim)` 全量 KV。

### 相对 GQA

| | GQA（Llama 路） | MLA（DeepSeek 路） |
| --- | --- | --- |
| 存什么 | 共享 KV 头 | Latent + 小 RoPE |
| 目标 | 砍头数 | 再砍秩 |
| 替换成本 | 都是结构级 |

### 从零

| 步骤 | 做什么 |
| --- | --- |
| 1 | 选 `kv_lora_rank`，可选 `q_lora_rank` |
| 2 | RoPE 维与内容 latent 解耦 |
| 3 | 解码步间缓存 latent |
| 4 | 在同等质量条下对比相对 GQA 的显存 |

### 自己核对

- 形状一致；cache 沿 seq 追加。
- cache 更小；质量接近你产品条下的全量 KV。
- 别把 MLA 套在 Llama GQA 权重下。

### 杠杆

| 杠杆 | 影响 |
| --- | --- |
| 更小 latent | 省显存 / 冒质量风险 |
| + paging | 与 GQA 栈同一类服务技巧 |
| 套错权重 | 静默错注意力 |

### 业界地图

| 线路 | 注意力 / KV |
| --- | --- |
| **DeepSeek-V2 / V3** | MLA（+ MoE 躯干） |
| **Llama 3 / Qwen 稠密** | GQA 主线 |

### 本课

```bash
python chapters/01-inference/ep15_mla.py
```

## 动手

```bash
python chapters/01-inference/ep15_mla.py
```

## 下一章

**前沿检查点**（`/cn/frontier-ckpt/`）。
