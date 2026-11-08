## 要点

**KV cache** 在 prefill 之后保留每层的 K/V，解码时 **追加** 一步，而不是每个新 token 都把整段历史重算一遍。没有 cache 的正确生成环在每一步上对 growing 历史做注意力——实验长度还行，产品长度会拖死。生产解码器都会缓存 KV；GQA、MLA、paging 是在此之上再缩小存量。

### 结构

[`KVCache`](https://github.com/chaxus/arche/blob/main/arche/cache.py)：

```python
@dataclass
class KVCache:
    k: torch.Tensor  # (batch, kv_heads, seq, head_dim)
    v: torch.Tensor

    def append(self, k, v):
        return KVCache(k=torch.cat([self.k, k], dim=-2), v=torch.cat([self.v, v], dim=-2))
```

注意力（[`CausalSelfAttention`](https://github.com/chaxus/arche/blob/main/arche/attention.py)）对新 Q/K 做 RoPE，并入 cache，再对**完整**缓存 K/V 做 SDPA。

### 成本长什么样

| 模式 | 每个新 token 的活 |
| --- | --- |
| 无 cache | 整段过去 id 重跑嵌入 + 全栈 |
| 有 cache | 只算新位置 Q/K/V + 对 cache 做注意力 |
| 显存 | 随 `层数 × n_kv × seq × dim` 涨 |

GQA 已砍 `n_kv`；paging（后章）解决多用户碎片。

### 解码怎么接

| 步骤 | 做什么 |
| --- | --- |
| 1 | 每层一个 cache 槽 |
| 2 | Prefill：`cache=None`，写入整段 prompt K/V |
| 3 | Decode：传入 cache + `start_pos`，追加一步 |
| 4 | 绝不串层混用 cache |

### 自己核对

| 必须成立 | 玩具上与全量重算数值一致 |
| 必须成立 | 不串层；只在 seq 维增长 |
| 尽量做到 | 目标上下文 × batch 装得下 |

### 省显存的杠杆

| 杠杆 | 影响 |
| --- | --- |
| GQA / MLA / KV 量化 | 更小 cache |
| Paging | 多用户打包 |
| 无 cache | 只适合玩具长度 |

### 本课

```bash
python chapters/01-inference/ep12_kv_cache.py
```

## 动手

```bash
python chapters/01-inference/ep12_kv_cache.py
```

## 下一章

**RoPE 缩放**（`/cn/rope-scaling/`）。
