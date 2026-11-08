## 要点

**是什么（What）：** **分组查询注意力（GQA）**——Q 头多于 KV 头；多组 Q **共享**一组 KV。仍是因果注意力。

**为什么（Why）：** 纯 MHA 每个 Q 头存一对 KV → 解码显存随长度爆。GQA 大幅砍 cache，质量损失通常可控。Llama 3.2 ~1B：**32 Q / 8 KV**。

**怎么做（How）：** 选可整除的 `n_q` / `n_kv`，投影，RoPE，对 KV **repeat_interleave** 对齐 Q，再 SDPA；cache 只存 `n_kv`。

### 1. What — 本仓库形状

[`CausalSelfAttention`](https://github.com/chaxus/arche/blob/main/arche/attention.py)：

```python
q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
# ... RoPE ...
k_full = new_cache.k.repeat_interleave(self.n_rep, dim=1)
v_full = new_cache.v.repeat_interleave(self.n_rep, dim=1)
attn = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=...)
```

`n_rep = num_heads // num_kv_heads`。Cache 追加**窄** K/V；只在 matmul 时广播。

### 2. Why — MHA / GQA / MQA / MLA

| 布局 | 想法 | 代价 |
| --- | --- | --- |
| **MHA** | `n_kv = n_q` | 最强，cache 最大 |
| **GQA** | `n_kv` 整除 `n_q` | 开源中型默认 |
| **MQA** | `n_kv = 1` | 极致省 cache |
| **MLA** | 压缩潜变量 KV | 前沿（DeepSeek）—后部 |

改头数 ⇒ **重训**（否则对不上 checkpoint）。

### 3. How — 从零

| 步骤 | 做什么 |
| --- | --- |
| 1 | 固定 `embed_dim = n_q × head_dim` |
| 2 | 选能整除 `n_q` 的 `n_kv` |
| 3 | 无 bias 的 Q/K/V/O（Llama 约定） |
| 4 | Prefill 用因果；有 cache 时看全部过去 key |

### 4. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 分组合法；因果；权重头数对齐 config |
| **软** | Cache 字节约 MHA 的 `n_kv/n_q` |
| **软** | 同宽度下相对 MHA 质量站得住 |

### 5. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 更少 KV 头 | 解码显存 / 速度 |
| 太少 | 质量断崖 |
| MLA 路线 | 另一套压缩（Frontier） |

### 6. 本课

```bash
python chapters/01-inference/ep06_gqa.py
```

## 动手

```bash
python chapters/01-inference/ep06_gqa.py
```

## 下一章

**SwiGLU**（`/cn/swiglu/`）：稠密 FFN。
