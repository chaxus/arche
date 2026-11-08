## 要点

### 1. 为什么要 GQA

完整 MHA：每个 Q 头自带 KV → 解码时 **KV 缓存**涨得快。

### 2. GQA 做什么

**分组查询注意力**：KV 头少于 Q 头，多组 Q **共用**一组 KV。仍是**因果**（`t` 只看 `≤ t`）。

流程：投影 → 拆头 → RoPE → 扩 KV → SDPA → 输出投影。见本页图。

### 3. 注意力布局家族

| 布局 | 人话 | 今天 |
| --- | --- | --- |
| MHA | Q 头数 = KV 头数 | 经典 |
| MQA | 全体 Q 共用一套 KV | 极致省 |
| **GQA** | 分组共用 | **Llama 3.2 1B：32Q / 8KV** |
| MLA | 把 KV 压成 latent | DeepSeek 风（后面章） |

### 4. 业界实际

Llama 3.2 1B 用 GQA；很多开放 7B–70B 为了解码显存也用 GQA。

### 5. 怎样算对

- `n_rep = n_q / n_kv` 正确  
- prefill / decode 的因果设置正确  
- 缓存比满 MHA 小，质量别塌光  

### 6. 本课怎么做

`arche/attention.py` 的 `CausalSelfAttention`。实验打印头数与形状。

## 动手

```bash
python chapters/01-inference/ep06_gqa.py
```

## 下一章

**SwiGLU**（`?c=swiglu`）：前馈那半边。
