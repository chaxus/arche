## 要点

**是什么（What）：** **RoPE**（旋转位置编码）——按 token 位置旋转 Q、K，让**相对距离**进入注意力点积。

**为什么（Why）：** 嵌入说明「是哪个词」；注意力还要「在哪」。嵌绝对位置表是旧派；现代稠密 Llama/Qwen/DeepSeek 在 Q/K 上用 RoPE。

**怎么做（How）：** 选定 `rope_theta`，缓存 cos/sin，`rotate_half` 后与 Q/K 组合；长上下文缩放是后话。

### 1. What — 施加一步

仓库（[`arche/rope.py`](https://github.com/chaxus/arche/blob/main/arche/rope.py)）：

```python
def apply_rotary_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)
```

`build_rope_cache` 预计算 `0 .. seq_len-1` 的 cos/sin。带 cache 解码时按 `start_pos` 切片。

### 2. Why — 设计理由

| 选择 | 原因 |
| --- | --- |
| 只旋 Q/K | V 保持内容；位置进分数 |
| 旋转带来相对性 | Dot(q_m, k_n) 依赖 `m-n` |
| 大 `theta`（Llama 3: 50 万） | 比旧 10k 更能撑长上下文 |
| 对齐 checkpoint 的 theta/scale | PE 错了，加载权重后会静默变差 |

### 3. How — 从零

| 步骤 | 做什么 |
| --- | --- |
| 1 | `head_dim` 偶数；由 `theta` 建 inv-freq |
| 2 | 缓存到最大序列（显存折中） |
| 3 | Q/K 投影之后、SDPA 之前施加 |
| 4 | 长上下文：Llama-3 波长平滑（后章 `rope-scaling`） |

### 4. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 形状不变；头维偶数；Q、K 都施加 |
| **硬** | checkpoint 的 `theta` / scale 一致 |
| **软** | 近邻与远邻分数有别 |

### 5. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| Theta / scaling | 长上下文表现 |
| 绝对 PE → RoPE | 现代默认；勿与外来权重混家族 |
| 跳过 RoPE | 模型难以可靠用顺序 |

### 6. 本课

```bash
python chapters/01-inference/ep05_rope.py
```

## 动手

```bash
python chapters/01-inference/ep05_rope.py
```

## 下一章

**GQA**（`/cn/gqa/`）：省 KV cache 的头布局。
