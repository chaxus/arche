## 要点

**是什么（What）：** **自回归生成**——把 logits 变成文本的循环：打分 → 选下一 id → 拼上 → 停止。

**为什么（Why）：** 训好/加载好的栈只吐分数。产品差在采样与停止规则；共同点是这条环。没有它只有「分类器形」前向，没有聊天。

**怎么做（How）：** 最后位置 logits → 可选 temperature / top-k → 贪心或采样 → 追加 → 遇 EOS 或达长度停下。KV cache（下一部分）让同一环变快。

### 1. What — 骨架

[`arche/generate.py`](https://github.com/chaxus/arche/blob/main/arche/generate.py) 大致：

```python
# 每一步
logits, caches = model(ids_or_last_token, caches=caches, start_pos=...)
next_id = sample(logits[:, -1, :], temperature=..., top_k=...)  # 或 argmax
# 追加；EOS / max_new_tokens 时 break
```

Prefill 一次跑完 prompt；开 cache 时 decode 每次只喂新 token。

### 2. Why — 旋钮 ≠ 权重

| 旋钮 | 改变的是 |
| --- | --- |
| Temperature / top-k / top-p | 挑选风格——不是模型小数 |
| 贪心 argmax | 确定、常偏闷 |
| EOS / 最大长度 | 体验与安全边界 |
| Chat template | 角色如何变成 id（后章） |

### 3. How — 从零

| 步骤 | 做什么 |
| --- | --- |
| 1 | 用**配对**分词器编码 prompt |
| 2 | 前向 → 取最后时刻 logits |
| 3 | 采样或 argmax → 追加 id |
| 4 | 直到停止；id 解码回文本 |

### 4. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 会停；只见过去；合法词表 id |
| **软** | 真权重下短补全连贯 |
| **软** | 同 prompt + 贪心 ≈ 可复现 |

### 5. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 只改采样 | 不重训也能改行为 |
| 无 KV cache | 正确但体感 O(T²)—下一部分修 |
| 约束解码 | 结构化输出 |
| 停止规则差 | 跑飞或截断 |

### 6. 本课

```bash
python chapters/01-inference/ep11_generate.py
```

## 动手

```bash
python chapters/01-inference/ep11_generate.py
```

## 下一章

**HF 对齐**（`/cn/hf-align/`）：证明手写栈对齐 HF。
