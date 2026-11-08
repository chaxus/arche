## 要点

**是什么（What）：** **数值核对**——同一短 prompt 分别过 arche 与 Hugging Face Llama 3.2，比较 logits（最大绝对差 + argmax 是否一致）。

**为什么（Why）：** 第 1 部分收官证明：键映射、RMSNorm、RoPE、GQA、SwiGLU、整网布局不是「形状对、公式错」。别让后面的服务/优化章把 bug 藏起来。

**怎么做（How）：** 固定 dtype；两边加载；编码一条 prompt；前向；出报告。无权重/未登录时干净跳过。

### 1. What — 报告

[`compare_logits_to_hf`](https://github.com/chaxus/arche/blob/main/arche/align_hf.py)：

```python
hf_logits = hf(ids).logits
arche_logits, _ = arche(ids)
max_abs = (arche_logits - hf_logits).abs().max()
argmax_match = torch.equal(arche_logits.argmax(-1), hf_logits.argmax(-1))
```

`run_align_or_explain` 在 HF 权重不可达时返回 `None`——**跳过 ≠ 课逻辑坏了**。

### 2. Why — 这道门

| 失败形态 | 常见原因 |
| --- | --- |
| `max_abs` 很大 | 映射、RoPE theta、norm 或头布局错 |
| Argmax 不一致 | 同上——外加 dtype / 设备不一致 |
| 跳过 | 未登录 / 门禁 / 离线——修环境，不是公式 |

### 3. How — 从零

| 步骤 | 做什么 |
| --- | --- |
| 1 | 需要则登录；模型 id 与加载章相同 |
| 2 | 实验用 float32 CPU 比较更稳 |
| 3 | 短 prompt（如“The cat sat”） |
| 4 | 在容差内断言，或打印跳过原因 |

### 4. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 有权重：argmax 一致；`max_abs` 低于实验容差 |
| **硬** | 无权重：显式跳过，不当假通过 |
| **软** | 更长 prompt / 更多位置留给以后 CI |

### 5. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 失败 → 回头查 | 映射 / RoPE / norm，先别「优化」 |
| 通过 | 可进对齐（SFT/DPO）与服务 |
| 引擎 CI | 同类门禁卡 vLLM 级数回归 |

### 6. 本课

```bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
```

第 1 部分至此结束。第 2 部分转向底座猜词能力（SFT/DPO）。

## 动手

```bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
```

## 翻车

无权重/未登录 → 跳过表示「下不下来」，不是「数学错了」。

## 下一章

**SFT · DPO**（`/cn/sft-dpo/`）：在会猜词之后做对齐。
