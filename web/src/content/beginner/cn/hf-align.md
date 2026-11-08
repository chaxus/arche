## 要点

**HF 对齐**用同一条短 prompt 分别过 arche 和 Hugging Face Llama 3.2，再比 logits——最大绝对差和 argmax 是否一致。这是第 1 部分的收官证明：键映射、RMSNorm、RoPE、GQA、SwiGLU、整网布局不是「形状对、公式错」。固定 dtype，两边加载，编码一条 prompt，前向出报告；无权重或未登录时干净跳过。

### 脚本输出什么

[`compare_logits_to_hf`](https://github.com/chaxus/arche/blob/main/arche/align_hf.py)：

```python
hf_logits = hf(ids).logits
arche_logits, _ = arche(ids)
max_abs = (arche_logits - hf_logits).abs().max()
argmax_match = torch.equal(arche_logits.argmax(-1), hf_logits.argmax(-1))
```

`run_align_or_explain` 在 HF 权重不可达时返回 `None`——**跳过 ≠ 课逻辑坏了**。

### 数字对不上时

| 失败形态 | 常见原因 |
| --- | --- |
| `max_abs` 很大 | 映射、RoPE theta、norm 或头布局错 |
| Argmax 不一致 | 同上——外加 dtype / 设备不一致 |
| 跳过 | 未登录 / 门禁 / 离线——修环境，不是公式 |

### 怎么跑

| 步骤 | 做什么 |
| --- | --- |
| 1 | 需要则登录；模型 id 与加载章相同 |
| 2 | 实验用 float32 CPU 比较更稳 |
| 3 | 短 prompt（如“The cat sat”） |
| 4 | 在容差内断言，或打印跳过原因 |

### 自己核对

| 必须成立 | 有权重：argmax 一致；`max_abs` 低于实验容差 |
| 必须成立 | 无权重：显式跳过，不当假通过 |
| 尽量做到 | 更长 prompt / 更多位置留给以后 CI |

### 失败或通过后

| 杠杆 | 影响 |
| --- | --- |
| 失败 → 回头查 | 映射 / RoPE / norm，先别「优化」 |
| 通过 | 可进对齐（SFT/DPO）与服务 |
| 引擎 CI | 同类门禁卡 vLLM 级数回归 |

### 本课

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
