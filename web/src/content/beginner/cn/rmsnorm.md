## 要点

**RMSNorm** 按 token 稳住激活尺度：除以均方根，再乘可学习的 `γ`；不减均值、无 bias。深的 next-token 栈没有归一化容易炸或塌，Llama 风格 **pre-norm**（先 norm 再 attention / MLP）让从零堆深度训得动。实践上在每个 pre-norm 位点放 `RMSNorm(D)`，`γ` 初值约 1，与整网一起训练。

### 公式与代码

\[
\mathrm{RMSNorm}(x)=\frac{x}{\sqrt{\mathrm{mean}(x^2)+\varepsilon}}\odot\gamma
\]

仓库（[`arche/norms.py`](https://github.com/chaxus/arche/blob/main/arche/norms.py)）：

```python
variance = x_f.pow(2).mean(dim=-1, keepdim=True)
x_normed = x_f * torch.rsqrt(variance + self.eps)
return (self.weight.float() * x_normed).to(orig_dtype)
```

在 float32 里算再 cast 回去——与 HF Llama 同一类稳定性手法。

### 相对 LayerNorm 与摆放位置

| 选择 | 原因 |
| --- | --- |
| 不减均值 | 稍省；对齐 Llama/Qwen 菜谱 |
| 只有可学习 `γ` | 单位 RMS 后再按通道缩放 |
| **Pre-norm** | 稳住 attn/MLP *输入*，深度才活 |
| 嵌入之后、积木之前 | 有了向量就要先「稳住尺子」 |

GPT-2 教学常是 LayerNorm + post-norm。本课例子是 **RMSNorm + pre-norm**。

### 怎么接进网络

| 步骤 | 做什么 |
| --- | --- |
| 1 | 注意力前、SwiGLU 前各一个（见 block） |
| 2 | LM head 前再一个最终 RMSNorm（见 stack） |
| 3 | `ε` 对齐 checkpoint（`norm_eps`） |
| 4 | 训或加载；勿在外来权重下 LN↔RMSN 混用 |

### 自己核对

- 形状不变；有 `ε`；pre-norm 位点对齐菜谱。
- `γ=1` 时 RMS≈1；深前向更少 NaN。
- 整模 loss / HF 对齐才是终审。

### 对整模的影响

| 杠杆 | 影响 |
| --- | --- |
| LN → RMSN | 结构级变更 ⇒ 重训 |
| Pre vs post | 深度可训性 |
| `γ` / lr | 耦合收敛 |

### 本课

```bash
python chapters/01-inference/ep04_rmsnorm.py
```

## 动手

```bash
python chapters/01-inference/ep04_rmsnorm.py
```

## 下一章

**RoPE**（`/cn/rope/`）：把位置写进 Q/K。
