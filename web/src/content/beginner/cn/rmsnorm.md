## 要点

从零训深网时，没有归一化，数值很容易炸或塌——嵌入之后第一道「稳住尺子」就是它。

### 1. 有什么作用

**RMSNorm**：对每个 token 向量算 RMS，除掉它，再乘可学习 `γ`；**不减均值**（异于 LayerNorm）。Llama 风格 **pre-norm**：注意力前、MLP 前各一次。

$$\\mathrm{RMSNorm}(x)=x/\\sqrt{\\mathrm{mean}(x^2)+\\varepsilon}\\odot\\gamma$$

### 2. 从零开始：怎么生成 / 接到哪

| 步骤 | 做什么 |
| --- | --- |
| **1** | 每个需要稳住的入口建一个 `RMSNorm(D)`，`γ` 长度 = `D` |
| **2** | `γ` 常初始化为 1 |
| **3** | 按 pre-norm 接到 Attention / MLP 前 |
| **4** | 与整网一起训；推理原样前向 |

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 进出同形状；`ε` 防除零；位置符合配方（pre vs post） |
| **软** | `γ=1` 时 RMS≈1（公式体检）；训练少炸、少 NaN |
| **软** | 最终看整模 loss——Norm 很少单独打分 |

### 4. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| LayerNorm→RMSNorm | 少算均值，Llama/Qwen 默认；换了要和权重匹配 |
| pre-norm vs post-norm | 现代深网几乎 pre；post 更难训深 |
| 学 `γ` / 冻 `γ` | 微调策略；乱改易伤稳定性 |
| 与 lr/初始化搭配 | 从零训能否收敛的关键 |

### 5. 业界 · 开源

| 系 | 做法 |
| --- | --- |
| **Llama 3/3.2、Qwen** | RMSNorm + pre-norm |
| **GPT-2 教学老栈** | LayerNorm + 常 post-norm |
| **DeepSeek 等** | 主干仍 RMS/pre 一族（细节看配方） |

### 6. 本课

`arche/norms.py`：

```bash
python chapters/01-inference/ep04_rmsnorm.py
```

## 动手

```bash
python chapters/01-inference/ep04_rmsnorm.py
```

## 下一章

**RoPE**（`/cn/rope/`）：从零把位置写进 Q/K。
