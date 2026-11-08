## 要点

### 1. 为什么要归一化

层一深，数值容易炸或塌。归一化把尺度拉回可控。

### 2. RMSNorm 做什么

对每个 token 向量（最后一维）：

1. 算 RMS  
2. 除以 RMS（+ 很小的 `ε`）  
3. 乘可学习 `γ`  
4. **不减均值**（和 LayerNorm 不同）

我们用 **pre-norm**：注意力前一次、MLP 前一次。

$$\mathrm{RMSNorm}(x)=x/\sqrt{\mathrm{mean}(x^2)+\varepsilon}\odot\gamma$$

### 3. 你会听到的名字

| 名字 | 人话 | 今天 |
| --- | --- | --- |
| LayerNorm | 减均值再缩放 | 偏 GPT-2 老故事 |
| **RMSNorm** | 只按 RMS 缩放 | **Llama / Qwen 默认** |
| Pre-norm / post-norm | 子层前还是后归一化 | 现代栈：pre-norm |

### 4. 业界实际

Llama 3.2、Qwen dense、多数新开放解码器：RMSNorm + pre-norm。

### 5. 怎样算对

- 进出同形状  
- `γ=1` 时，归一化后 RMS ≈ 1  
- 稳住训练，但不等于整模型的全部故事  

### 6. 本课怎么做

`arche/norms.py` 的 `RMSNorm`。实验打印形状与每 token RMS。

## 动手

```bash
python chapters/01-inference/ep04_rmsnorm.py
```

## 下一章

**RoPE**（`?c=rope`）：把位置写进 Q/K。
