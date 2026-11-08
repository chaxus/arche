## 要点

层一深，数值容易炸或塌。**归一化**把向量尺度拉回可控范围。

**RMSNorm**（均方根归一化）是这里的现代默认：

- 对每个 token 向量，在最后一维算 RMS  
- 除以 RMS（再加一点 `ε`）  
- 再乘可学习缩放 `γ`  
- **不减均值**（和经典 LayerNorm 不同）——更省，实践里也够稳  

我们用 **pre-norm**：注意力前一次、MLP 前一次。Llama / Qwen 一类 dense 栈就是这样。

$$\mathrm{RMSNorm}(x)=x/\sqrt{\mathrm{mean}(x^2)+\varepsilon}\odot\gamma$$

## 动手

```bash
python chapters/01-inference/ep04_rmsnorm.py
```

预期：进出同形状；`γ` 初始为 1 时，归一化后每 token 的 RMS 约等于 1。

## 下一章

**RoPE**（`?c=rope`）：用旋转把位置写进 Q/K。
