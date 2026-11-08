## 要点

**RMSNorm** 用均方根缩放（不减均值）。**Pre-norm**：注意力前、MLP 前各一次。

$$\mathrm{RMSNorm}(x)=x/\sqrt{\mathrm{mean}(x^2)+\varepsilon}\odot\gamma$$

## 动手

```bash
python chapters/01-inference/ep04_rmsnorm.py
```

## 下一章

**RoPE**（`?c=rope`）。
