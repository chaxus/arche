## 一句话
RMSNorm 让层输入保持合适的尺度，又不必像 LayerNorm 那样先减均值。

## 开始之前
- 嵌入章已完成。约 12 分钟。

## 用人话讲
网络很深时，数字容易爆掉或消失。**RMSNorm** 用均方根大小去除尺度，再乘可学习增益；不做减均值，更便宜，对现代 LLM 也够稳。
每个 block 里它出现在注意力**之前**和 MLP**之前**（pre-norm）。

## 图或公式
$$\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\mathrm{mean}(x^2)+\varepsilon}} \odot \gamma$$

## 对应代码
- `arche/norms.py`
- `chapters/01-inference/ep04_rmsnorm.py`

## 现在就做
```bash
python chapters/01-inference/ep04_rmsnorm.py
```

## 如果翻车了
- 实验里出现 NaN：多半改了 epsilon——用 git 恢复脚本。

## 先别深挖
文献里所有 LayerNorm 变体。

## 下一章
**RoPE**（`?c=rope`）。
