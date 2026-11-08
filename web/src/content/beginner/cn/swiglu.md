## 一句话
SwiGLU 是现代前馈块：gate × up，再 down——三条无偏置线性层。

## 开始之前
- GQA 已完成。约 12 分钟。

## 用人话讲
老教程常见 Linear → GELU → Linear。Llama 风格更常用：
`down(silu(gate(x)) * up(x))`。

## 图或公式
$$\mathrm{FFN}(x)=W_d\big(\mathrm{SiLU}(W_g x)\odot W_u x\big)$$

## 对应代码
- `arche/mlp.py`
- `chapters/01-inference/ep07_swiglu.py`

## 现在就做
```bash
python chapters/01-inference/ep07_swiglu.py
```

## 如果翻车了
- 形状不对：检查 intermediate_dim 是否与三条线性层一致。

## 先别深挖
MoE（多专家）在第 2 部分。

## 下一章
**Transformer 块**（`?c=block`）。
