## 要点

注意力混合过各个位置之后，每个位置还要过一个**按位置的 MLP** 变换特征。

现代栈用 **SwiGLU**，而不是简单的 GELU MLP：

$$\mathrm{FFN}(x)=W_d\big(\mathrm{SiLU}(W_g x)\odot W_u x\big)$$

三条**无偏置**线性层：**gate**、**up**、**down**。gate 走 SiLU，再与 up 相乘，最后 down 投回。宽度思路与 Llama / Qwen dense 一致。

## 动手

```bash
python chapters/01-inference/ep07_swiglu.py
```

预期：进出同为 `embed_dim`，中间层更宽。

## 下一章

**Transformer 块**（`?c=block`）：Norm + 注意力 + SwiGLU + 残差拼起来。
