## 要点

一个 **Transformer 块**就是可重复的三明治：

```
x → RMSNorm → 注意力 → + 残差
  → RMSNorm → SwiGLU → + 残差
```

进出同形状，所以叠 `N` 次（Llama 3.2 1B：**16** 层）。现代 dense 用 **pre-norm**（子层**前**归一化）；残差让梯度还能流。

注意力可带 KV cache，解码时用（马上会用到）。

## 动手

```bash
python chapters/01-inference/ep08_block.py
```

预期：一块前向；残差路径上形状不变。

## 下一章

**堆叠 + LM head**（`?c=stack`）：嵌入 → 多层块 → 词表分数。
