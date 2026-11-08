## 要点

整模路径：

`token id → 嵌入 → Block×N → 最终 RMSNorm → Linear(词表)`

最后的线性层是 **LM head**：每个位置给词表里每个词一个分数（**logits**）。取最大（或采样）就是下一个 token。

Llama 3.2 常把 LM head 与嵌入表 **绑权**——一张矩阵两个用途。

## 动手

```bash
python chapters/01-inference/ep09_stack.py
```

预期：随机小模型前向，logits 形状 `(batch, seq, vocab)`。

## 下一章

**加载权重**（`?c=weights`）：用真 Llama 张量替换随机初始化。
