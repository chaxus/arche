## 要点

`TokenEmbedding`：id → 向量 `(vocab, dim)`。这里不加绝对位置表——位置靠后面的 **RoPE**。Llama 3.2 与 LM head 绑权。

## 动手

```bash
python chapters/01-inference/ep03_embedding.py
```

## 下一章

**RMSNorm**（`?c=rmsnorm`）。
