## 要点

`ids → Embed → [Block×N] → RMSNorm → Linear(vocab)`。最后一层分数即下一 token logits；与嵌入绑权。

## 动手

```bash
python chapters/01-inference/ep09_stack.py
```

## 下一章

**加载权重**（`?c=weights`）。
