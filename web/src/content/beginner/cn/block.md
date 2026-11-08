## 要点

```
x → RMSNorm → Attn → +x → RMSNorm → SwiGLU → +x
```

进出同形状，循环堆叠（1B：16 层）。

## 动手

```bash
python chapters/01-inference/ep08_block.py
```

## 下一章

**堆叠 + LM head**（`?c=stack`）。
