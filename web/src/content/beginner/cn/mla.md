## 要点

**MLA** 缓存低秩 latent（加一小段 RoPE 流），而不是每头完整 K/V——解码更省内存。

## 动手

```bash
python chapters/01-inference/ep15_mla.py
```

## 下一章

**设备 · INT8 · 对话**（`?c=device-quant-chat`）。
