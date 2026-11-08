## 要点

循环：取最后一步 logits →（可选 temperature / top-k）→ 采样或贪心 → 拼上 → EOS/长度停。

## 动手

```bash
python chapters/01-inference/ep11_generate.py
```

## 下一章

**KV 缓存**（`?c=kv-cache`）。
