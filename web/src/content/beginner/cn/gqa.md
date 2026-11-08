## 要点

**GQA**：KV 头少于 query 头（1B：32 Q / 8 KV），解码 cache 更小。见本页互动图。

流程：投影 → RoPE → 重复 KV → 因果注意力 → 输出投影。

## 动手

```bash
python chapters/01-inference/ep06_gqa.py
```

## 下一章

**SwiGLU**（`?c=swiglu`）。
