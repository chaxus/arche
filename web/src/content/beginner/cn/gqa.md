## 要点

经典多头：每个 query 头配自己的 K/V 头。解码时 **KV 缓存**（后面章节）会很大。

**GQA**（分组查询注意力）：KV 头少于 query 头，多个 query **共用**一组 K/V。Llama 3.2 1B 是 **32 个 Q / 8 个 KV**——质量差不多，缓存更小。

仍然是**因果**：位置 `t` 只能看见 `≤ t`。

代码流程：投影 Q/K/V → 拆头 → RoPE → `repeat_interleave` 扩 KV → 缩放点积注意力 → 输出投影。

## 动手

```bash
python chapters/01-inference/ep06_gqa.py
```

预期：一次小前向，打印头数与输出形状。可看本页互动图。

## 下一章

**SwiGLU**（`?c=swiglu`）：每一块里的前馈半边。
