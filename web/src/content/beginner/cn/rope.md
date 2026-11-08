## 要点

注意力靠 Q 和 K 做点积。若完全没有位置，「先 A 后 B」和「先 B 后 A」看起来一样。

**RoPE**（旋转位置编码）按**位置**给每个 Q/K 转一个角。相对距离会自然进点积——不必在嵌入上再贴一张大正弦位置表。

Llama 3.2 用很大的基频（`rope_theta`）照顾长文；后面还有**波长缩放**专讲更长上下文。

本页可拖互动图。代码里：预计算 cos/sin，`rotate_half`，再 `q' = q·cos + rotate(q)·sin`（K 同理）。

## 动手

```bash
python chapters/01-inference/ep05_rope.py
```

预期：演示一次 RoPE 应用并核对形状。

## 下一章

**GQA**（`?c=gqa`）：KV 头更少的注意力。
