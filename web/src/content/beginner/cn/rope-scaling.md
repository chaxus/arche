## 要点

RoPE 的角度是按训练长度调的。上下文拉太长，相对位置可能「绕坏」。

**Llama 3.2 风格 RoPE 缩放**：保持很大的 `rope_theta`，用 factor / low / high 频段平滑拉长**长波长**（与 HF 公式兼容）。近距离行为仍熟悉，远距离被拉伸，长 prompt 才好用。

本章讲的是**调度表**，不是另造一种注意力。

## 动手

```bash
python chapters/01-inference/ep13_rope_scaling.py
```

预期：对比未缩放 / 已缩放的频率或形状打印。

## 下一章

**MoE**（`?c=moe`）：稀疏专家，而不是一个巨大 MLP。
