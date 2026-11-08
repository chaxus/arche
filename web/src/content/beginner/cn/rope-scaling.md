## 要点

### 1. 为什么要缩放 RoPE

角度按训练长度调的；上下文拉太长可能「绕坏」。

### 2. Llama 风格缩放做什么

保持大 `rope_theta`；按 factor/low/high **平滑拉长波长**（与 HF 兼容）。

### 3. 版图

| 方法 | 人话 |
| --- | --- |
| NTK / YaRN / … | 其他长上下文 RoPE 配方 |
| **Llama 3 波长平滑** | 本课对照的 |
| 更长数据直接训 | 不只靠公式 |

### 4. 业界实际

Llama 3.2 官方配置里的长上下文调度。

### 5. 怎样算对

- 近距离仍熟悉  
- 远距离被拉伸而不是乱绕  

### 6. 本课怎么做

`arche/rope.py` 的缩放辅助 + 本章脚本。

## 动手

```bash
python chapters/01-inference/ep13_rope_scaling.py
```

## 下一章

**MoE**（`?c=moe`）。
