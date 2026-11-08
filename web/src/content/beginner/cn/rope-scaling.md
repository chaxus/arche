## 要点

**是什么（What）：** **RoPE 缩放**——调整旋转频率（Llama 3 风格波长平滑），让超出原训练窗的位置仍可用。

**为什么（Why）：** 短上下文预训练 + 天真拉长下标 → 旋转缠坏，远端质量塌。长上下文产品需要**菜谱扩展**，不是「把 max_seq 改大就行」。

**怎么做（How）：** 对齐 checkpoint 的 `theta` + scale 因子；可选长数据继续训；短/长都要评。

### 1. What — 代码钩子

[`llama3_inv_freq`](https://github.com/chaxus/arche/blob/main/arche/rope.py) / `build_rope_cache(..., scale_factor=..., low_freq_factor=..., high_freq_factor=..., original_max_position=...)`。

`scale_factor == 1.0` 时是普通 RoPE；否则按 HF Llama-3 方式缩放/平滑波段——**与宣称 128k 级配置同一套数学**。

### 2. Why — 不是免费变长

| 错觉 | 现实 |
| --- | --- |
| 只拉长下标 | 高频带会乱 |
| 更大窗口免费 | KV 显存随长度涨 |
| 缩放 = 长程知识 | 硬长程任务仍要数据 / 续训 |

### 3. How — 从零

| 步骤 | 做什么 |
| --- | --- |
| 1 | 从 ckpt config 读官方 rope scaling 字段 |
| 2 | 用这些因子建 cos/sin |
| 3 | 保留短上下文健全性检查 |
| 4 | 宣称 128k 前做 long-bench / needle |

### 4. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 因子与 checkpoint 一致 |
| **软** | 近处仍稳；远处可用 |
| **软** | long-bench 达到产品条 |

### 5. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 缩放设错 | 长（甚至短）都坏 |
| YaRN / NTK / … | 其他开源菜谱——别套在 Llama 权重下混用 |
| 更长窗口 | 永远要付 KV 账 |

### 6. 本课

```bash
python chapters/01-inference/ep13_rope_scaling.py
```

## 动手

```bash
python chapters/01-inference/ep13_rope_scaling.py
```

## 下一章

**设备 · INT8 · 对话**（`/cn/device-quant-chat/`）。
