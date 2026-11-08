## 要点

**RoPE 缩放**调整旋转频率（Llama 3 风格波长平滑），让超出原训练窗的位置仍可用。短上下文预训练再加上天真拉长下标，旋转会缠坏，远端质量塌。长上下文产品需要 **菜谱扩展**，不是「把 max_seq 改大就行」。对齐 checkpoint 的 `theta` 和 scale 因子，可选长数据继续训，短、长上下文都要评。

### 代码钩子

[`llama3_inv_freq`](https://github.com/chaxus/arche/blob/main/arche/rope.py) / `build_rope_cache(..., scale_factor=..., low_freq_factor=..., high_freq_factor=..., original_max_position=...)`。

`scale_factor == 1.0` 时是普通 RoPE；否则按 HF Llama-3 方式缩放/平滑波段——**与宣称 128k 级配置同一套数学**。

### 变长不是免费的

| 错觉 | 现实 |
| --- | --- |
| 只拉长下标 | 高频带会乱 |
| 更大窗口免费 | KV 显存随长度涨 |
| 缩放 = 长程知识 | 硬长程任务仍要数据 / 续训 |

### 套上菜谱

| 步骤 | 做什么 |
| --- | --- |
| 1 | 从 ckpt config 读官方 rope scaling 字段 |
| 2 | 用这些因子建 cos/sin |
| 3 | 保留短上下文健全性检查 |
| 4 | 宣称 128k 前做 long-bench / needle |

### 自己核对

| 必须成立 | 因子与 checkpoint 一致 |
| 尽量做到 | 近处仍稳；远处可用 |
| 尽量做到 | long-bench 达到产品条 |

### 对整模的影响

| 杠杆 | 影响 |
| --- | --- |
| 缩放设错 | 长（甚至短）都坏 |
| YaRN / NTK / … | 其他开源菜谱——别套在 Llama 权重下混用 |
| 更长窗口 | 永远要付 KV 账 |

### 本课

```bash
python chapters/01-inference/ep13_rope_scaling.py
```

## 动手

```bash
python chapters/01-inference/ep13_rope_scaling.py
```

## 下一章

**设备 · INT8 · 对话**（`/cn/device-quant-chat/`）。
