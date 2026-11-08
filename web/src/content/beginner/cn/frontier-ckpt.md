## 要点

**是什么（What）：** 会说 MoE/MLA 类开源权重的 **检查点名字与形状**——带键映射地存/取，包括 Mixtral 风格的 `w1/w2/w3` expert 张量。

**为什么（Why）：** 前沿结构停在纸上没用，张量来回不过就白搭。映射烂 → expert 静默变随机。

**怎么做（How）：** 名字映射 ↔ 开源名；[`save_checkpoint` / `load_checkpoint`](https://github.com/chaxus/arche/blob/main/arche/checkpoint.py)；缺键要大声失败。

### 1. What — 往返

教学 ckpt：写下 state + meta，装回匹配的模块图，断言对齐。Expert 键必须齐全——部分加载的 MoE 看起来「能跑」，直到某个 expert 是噪声。

### 2. Why — 命名债

| 风险 | 后果 |
| --- | --- |
| HF vs 本地名漂移 | 缺键 / 槽位错 |
| Expert 下标差一 | 一个 expert 永久死亡 |
| 滥用 strict=False | 藏映射 bug |

### 3. How — 从零

| 步骤 | 做什么 |
| --- | --- |
| 1 | 列出开源 hub 键模式（Mixtral / DeepSeek 文档） |
| 2 | 显式映射进你的模块 |
| 3 | 先在微型合成 ckpt 上往返 |
| 4 | 再碰数 GB 下载 |

### 4. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 往返 `allclose`；expert 不缺 |
| **硬** | 意外键失败或被显式忽略 |
| **软** | 钩子能接真 hub |

### 5. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 命名一致 | 少转换债 |
| 烂映射 | 静默质量窟窿 |
| 转换器成标配 | 与稠密 HF 映射同一故事 |

### 6. 本课

```bash
python chapters/02-training/ep20_frontier_ckpt.py
```

## 动手

```bash
python chapters/02-training/ep20_frontier_ckpt.py
```

## 下一章

**分片 MoE / MLA**（`/cn/sharded-frontier/`）。
