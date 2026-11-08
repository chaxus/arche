## 要点

前沿 MoE/MLA 只有在你会说它们的 **检查点名字与形状** 时才落地——带键映射地存取，包括 Mixtral 风格的 `w1/w2/w3` expert 张量。结构停在纸上没用；映射烂会留下静默的随机 expert。

把开源名映射进你的模块，用 [`save_checkpoint` / `load_checkpoint`](https://github.com/chaxus/arche/blob/main/arche/checkpoint.py)，缺键要大声失败。

### 往返

教学 ckpt：写下 state + meta，装回匹配的模块图，断言对齐。Expert 键必须齐全——部分加载的 MoE 看起来能跑，直到某个 expert 是噪声。

### 命名债

| 风险 | 后果 |
| --- | --- |
| HF vs 本地名漂移 | 缺键 / 槽位错 |
| Expert 下标差一 | 一个 expert 永久死亡 |
| 滥用 strict=False | 藏映射 bug |

### 从零

| 步骤 | 做什么 |
| --- | --- |
| 1 | 列出开源 hub 键模式（Mixtral / DeepSeek 文档） |
| 2 | 显式映射进你的模块 |
| 3 | 先在微型合成 ckpt 上往返 |
| 4 | 再碰数 GB 下载 |

### 自己核对

- 往返 `allclose`；expert 不缺。
- 意外键失败或被显式忽略。
- 钩子能接真 hub。

### 杠杆

| 杠杆 | 影响 |
| --- | --- |
| 命名一致 | 少转换债 |
| 烂映射 | 静默质量窟窿 |
| 转换器成标配 | 与稠密 HF 映射同一故事 |

### 本课

```bash
python chapters/02-training/ep20_frontier_ckpt.py
```

## 动手

```bash
python chapters/02-training/ep20_frontier_ckpt.py
```

## 下一章

**分片 MoE / MLA**（`/cn/sharded-frontier/`）。
