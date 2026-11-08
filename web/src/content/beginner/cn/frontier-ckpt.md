## 要点

从零若走上 MoE/MLA，检查点**名字与形状**必须接得住开放权重——否则前沿结构只停留在纸上。

### 1. 有什么作用

教学往返检查点；Mixtral 风 `w1/w2/w3` 映射到本课模块名。

### 2. 从零开始：怎么做

设计模块名 ↔ 开源名映射；保存/加载；缺 key 失败要响。

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 往返对齐；专家权重全覆盖 |
| **软** | 能接真开放权重仓库 |

### 4. 如何改进 · 对整模影响

映射错 → 静默随机专家。统一命名减少工程债。

### 5. 业界 · 开源

HF 分片 + 文档化 key；Mixtral/DeepSeek 各自命名。转换脚本是生态标配。

### 6. 本课

```bash
python chapters/02-training/ep20_frontier_ckpt.py
```

## 动手

```bash
python chapters/02-training/ep20_frontier_ckpt.py
```

## 下一章


**分片 MoE / MLA**（`?c=sharded-frontier`）。
