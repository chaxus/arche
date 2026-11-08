## 要点

**是什么（What）：** 两件服务加速——**paged KV**（block table + 共享页池）抗显存碎片；**CUDA Graph** 捕获固定形状的 decode 步，砍启动税。

**为什么（Why）：** 连续批没有 paging，长短不齐会浪费显存；每次小小 decode 的 eager 启动也贵。合在一起指向 vLLM 级引擎——本章是教学替身，不是可替换插件。

**怎么做（How）：** 逻辑 KV → 经 block table 申请/释放页 → `gather_kv` 供注意力；形状稳定时可选用 graph 捕获。

### 1. What — paged KV

[`PagedKVPool` / `BlockTable` / `gather_kv`](https://github.com/chaxus/arche/blob/main/arche/paged_kv.py)：

```python
append_kv(pool, table, k, v)
cache = gather_kv(pool, table)
```

物理页共享；每条序列持有一串 page id。

### 2. What — CUDA Graph（可选）

[`capture_module_step` / `generate_with_optional_graph`](https://github.com/chaxus/arche/blob/main/arche/cuda_graphs.py)：有 CUDA 时捕获一步并回放。需要**稳定形状**——引擎才在乎批策略。

### 3. Why — 朝引擎迈一步

| 零件 | 打中的痛 |
| --- | --- |
| Paging | 碎片 + 多用户打包 |
| Graph | 每步 CPU 启动开销 |
| 都不改权重 | 只抬高服务天花板 |

### 4. How — 从零

| 步骤 | 做什么 |
| --- | --- |
| 1 | 先有连续批心智模型 |
| 2 | 页池 + 每序列 block table |
| 3 | 注意力前 gather（教学路径） |
| 4 | 形状冻住后可选捕获 decode |

### 5. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | gather 对齐逻辑 cache；表不越界 |
| **软** | 更少碎片 / 启动税；更高吞吐 |
| **软** | 无 CUDA 时干净跳过 |

### 6. 本课

```bash
python chapters/01-inference/ep23_paged_cuda_graph.py
```

第 3 部分结束；第 4 部分进入 MoE/MLA 前沿结构地图。

## 动手

```bash
python chapters/01-inference/ep23_paged_cuda_graph.py
```

## 下一章

**MoE**（`/cn/moe/`）：用稀疏专家换前沿容量。
