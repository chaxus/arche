## 要点

### 1. 有什么作用

超大模型权重常是 `index.json` + 许多分片文件。本章用**合成** index + 分片（Mixtral 形 MoE / DeepSeek 形 MLA）练习打包/拆包，装回后 `allclose`——不必真下几十 GB。

### 2. 为什么大模型过程需要它

单文件装不下或不好分发。不会读分片 = 前沿开放权重用不成。教学用合成数据，专练**索引与拼接**，不练下载耐力。

### 3. 怎么来的 · 怎么做

生成合成分片 → 按 index 装回 → 数值对齐检查。

### 4. 有什么标准 · 怎样算好

| 类型 | 标准 | 人话 |
| --- | --- | --- |
| **硬** | index 与分片一致 | 缺片/错 key 要失败得明白 |
| **硬** | 装回 `allclose` | 拼接正确 |
| **软** | 接真仓库格式 | 产品工程 |

### 5. 本课怎么做

本章脚本。

## 动手

```bash
python chapters/02-training/ep22_sharded_frontier.py
```

## 下一章

**Paged KV · CUDA Graph**（`?c=paged-cuda-graph`）。
