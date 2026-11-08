## 要点

从零做长上下文服务：GQA 之后 KV 仍可能爆——**MLA** 用低秩 latent 压缓存，是 DeepSeek 系对接前沿的关键一招。

### 1. 有什么作用

缓存 latent（+ 小段 RoPE 流）而非满头 K/V，需要时再展开。

### 2. 从零开始：怎么生成

| 步骤 | 做什么 |
| --- | --- |
| **1** | 定 latent 维与投影 |
| **2** | 训练期按 MLA 前向（与论文/开源实现对齐） |
| **3** | 推理存 latent 缓存 |
| **4** | 本课只给形状引物，不比特级复刻 |

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 缓存更小；前向形状自洽 |
| **软** | 质量接近满 KV；实现与某正式版对齐（产品级） |

### 4. 如何改进 · 对整模影响

更小 latent → 更省、表达风险↑。与 GQA/分页正交可组合。换 MLA = 架构级，需按该配方训。

### 5. 业界 · 开源

DeepSeek-V2/V3 推 MLA；Llama/Qwen 主线仍 GQA。引擎侧逐步增加 MLA 支持。

### 6. 本课

```bash
python chapters/01-inference/ep15_mla.py
```

## 动手

```bash
python chapters/01-inference/ep15_mla.py
```

## 下一章


**前沿检查点**（`/cn/frontier-ckpt/`）。
