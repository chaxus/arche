## 要点

本章是**课程地图**：一条从零对接前沿的路，**Llama 3.2 ~1B** 作贯穿稠密例子。你会早点打印配置，把尺寸钉死，而不是凭感觉。

例子配方错了，后面「业界怎么做」会对不上。脊柱是建表 → 猜词训练 → 积木 → 对齐 → 服务 → MoE/MLA——每块积木先在这根线上有位置，细节才好记。

实操顺序：钉死例子 → 长分词/嵌入 → next-token 训练（目标→数据→优化→规模）→ 手写 RMSNorm…stack → 加载/生成/HF 对照 → SFT/DPO → 服务 → 前沿引物。

### 整条路

| 阶段 | 人话 |
| --- | --- |
| **0 开场** | 环境；读一章；跑一条命令 |
| **1 表 + 训练 + 积木** | 分词 → 嵌入 → **预训练×4** → RMSNorm…stack → 权重 → 生成 → HF |
| **2 对齐** | SFT / DPO |
| **3 服务加速** | KV、RoPE 缩放、设备/量化/对话、连续批、分页 |
| **4 前沿** | MoE → MLA → ckpt → 分片 |

```mermaid
flowchart LR
  tok[分词] --> emb[嵌入]
  emb --> train[预训练环]
  train --> bricks[dense积木]
  bricks --> align[SFT_DPO]
  align --> serve[服务]
  serve --> tip[MoE_MLA]
```

你手写的是 **Llama 3.2 风格稠密解码器**。官方 Llama 文件是这份配方的成品一对——不是第二条课。

### 为什么是 Llama 3.2 ~1B

与更大 Llama 3、多数 Qwen 类稠密同属：**RMSNorm + RoPE + GQA + SwiGLU + 字节 BPE**。~1B 让词表、宽度、层数、头数可数，权重也能下——**前沿同构架构，可走完的规模**。

| 选择 | 含义 |
| --- | --- |
| 前沿同构积木 | 名字对齐当前开源 dense / 推理栈 |
| 可学规模 | 从零写得完 |
| 开放成品一对 | 分词 + 检查点真对照 |
| 先 dense | MoE/MLA 稍后当地图 |

不是「聊天最强」，也不是「在家训 70B」。

### DeepSeek 接到哪

| | **Llama 3.2（例子）** | **DeepSeek-V2/V3** |
| --- | --- | --- |
| 容量 | Dense FFN | **MoE** top-k |
| KV | **GQA** | **MLA** latent |
| 课内 | 第 1–3 部分 | 第 4 部分引物 |

同属 Transformer 家族，升级点不同。先做 dense 主干。

### 你会反复看到的配置

实验（[`ep01_overview.py`](https://github.com/chaxus/arche/blob/main/chapters/01-inference/ep01_overview.py)）打印 [`llama32_1b()`](https://github.com/chaxus/arche/blob/main/arche/config.py)：

| 字段 | ~1B 含义 |
| --- | --- |
| `vocab_size` | 128256 |
| `num_layers` | 16 |
| `embed_dim` | 2048 |
| `num_heads` / `num_kv_heads` | 32 / 8（GQA） |
| `rope_theta` | 500000 |

### 自己核对

- 配置自洽；积木与声称配方一致。
- 从零能走通：表可训、网可叠、权可装。
- 文档与权重可得；MoE/MLA 引物有处挂。

### 换例子会怎样

换例子（如 Qwen）⇒ 分词/嵌入/块全换。加宽加深 ⇒ 重训。本课钉 1B，更大想法用引物。

### 本仓库

```bash
python chapters/01-inference/ep01_overview.py
```

打印尺寸——还不是聊天。

## 动手

```bash
python chapters/01-inference/ep01_overview.py
```

## 下一章

**分词器**（`/cn/tokenizer/`）：从零长出第一张表。
