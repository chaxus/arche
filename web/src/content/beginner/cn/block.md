## 要点

从零搭模型：零件齐了要收成**可重复的一层**，才能叠深度。

### 1. 有什么作用

**Transformer 块**（pre-norm 三明治）：

```
x → RMSNorm → 注意力 → + 残差
  → RMSNorm → SwiGLU → + 残差
```

进出同形状；注意力可接 KV cache。

### 2. 从零开始：怎么生成

| 步骤 | 做什么 |
| --- | --- |
| **1** | 按上图接线（本课布局） |
| **2** | 每块独立一套权重 |
| **3** | 定深度 N（Llama 3.2 1B = **16**） |
| **4** | 堆叠进模型，整网训 |

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 残差可加；pre-norm 位置对；有 cache 时每层隔离 |
| **软** | 加深仍可训（残差+norm）；深度由算力与目标定 |

### 4. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 加深 N | 能力↑、显存/算力↑，需重训 |
| post→pre-norm | 现代默认，利于从零训深 |
| 注意力∥MLP 并行 | 非 Llama 主线；换布局要重训 |

### 5. 业界 · 开源

Llama/Qwen dense：pre-norm 块重复 N 次。DeepSeek 等在块内换注意力/FFN 变体，**块边界**仍在。

### 6. 本课

```bash
python chapters/01-inference/ep08_block.py
```

## 动手

```bash
python chapters/01-inference/ep08_block.py
```

## 下一章

**堆叠 + LM head**（`/cn/stack/`）。
