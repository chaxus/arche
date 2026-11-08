## 要点

### 1. 为什么要「块」

一个可重复单元，叠 `N` 次。

### 2. 长什么样

```
x → RMSNorm → 注意力 → + 残差
  → RMSNorm → SwiGLU → + 残差
```

进出同形状。注意力可带 KV cache。

### 3. 布局选择

| 选择 | 人话 | 今天 |
| --- | --- | --- |
| Post-norm | 残差后再归一化 | 偏老 |
| **Pre-norm** | 子层前归一化 | **现代默认** |
| 注意力∥MLP 并行 | 非 Llama 布局 | 不是本课主干 |

### 4. 业界实际

Llama 3.2 1B：**16** 个这样的块。

### 5. 怎样算对

- 残差路径形状不变  
- cache 别漏层  

### 6. 本课怎么做

`arche/block.py` 的 `TransformerBlock`。

## 动手

```bash
python chapters/01-inference/ep08_block.py
```

## 下一章

**堆叠 + LM head**（`?c=stack`）。
