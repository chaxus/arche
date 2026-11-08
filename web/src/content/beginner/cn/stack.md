## 要点

### 1. 为什么要堆叠

一块还不是语言模型——多层 + 词表头才是。

### 2. 整条路

`ids → 嵌入 → Block×N → 最终 RMSNorm → Linear(词表)`

最后分数 = 下一 token 的 **logits**。常 **绑** LM head ↔ 嵌入。

### 3. 零件

| 零件 | 作用 |
| --- | --- |
| 嵌入 | id → 向量 |
| 块 | 上下文混合 |
| 最终 norm | 进 head 前稳住 |
| LM head | 向量 → 词表分 |

### 4. 业界实际

Llama 风格 dense 端到端（下一章前仍是随机权重）。

### 5. 怎样算对

- logits `(B, T, vocab)`  
- 绑权与声称的配置一致  

### 6. 本课怎么做

`arche/model.py` 的 `Llama`。

## 动手

```bash
python chapters/01-inference/ep09_stack.py
```

## 下一章

**加载权重**（`?c=weights`）。
