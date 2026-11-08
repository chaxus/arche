## 要点

**稠密因果 LM 壳**把嵌入、`Block×N`、最终 RMSNorm 和词表头串起来：`ids → embed → Block×N → 最终 RMSNorm → Linear(V) → logits`。单层 block 还不算语言模型——预训练里的 `model(batch)` 调用的就是这整坨。按 config（`V,D,N,heads,…`）搭建，可选 **绑定** LM head ↔ embed（小 Llama 常绑），初始化后再加载或预训练。

### 前向

[`Llama`](https://github.com/chaxus/arche/blob/main/arche/model.py)：

```python
x = self.embed_tokens(input_ids)
for layer, cache in zip(self.layers, caches):
    x, new_cache = layer(x, cache=cache, start_pos=start_pos)
x = self.norm(x)
return self.lm_head(x), new_caches
```

Logits 形状 `(B, T, V)`——正是 `lm_loss` 错位后吃的东西。

### 绑定与最终 norm

| 选择 | 意义 |
| --- | --- |
| 最终 RMSNorm | 投向词表前再稳住 |
| 绑定 head↔embed | 参数更少；~1B 常见 |
| 不绑定 | 更大模型更自由 |
| 共享 config | HF 映射的唯一真相源 |

### 从零搭

| 步骤 | 做什么 |
| --- | --- |
| 1 | 填好 `Llama32Config` |
| 2 | `ModuleList` 放 `N` 个 block |
| 3 | 在 init/load 前设 `tie_word_embeddings` |
| 4 | 用预训练环训，或 `from_pretrained` |

### 自己核对

| 必须成立 | Logits `(B,T,V)`；config ↔ 权重；绑定一致 |
| 尽量做到 | 加载后或足够预训练步后质量可读 |

### 对整模的杠杆

| 杠杆 | 影响 |
| --- | --- |
| 深度 / 宽度 / V | 结构级；重训或谨慎迁移 |
| 换成 MoE | 替换 block 内 SwiGLU——骨架仍在 |
| 只拧栈 | 没有数据/优化配合意义有限 |

### 本课

```bash
python chapters/01-inference/ep09_stack.py
```

## 动手

```bash
python chapters/01-inference/ep09_stack.py
```

## 下一章

**加载权重**（`/cn/weights/`）：用 Llama 小数填满空壳。
