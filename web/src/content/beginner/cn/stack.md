## 一句话
Embed → N 个 block → 最终 RMSNorm → 线性到词表分数（常与嵌入绑权）。

## 开始之前
- block 章已完成。约 12 分钟。

## 用人话讲
`ids → Embedding → [Block × N] → RMSNorm → Linear(vocab)`。
最后线性输出下一个 token 的分数（logits）。Llama 3.2 把 `lm_head` 与嵌入绑在一起。

## 对应代码
- `arche/model.py`
- `chapters/01-inference/ep09_stack.py`

## 现在就做
```bash
python chapters/01-inference/ep09_stack.py
```

## 如果翻车了
- 词表大小不一致：用仓库提供的配置助手。

## 先别深挖
加载 Meta 权重（下一章）与采样技巧。

## 下一章
**加载权重**（`?c=weights`）。
