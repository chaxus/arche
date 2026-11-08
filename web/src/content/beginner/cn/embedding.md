## 一句话
每个 token id 变成一串学来的数字——大查找表里的一行。

## 开始之前
- 理解「先有 token id」。
- 约 12 分钟。

## 用人话讲
把它想成词典：token id → 长度为 `embed_dim` 的向量。这张表就是 **TokenEmbedding**。

现代 dense 模型通常**不会**在这里再加一张绝对位置表。位置稍后由注意力里的 **RoPE** 提供。

Llama 3.2 还会把嵌入与最终 LM head **绑权**（共用同一份数字）。

## 图或公式
$$X = \mathrm{Embed}(t_{1:L}) \in \mathbb{R}^{L \times d}$$

## 对应代码
- `arche/embedding.py`
- 配套：`chapters/01-inference/ep03_embedding.py`

## 现在就做
```bash
python chapters/01-inference/ep03_embedding.py
```
关注形状是否像 `(batch, seq, embed_dim)`。

## 如果翻车了
- 形状对不上：多半改乱了示例尺寸——用 git 恢复脚本。

## 先别深挖
嵌入如何训练、segment id 等花样。

## 下一章
**RMSNorm**（`?c=rmsnorm`）。
