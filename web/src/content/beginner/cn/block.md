## 一句话
一个 Transformer 块就是可堆叠的 pre-norm 残差三明治。

## 开始之前
- 注意力与 SwiGLU 已完成。约 12 分钟。

## 用人话讲
```
x → RMSNorm → Attn → +x → RMSNorm → SwiGLU → +x
```
进出同形状，所以深度只是循环。Llama 3.2 1B 堆 16 次（3B：28）。

## 对应代码
- `arche/block.py`
- `chapters/01-inference/ep08_block.py`

## 现在就做
```bash
python chapters/01-inference/ep08_block.py
```

## 如果翻车了
- 残差相加要求形状一致。

## 先别深挖
某些巨型模型的并行 Attention/MLP 布局。

## 下一章
**堆叠 + LM head**（`?c=stack`）。
