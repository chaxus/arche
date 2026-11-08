## 一句话
RoPE 旋转 Q/K，让注意力带上相对位置——不需要绝对位置表。

## 开始之前
- 已读 RMSNorm。约 15 分钟。

## 用人话讲
不靠「加一个位置向量」，**RoPE** 按 token 下标旋转成对维度；相对距离会从点积里出现。
Llama 3.2 还有很大的 `rope_theta`，以及后面的长上下文 **scale**（单独一章）。

## 图或公式
$$q_m' = R_{\Theta,m} q_m,\quad k_n' = R_{\Theta,n} k_n$$

## 对应代码
- `arche/rope.py`
- `chapters/01-inference/ep05_rope.py`

## 现在就做
```bash
python chapters/01-inference/ep05_rope.py
```

## 如果翻车了
- head 维度为奇数会破坏成对旋转——请用脚本默认值。

## 先别深挖
所有 RoPE 变体；长上下文缩放到 ep13。

## 下一章
**GQA 注意力**（`?c=gqa`）。
