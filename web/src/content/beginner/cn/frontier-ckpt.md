## 一句话
存取 MoE/MLA，并把 Mixtral 风格专家 key 映射进 arche。

## 开始之前
- MoE + MLA。约 15 分钟。

## 用人话讲
前沿零件也要会存盘。我们做 state dict 往返，并把 Mixtral 的 `w1/w2/w3` 翻成我们的专家名。
分片 index 下一章继续。

## 对应代码
- `arche/checkpoint.py`、`load_mixtral.py`
- `chapters/02-training/ep20_frontier_ckpt.py`

## 现在就做
```bash
python chapters/02-training/ep20_frontier_ckpt.py
```

## 如果翻车了
- 缺专家 key：合成 dict 要含 gate 与全部专家。

## 先别深挖
下载完整 Mixtral 8x7B。

## 下一章
**HF logits 对齐**（`?c=hf-align`）。
