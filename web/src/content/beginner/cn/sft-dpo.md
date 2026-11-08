## 一句话
SFT 只监督回答段；DPO 让模型更偏好「更好」的回复。

## 开始之前
- 预训练章。约 18 分钟。

## 用人话讲
**SFT** 掩掉 prompt，损失盯着回复。
**DPO** 在冻结 reference 下比较 chosen / rejected（Rafailov 等）——一步教学更新即可。

## 对应代码
- `arche/train/sft.py`、`dpo.py`
- `chapters/02-training/ep19_sft_dpo.py`

## 现在就做
```bash
python chapters/02-training/ep19_sft_dpo.py
```

## 如果翻车了
- prompt 长度超过序列：保持 demo 张量。

## 先别深挖
完整 RLHF 与偏好数据工程。

## 下一章
**前沿检查点**（`?c=frontier-ckpt`）。
