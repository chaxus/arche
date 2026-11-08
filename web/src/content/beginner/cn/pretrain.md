## 一句话
用很小的 AdamW 循环，在玩具 batch 上做 next-token 预测。

## 开始之前
- 会跑脚本；有第 1 部分堆叠概念更好。约 15 分钟。

## 用人话讲
这里的**预训练**：给随机 token id，让模型预测下一个，算交叉熵，再优化一步。
尺寸故意很小——体会 API，不是训练全网。

## 对应代码
- `arche/train/pretrain.py`
- `chapters/02-training/ep18_pretrain.py`

## 现在就做
```bash
python chapters/02-training/ep18_pretrain.py
```

## 如果翻车了
- 损失乱跳：玩具数据会抖；测试里用重复 batch 更稳。

## 先别深挖
数据管线、分布式、学习率玄学。

## 下一章
**SFT · DPO**（`?c=sft-dpo`）。
