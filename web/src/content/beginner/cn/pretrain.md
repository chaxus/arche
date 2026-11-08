## 要点

**预训练** = 在大量文本上学习猜下一个 token。损失通常是 logits 与真实下一 id 的交叉熵；优化器（这里是 **AdamW**）更新权重。

循环故意**极小**（玩具 batch、很少步）：感受 API、看 loss 动——不是真去训 1B 全网数据。本页可调学习率/步数，在浏览器里真跑一小段。

## 动手

```bash
python chapters/02-training/ep18_pretrain.py
```

预期：几步玩具 loss。本页 demo 用示意数字演同一思路。

## 下一章

**SFT · DPO**（`?c=sft-dpo`）：跟指令、跟偏好。
