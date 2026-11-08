## 要点

朴素服务：每人一个生成循环——人一多 GPU 经常空转。

**连续批处理**（这里是教学调度器）：多个短解码任务放进**同一个**步进循环；谁先打满长度（或 EOS）就退役，必要时再进新任务。大家共享同一拍「时间步」。

这**不是**完整 paged attention / vLLM——只是调度形状。可试本页 demo。

## 动手

```bash
python chapters/01-inference/ep17_continuous_batching.py
```

预期：几个任务一起推进直到各自结束。

## 下一章

**预训练**（`?c=pretrain`）：一个极小的 next-token 训练循环。
