## 一句话
教学调度器让多个短解码任务一起步进，直到各自结束。

## 开始之前
- 生成与 KV。约 15 分钟。

## 用人话讲
线上服务不会永远「一个接一个」。它们维护活跃序列池，一起步进，完成的退役。
我们的版本每步 pad——**不是** paged attention，但控制流同形。

## 对应代码
- `arche/batching.py`
- `chapters/01-inference/ep17_continuous_batching.py`

## 现在就做
```bash
python chapters/01-inference/ep17_continuous_batching.py
```

## 如果翻车了
- 空输出：保证 `max_new_tokens >= 1`。

## 先别深挖
CUDA graph 与 paged 融合（ep23）。

## 下一章
第 3 部分：**预训练**（`?c=pretrain`）。
