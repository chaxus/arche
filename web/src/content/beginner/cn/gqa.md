## 一句话
GQA 让多个 query 头共享 KV 头，从而减小解码时的显存/内存。

## 开始之前
- 理解 RoPE。约 18 分钟。

## 用人话讲
经典多头：每个 query 头自带 K/V，准确但 **KV cache** 很大。
**GQA** 减少 KV 头，一组 query 共用。Llama 3.2 1B：32 个 query 头、8 个 KV 头。
流程：投影 → RoPE → 重复 KV → 因果注意力 → 输出投影。

## 对应代码
- `arche/attention.py`
- `chapters/01-inference/ep06_gqa.py`

## 现在就做
```bash
python chapters/01-inference/ep06_gqa.py
```

## 如果翻车了
- 必须满足 `num_heads % num_kv_heads == 0`。

## 先别深挖
FlashAttention；MLA 稍后讲。

## 下一章
**SwiGLU**（`?c=swiglu`）。
