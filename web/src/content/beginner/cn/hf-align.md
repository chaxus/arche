## 要点

### 1. 为什么

信任 dense 主干接线，但要验证。

### 2. 做什么

同一短 prompt 跑 arche 与 HF Llama 3.2；比 argmax 与 logits 最大绝对差。无权重 → 干净退出。

### 3–6. 本课怎么做

本章脚本 + CI 合成映射测试。

## 动手

```bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
```

## 翻车

无权重/未登录 → 跳过或提示，不等于逻辑坏了。

## 下一章

**分片 MoE/MLA**（`?c=sharded-frontier`）。
