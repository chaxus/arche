## 要点

Part 1 收尾验收：**与权威 HF 实现对短 prompt logits**——证明映射与公式没大错。过关后再进入 Part 2，体会权重如何从零训练出来。

### 1. 有什么作用

同 prompt 跑 arche vs HF Llama 3.2；比 argmax 与 max abs diff。

### 2. 从零开始：怎么做

固定种子与精度；加载同一权重；比短序列；无权重则干净跳过。

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 有权时足够接近；无权时明确跳过 |
| **软** | 覆盖更长序列/更多算子 |

### 4. 如何改进 · 对整模影响

对齐失败 → 回查映射/RoPE/norm。通过对齐再谈优化与服务。

### 5. 业界 · 开源

开源复现常把「对上 HF/官方」当 CI 门禁；vLLM 等也有数值回归测。

### 6. 本课

```bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
```

## 动手

```bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
```

## 翻车

无权/未登录 → 跳过 ≠ 逻辑坏。

## 下一章


**预训练**（`?c=pretrain`）：从零写出权重（玩具 API）。
