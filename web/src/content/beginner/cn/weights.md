## 要点

本章先走**路径 A（标本）**：加载开源检查点，让手写栈立刻能说话。  
**路径 B（从零写出权重）**在下一 Part 的预训练 / SFT·DPO——真正的小数多半不是手填的，是训练写的。加载错 = 前面手写全白费。

### 1. 有什么作用

按 key 映射把张量拷进模块。随机权重图对但无意义；装上 Llama 3.2 权重后，同前向才像语言模型。

### 2. 从零开始：怎么做

| 路径 | 步骤 |
| --- | --- |
| **A 加载开源（本章）** | 许可 + 登录 → 下 HF 检查点 → **key 映射**进 arche → 配官方分词 |
| **B 自训（Part 2）** | 玩具/真预训练写出 `state_dict`；记下分词版本 |
| **校验** | 参数量、缺 key；生成后一章 **HF 对齐**比短 prompt logits |

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 形状/数量对；映射无漏；分词配对 |
| **软** | 与 HF 对齐接近；生成可读 |

### 4. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 修映射 | 消除静默错层 |
| 精度（fp16/bf16） | 速度/显存/数值 |
| 从中间检查点恢复 | 续训；步数与 RNG 要一致 |
| 错分词 | 整模语义崩 |

### 5. 业界 · 开源

HF `transformers` + safetensors 是事实标准；Llama/Qwen/DeepSeek 均如此分发。引擎（vLLM）读同一类检查点。

### 6. 本课

```bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
```

## 动手

```bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
```

## 翻车

许可、登录、镜像——同分词章。

## 下一章

**文本生成**（`/cn/generate/`）。
