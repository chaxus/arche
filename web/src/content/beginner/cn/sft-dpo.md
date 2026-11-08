## 要点

Part 2 收尾：预训练后仍要**对齐**——SFT 教会跟随；DPO/RLHF 一类推向偏好。之后进入 Part 3，把生成做成可服务的解码。

### 1. 有什么作用

- **SFT**：提示→回答；掩提示；loss 只在回复  
- **DPO**：相对 reference，抬 chosen、压 rejected  

### 2. 从零开始：怎么做

| 步骤 | 做什么 |
| --- | --- |
| **1** | 准备指令数据 / 偏好对 |
| **2** | 从预训练检查点续训 SFT |
| **3** | 再 DPO/RLHF（配方任选） |
| **4** | 评估拒答、跟随、回归 |

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | SFT 掩码对；DPO 样本与 reference 形对 |
| **软** | 更听指令、更安全；少伤预训练能力 |

### 4. 如何改进 · 对整模影响

数据质量 > 盲目步数。过强对齐可能「变傻」。可与扩特殊符、改模板一起做。

### 5. 业界 · 开源

Llama/Qwen 均有 Instruct/Chat 变体；Zephyr 等推 DPO；OpenAI/Anthropic 用 RLHF 变体（细节不全开）。TRL 等库是开源对齐常用工具。

### 6. 本课

```bash
python chapters/02-training/ep19_sft_dpo.py
```

## 动手

```bash
python chapters/02-training/ep19_sft_dpo.py
```

## 下一章


**KV 缓存**（`?c=kv-cache`）：生成要能服务，先把解码成本打下来。
