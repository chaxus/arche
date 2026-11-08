## 要点

从零的核心阶段：**预训练**写出嵌入与几乎全部主干权重。没有它，只有随机空壳。

### 1. 有什么作用

海量文本上 next-token：logits→CE→反向更新（含嵌入）。

### 2. 从零开始：怎么生成

| 步骤 | 做什么 |
| --- | --- |
| **1** | 分词冻结、模型空壳、数据管线 |
| **2** | AdamW 等优化器；拼 batch；shift 预测下一 id |
| **3** | 大规模多卡长时间跑；存检查点 |
| **4** | 本课：玩具步数 + 页内 playground 感受 API |

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 目标确是下一 token；梯度到该到的参数 |
| **软** | loss 下降；下游/探针指标；稳定无炸 |

### 4. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 数据质量与配比 | 能力上限主因 |
| 规模（参数/token） | 缩放律 |
| 稳定性（norm、lr、裁剪） | 能否训完 |
| 训练不足 | 嵌入与深层都欠拟合 |

### 5. 业界 · 开源

Llama/Qwen/DeepSeek 均公开「预训练→再对齐」叙事；数据细节多不全开。Megatron/DeepSpeed/torchtitan 等是从零训练栈。

### 6. 本课

```bash
python chapters/02-training/ep18_pretrain.py
```

页内可拖 lr/步数。

## 动手

```bash
python chapters/02-training/ep18_pretrain.py
```

## 下一章

**SFT · DPO**（`?c=sft-dpo`）。
