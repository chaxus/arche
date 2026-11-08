## 要点

**是什么（What）：** 预训练一步的**优化器一侧**——AdamW、学习率（与日程）、梯度裁剪、梯度累积——让 CE 目标能稳定改写权重。

**为什么（Why）：** 损失写对了、优化器写烂 → NaN、发散或 loss 卡死。离开玩具规模后，**怎么迈步**决定跑得完跑不完。

**怎么做（How）：** `zero_grad` → 前向 → loss → `backward` →（可选 clip）→ `optimizer.step` →（可选 lr schedule）。`B` 必须很小时，用微 batch **累积梯度**。

### 1. What — 旋钮

| 旋钮 | 作用 |
| --- | --- |
| **AdamW** | 自适应动量 + **解耦 weight decay**（Llama/nanoGPT 类菜谱默认） |
| **lr** | 步长；太大易炸，太小爬行 |
| **schedule** | 先 warmup 再衰减（cosine / WSD / …），避免早期撕碎随机嵌入 |
| **grad clip** | 限制全局梯度范数（如 1.0）抗尖峰 |
| **grad accum** | `K` 个微 batch 累加梯度 ≈ 更大有效 batch |

本仓库玩具环（[`train_pretrain_steps`](https://github.com/chaxus/arche/blob/main/arche/train/pretrain.py)）：

```python
opt = torch.optim.AdamW(model.parameters(), lr=lr)
for batch in batches:
    opt.zero_grad(set_to_none=True)
    logits, _ = model(batch)
    loss = lm_loss(logits, batch)
    loss.backward()
    opt.step()
```

尚无 schedule/clip——刻意如此。生产在同一骨架外包一层。

### 2. Why — 各自的理由

| 选择 | 原因 |
| --- | --- |
| AdamW 而非纯 SGD | token 梯度偏稀疏 + 参数极多；自适应步长更稳 |
| Weight decay | 轻度正则；AdamW 形式对齐业界 LLM 实践 |
| Warmup | 早期 CE 很大（`~log V`）；硬 lr 会打烂初始化 |
| Clip | 一个坏 batch 不该炸掉浮点 |
| 累积 | 显存不够时仍保持有效 batch |

### 3. How — 累积草图

```python
opt.zero_grad(set_to_none=True)
for micro in micro_batches:
    loss = lm_loss(*model_and_shift(micro)) / K
    loss.backward()          # 梯度累加
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()
# scheduler.step()  # 若有
```

```mermaid
flowchart LR
  mb[微batch] --> bwd[反传_x_K]
  bwd --> clip[梯度裁剪]
  clip --> step[AdamW一步]
  step --> sch[lr日程]
```

### 4. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | loss/梯度无 NaN/Inf（clip 后） |
| **硬** | 每个**优化器**步才 `zero_grad`（累积时别清早了） |
| **软** | 合理窗口内 loss 下行 |
| **软** | 放大时 lr 日程对齐已知菜谱 |

### 5. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| lr / warmup | 深层 + 嵌入是否训得动 |
| 有效 batch（B×accum×卡数） | 更新噪声 vs 稳定 |
| clip 阈值 | 扛尖峰 vs 更新过弱 |
| decay 设错 | 过度收缩或正则不足 |

### 6. 本课

```bash
python chapters/02-training/ep18_pretrain.py
```

页内把 lr 拧高到 loss 发疯——就是缩小版的优化课。

## 动手

```bash
python chapters/02-training/ep18_pretrain.py
```

## 下一章

**预训练规模**（`/cn/pretrain-scale/`）：前沿菜谱如何拉长同一条环。
