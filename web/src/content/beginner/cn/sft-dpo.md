## 要点

**是什么（What）：** **预训练之后的对齐**——SFT 教「像示范那样答」；DPO（以及 RLHF 族）在**选中 vs 拒绝**偏好上推一把（本章不做完整 RL）。

**为什么（Why）：** Next-token 预训练做出底座 LM；它本身不会自动听话或贴合产品偏好。对齐是转向，**不是**从零发明猜词能力。

**怎么做（How）：** 指令数据 → 从底座 ckpt 做 mask-prompt SFT → 偏好对 → DPO（或 RLHF）→ 回归评测。

### 1. What — SFT 损失

[`sft_loss`](https://github.com/chaxus/arche/blob/main/arche/train/sft.py)：与预训练同样错位，但 **只对回复 token** 算 CE：

```python
mask[i, start:] = True  # start ≈ prompt_len - 1
return losses[mask].mean()
```

Prompt 是上下文，不是监督目标——否则模型在「学着改写用户」。

### 2. What — DPO 草图

[`dpo_loss`](https://github.com/chaxus/arche/blob/main/arche/train/dpo.py)：策略 vs 冻结的 **reference**；相对 ref 拉高 chosen、压低 rejected（β 缩放）。

| 阶段 | 数据 | 目标 |
| --- | --- | --- |
| **SFT** | prompt → 好回复 | 回复段 CE |
| **DPO** | prompt + chosen/rejected | 相对 ref 的 logprob 边际 |
| **RLHF**（仅地图） | 奖励模型 + PPO 类 | 本章不实现 |

### 3. Why — 顺序与风险

| 选择 | 原因 |
| --- | --- |
| 放在预训练后 | 需要会猜词的底座 |
| Mask prompt | 别把 loss 烧在用户文本上 |
| DPO 里冻结 ref | 锚定，防策略无界漂 |
| 数据 > 盲目加步 | 烂示范教出烂习惯 |

过度对齐会钝化底座；常与新 special / chat 模板一起上。

### 4. How — 从零

| 步骤 | 做什么 |
| --- | --- |
| 1 | 从 base（或 instruct）checkpoint 出发 |
| 2 | 造 prompt/response ids + `prompt_lens` |
| 3 | 玩具 `train_sft_step` / `train_dpo_step` |
| 4 | 上线前评 following + 回归 |

### 5. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | mask/形状正确；DPO 中 ref 冻结 |
| **软** | 更听话 / 更安全；能力损失可控 |
| **软** | 模板与分词器仍配对 |

### 6. 本课

```bash
python chapters/02-training/ep19_sft_dpo.py
```

## 动手

```bash
python chapters/02-training/ep19_sft_dpo.py
```

## 下一章

**KV cache**（`/cn/kv-cache/`）：服务前先让解码负担得起。
