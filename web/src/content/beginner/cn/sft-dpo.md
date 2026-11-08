## 要点

### 1. 为什么

预训练后仍要会跟对话、更偏好好回答。

### 2. 做什么

- **SFT：** 提示→回答；**掩掉提示**；损失只打在回复  
- **DPO：** 相对冻结 reference，偏好 chosen（教学一步）

### 3. 版图

RLHF / RLAIF / DPO / … 配方很多；这里尝两个 API。

### 4–6. 本课怎么做

`arche/train/sft.py`（及 DPO）+ 本章脚本。

## 动手

```bash
python chapters/02-training/ep19_sft_dpo.py
```

## 下一章

**前沿检查点**（`?c=frontier-ckpt`）。
