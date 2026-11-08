## 要点

### 1. 为什么要预训练

先在大量文本上学会猜下一个 token，再谈跟对话。

### 2. 我们跑什么

玩具 AdamW：logits 对下一 id、交叉熵、一步更新。**本页**可拖学习率/步数，在浏览器里真跑一小段。

### 3. 版图

| 规模 | 含义 |
| --- | --- |
| 本实验 | 感受 API |
| 真预训练 | 海量数据、多卡、很久 |

### 4–6. 本课怎么做

`arche/train/pretrain.py` + 页内试跑。不是全网训练。

## 动手

```bash
python chapters/02-training/ep18_pretrain.py
```

## 下一章

**SFT · DPO**（`?c=sft-dpo`）。
