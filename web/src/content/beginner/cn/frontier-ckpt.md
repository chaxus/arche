## 要点

前沿模块（MoE、MLA）和 dense 一样要会**存取**，不能静默形状错乱。

本章往返教学用 MoE/MLA 检查点，并演示 **Mixtral 风格**专家 key（`w1/w2/w3`）如何映射到我们的模块名——读真实开放权重时少猜一点。

## 动手

```bash
python chapters/02-training/ep20_frontier_ckpt.py
```

预期：存→载成功，并看到 key 映射演示。

## 下一章

**HF 对齐**（`?c=hf-align`）：有权重时证明 dense logits 能对上 Hugging Face。
