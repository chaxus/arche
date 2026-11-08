## 要点

信任但要验证：`from_pretrained` 之后，用**同一句短 prompt**分别跑 arche 与 Hugging Face Llama 3.2，比 **argmax** 和 **logits 最大绝对差**。

若没有门禁权重，本章干净退出（CI 仍有合成映射测试）。这是「dense 主干接线对不对」的关卡。

## 动手

```bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
```

有权重时预期：一份很小的差异报告。

## 翻车

无权重 / 未登录 → 脚本干净跳过或给出 Fix；不等于 arche 逻辑坏了。

## 下一章

**分片 MoE/MLA**（`?c=sharded-frontier`）：大检查点在磁盘上怎么拆。
