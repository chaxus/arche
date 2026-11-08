## 要点

GQA 已经减少了 KV 头。**MLA**（Multi-head Latent Attention，DeepSeek 思路）为了解码显存走得更远：缓存**低秩 latent**（外加一小段 RoPE 流），而不是每头完整 K/V，注意力需要时再展开。

教学目标：体会 latent KV **为什么**省内存——不是比特级复刻 DeepSeek 权重。

## 动手

```bash
python chapters/01-inference/ep15_mla.py
```

预期：紧凑前向，展示 latent 缓存形状。

## 下一章

**设备 · INT8 · 对话**（`?c=device-quant-chat`）：跑在哪、权重怎么变小、对话怎么排版。
