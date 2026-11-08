## 要点

有门禁权重时对比 arche 与 HF logits（argmax + max-abs）。离线则干净退出；CI 仍测合成映射。

## 动手

```bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
```

## 下一章

**分片 MoE/MLA**（`?c=sharded-frontier`）。
