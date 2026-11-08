## 要点

文字经官方 Llama 3.2 BPE 变成 **token id**（词表 `128256`）。不重造 BPE；跑实验需要 HF 门禁登录。

## 动手

```bash
huggingface-cli login
python chapters/01-inference/ep02_tokenizer.py
```

## 下一章

**嵌入层**（`?c=embedding`）。
