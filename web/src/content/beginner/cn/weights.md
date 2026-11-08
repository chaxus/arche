## 要点

到目前图是对的，但权重是随机的——输出没有意义。

**`from_pretrained`** 先搭好我们的 `Llama` 模块，再按 **key 映射**（HF 名 → arche 名）把官方 Hugging Face Llama 3.2 检查点拷进来。之后前向应与发布模型一致（差在数值误差）。

需要接受模型许可并 `huggingface-cli login`（和分词器同一道门）。

## 动手

```bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
```

预期：确认权重已装入（参数量 / 简短自检）。镜像：`export HF_ENDPOINT=https://hf-mirror.com`。

## 翻车

与分词器相同：许可、登录、网络。

## 下一章

**文本生成**（`?c=generate`）：把 logits 一个个变成字。
