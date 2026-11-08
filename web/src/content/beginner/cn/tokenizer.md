## 一句话
文字要先变成整数 **token id**，模型才能用向量计算。

## 开始之前
- 已读总览。
- 约 15 分钟。
- 官方 Llama 3.2 分词器在 Hugging Face 上是**门禁**的——跑本章实验时再登录即可。

## 用人话讲
模型不像人那样直接读字母。**分词器**把文本切成 token，并映射到固定词表里的编号（Llama 3.2：128256）。

我们用**官方**分词器。本课重点是神经网络主干，不重造 BPE。

## 对应代码
- `arche/tokenizer.py`
- 配套：`chapters/01-inference/ep02_tokenizer.py`

## 现在就做
```bash
huggingface-cli login
python chapters/01-inference/ep02_tokenizer.py
```
会打印示例字符串的 token id。若暂时不登录，可先继续读，稍后再回来跑。

## 如果翻车了
- 401/403：先在网页接受模型协议，再 login。
- 下载慢：可设 `export HF_ENDPOINT=https://hf-mirror.com`。

## 先别深挖
自己训练分词器，或纠结所有 Unicode 边角。

## 下一章
**嵌入层**（`?c=embedding`）。
