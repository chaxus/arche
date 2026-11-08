## 一句话
有权重时，验证手写图与 Hugging Face Llama 3.2 在短 prompt 上 logits 对齐。

## 开始之前
- 权重与生成章。约 20 分钟。
- 需要门禁权重/登录。

## 用人话讲
同一权重、同一数学：argmax 应一致，max-abs 漂移应很小。
无权重时章节干净退出；CI 仍测合成映射 bit-exact。

## 对应代码
- `arche/align_hf.py`
- `chapters/01-inference/ep21_hf_align.py`

## 现在就做
```bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
```

## 如果翻车了
- 不可达：先完成 HF 设置；合成测试仍验证映射器。

## 先别深挖
所有 kernel/dtype 的逐比特一致。

## 下一章
**分片 MoE / MLA**（`?c=sharded-frontier`）。
