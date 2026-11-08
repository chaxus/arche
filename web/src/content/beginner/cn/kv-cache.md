## 一句话
先对 prompt 做一次 prefill，再解码时复用已存的 K/V。

## 开始之前
- 生成章已完成。约 15 分钟。

## 用人话讲
没有 cache 时，每来一个新 token 都要对整段前缀重算注意力。
有了 **KV cache**：prompt 的 K/V 算一次；之后每个新 token 只追加自己的 K/V。

## 对应代码
- `arche/cache.py`；`generate(..., use_cache=True)`
- `chapters/01-inference/ep12_kv_cache.py`

## 现在就做
```bash
python chapters/01-inference/ep12_kv_cache.py
```

## 如果翻车了
- `start_pos` 乱了：恢复章节脚本。

## 先别深挖
Paged KV（ep23）与生产调度器。

## 下一章
**RoPE 缩放**（`?c=rope-scaling`）。
