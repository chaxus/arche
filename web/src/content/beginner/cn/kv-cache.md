## 要点

朴素生成每一步都对**全部**历史做注意力——prompt 的 K/V 被反复重算。

**KV 缓存：** 第一次（**prefill**）存下每层的 K/V；之后每个新 token（**decode**）只算自己的 K/V 并**追加**，再对整段缓存做注意力。

结果（精确算术下）一样，回复越长越省。看本页互动条。

## 动手

```bash
python chapters/01-inference/ep12_kv_cache.py
```

预期：演示带缓存的逐步解码。

## 下一章

**RoPE 缩放**（`?c=rope-scaling`）：长 prompt 下 RoPE 仍好用。
