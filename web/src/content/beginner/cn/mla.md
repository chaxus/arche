## 一句话
MLA 把 KV 压进低秩 latent 缓存（DeepSeek 风格）。

## 开始之前
- 有 KV cache 直觉。约 18 分钟。

## 用人话讲
不缓存每个 head 的完整 K/V，**MLA** 存 latent（外加一小段 RoPE 流），用时再展开。
注意力思路还在，解码更省内存。

## 对应代码
- `arche/mla.py`
- `chapters/01-inference/ep15_mla.py`

## 现在就做
```bash
python chapters/01-inference/ep15_mla.py
```

## 如果翻车了
- rank/head_dim 不一致：用章节默认构造参数。

## 先别深挖
DeepSeek 生产内核——key 映射在第 4 部分。

## 下一章
**设备 · INT8 · 对话**（`?c=device-quant-chat`）。
