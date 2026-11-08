## 要点

### 1. 为什么要循环

一次前向只给出「下一个」的分数，不是整段回复。

### 2. 生成做什么

1. 取**最后位置** logits  
2. 可选 temperature / top-k  
3. 贪心或采样  
4. 拼上；EOS 或长度则停  

KV 缓存（下一章）让第 1 步在 prefill 后变便宜。

### 3. 旋钮

| 旋钮 | 效果 |
| --- | --- |
| temperature → 0 | 更贪心/更尖 |
| temperature 升高 | 更随机 |
| top-k | 只保留前 k 个 |

### 4. 业界实际

所有聊天栈同一类循环；采样细节各家不同。

### 5. 怎样算对

- 能干净停下  
- 有 EOS 时尊重它  
- 有真权重时文字连贯（不是本章唯一目标）  

### 6. 本课怎么做

`arche/generate.py` 的 `generate`。

## 动手

```bash
python chapters/01-inference/ep11_generate.py
```

## 下一章

**KV 缓存**（`?c=kv-cache`）。
