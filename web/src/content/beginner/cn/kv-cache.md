## 要点

从零到可服务：生成循环正确还不够——**无 KV 缓存则长解码不可用**。这是对接业界推理的第一刀。

### 1. 有什么作用

Prefill 存每层 K/V；decode 只算新位置并追加，对缓存做注意力。

### 2. 从零开始：怎么生成

| 步骤 | 做什么 |
| --- | --- |
| **1** | 每层预分配/动态 `cat` 的 K/V 缓冲 |
| **2** | Prefill 写入；decode 追加 |
| **3** | 注意力改为读缓存 |
| **4** | 再演进分页（后章）与 GQA/MLA 减体积 |

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 与全量重算等价（浮点差内）；不串层、不丢步 |
| **软** | 算力随新 token 涨；显存可接受 |

### 4. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| GQA/MLA | 缓存更小 |
| Paged KV | 并发碎片 |
| 量化 KV | 显存↓、精度权衡 |
| 无缓存 | 仅玩具长度可行 |

### 5. 业界 · 开源

所有生产 decoder（vLLM、TGI、llama.cpp…）必有 KV 缓存；布局各异。Llama/Qwen 推理默认开启。

### 6. 本课

```bash
python chapters/01-inference/ep12_kv_cache.py
```

## 动手

```bash
python chapters/01-inference/ep12_kv_cache.py
```

## 下一章

**RoPE 缩放**（`?c=rope-scaling`）。
