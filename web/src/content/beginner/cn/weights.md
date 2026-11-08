## 要点

### 1. 为什么要加载权重

图对了但权重随机 → 输出无意义。

### 2. 做什么

`from_pretrained` 搭好模块，再按 **key 映射**拷入 HF Llama 3.2 张量。

### 3. 常见坑

| 问题 | 含义 |
| --- | --- |
| 分词器不对 | id 对不上嵌入行（见分词器章） |
| 没过门禁 | 下不了 |
| key 映射错 | 静默装错层 |

### 4. 业界实际

官方 HF Llama 3.2 检查点（许可 + 登录）。

### 5. 怎样算对

- 参数量看起来对  
- 后面 HF 对齐章：短 prompt 上 logits 接近  

### 6. 本课怎么做

`arche/load_weights.py` + 本章脚本。

## 动手

```bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
```

## 翻车

与分词器相同：许可、登录、镜像。

## 下一章

**文本生成**（`?c=generate`）。
