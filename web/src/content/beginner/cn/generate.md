## 要点

生成是一个**循环**，不是算一次就完：

1. 用当前 id 跑模型 → 取**最后位置**的 logits  
2. 可选：**temperature** 软化，**top-k** 截断  
3. **贪心**（`argmax`）或按分布**采样**  
4. 把新 id 拼上；遇到 EOS 或达到长度就停  

有了 KV 缓存（下一章），第 1 步在首次 prefill 之后会便宜很多。本章先把采样循环本身搞清。

## 动手

```bash
python chapters/01-inference/ep11_generate.py
```

预期：一段短解码文本（真文案需要上一章的权重；否则看脚本说明）。

## 下一章

**KV 缓存**（`?c=kv-cache`）：别每步都把整段 prompt 重算一遍。
