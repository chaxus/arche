## 要点

我们要亲手写出一个小 **Llama 3.2**（约 10 亿参数）：够小能装进脑子，权重能下载，零件和生产里常见的一样。业界常把这种「层层都算、猜下一个词」的结构叫做 **dense 解码器**——名字后面混熟即可。目标：前向每一块都自己写过。

```mermaid
flowchart LR
  tok[分词] --> emb[嵌入]
  emb --> stack[堆叠]
  stack --> head[LM head]
```

## 动手

```bash
python chapters/01-inference/ep01_overview.py
```

打印层数/头数/维度等配置，不是聊天。

## 下一章

**分词器**（`?c=tokenizer`）。
