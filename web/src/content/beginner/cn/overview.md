## 要点

手写 **Llama 3.2 风格** dense 解码器：RMSNorm、RoPE、SwiGLU、GQA。教学目标尺寸：**1B**。

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
