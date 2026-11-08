import type { Chapter } from './types';

export const CHAPTERS_CN: Chapter[] = [
  {
    id: 'overview',
    ep: '01',
    title: '总览',
    summary: '现代 dense 主干——Llama 3.2 谱系，不做 GPT-2 考古。',
    markdown: `## 为什么走这条路

**arche**（ἀρχή）意为「本原 / 第一原理」。我们手写的是**当下**的 dense 解码器：

- **RMSNorm**，而不是把 LayerNorm 当默认故事
- **RoPE**，而不是绝对位置嵌入表
- **SwiGLU**，而不是只讲 GELU FFN
- **GQA**，让解码时的 KV cache 更省

目标权重：**Llama 3.2 1B**（与 3B 同族）。能力来自开源权重；理解来自自己写前向。

\`\`\`mermaid
flowchart LR
  tok[Tokenizer] --> emb[Embedding]
  emb --> stack[Block_x_N]
  stack --> head[LM_head]
  stack --- rms[RMSNorm]
  stack --- rope[RoPE]
  stack --- gqa[GQA]
  stack --- swiglu[SwiGLU]
\`\`\`
`,
  },
  {
    id: 'tokenizer',
    ep: '02',
    title: '分词器',
    summary: '官方 Llama 3.2 BPE——文本变成 token id。',
    markdown: `## 作用

模型从不直接看字符。分词器在固定词表（Llama 3.2 为 \`128256\`）上做文本 ↔ 整数 id。

这里使用**官方**分词器（Hugging Face 门禁模型）。本仓库的学习重点是 dense 主干，而不是重造 BPE。

本地运行：

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep02_tokenizer.py
\`\`\`
`,
  },
  {
    id: 'embedding',
    ep: '03',
    title: '嵌入层',
    summary: 'token id → 稠密向量。没有绝对位置表。',
    markdown: `## 查表，不是位置

\`TokenEmbedding\` 是形状为 \`(vocab_size, embed_dim)\` 的查找表。

现代 dense 堆叠**不会**在这里再加一份可学习的绝对位置嵌入。位置稍后通过 Q/K 上的 **RoPE** 进入。

Llama 3.2 1B/3B 还会把嵌入权重与 LM head **绑在一起**（weight tying）。
`,
    latex: 'X = \\mathrm{Embed}(t_{1:L}) \\in \\mathbb{R}^{L \\times d}',
  },
  {
    id: 'rmsnorm',
    ep: '04',
    title: 'RMSNorm',
    summary: '不做均值中心化的 pre-norm。',
    markdown: `## 为什么是 RMSNorm

比 LayerNorm 更便宜（不做减均值），在现代 LLM 上足够稳，也是 Llama / Qwen dense 堆叠的默认 pre-norm。

在每个 block 里，它出现在注意力**之前**与 MLP**之前**。
`,
    latex:
      '\\mathrm{RMSNorm}(x) = \\frac{x}{\\sqrt{\\mathrm{mean}(x^2)+\\varepsilon}} \\odot \\gamma',
  },
  {
    id: 'rope',
    ep: '05',
    title: 'RoPE',
    summary: '旋转 Q/K，让注意力带上相对位置。',
    markdown: `## 旋转位置编码

RoPE 按 token 位置给 query/key 的成对维度施加旋转角。相对偏移会从点积里自然出现——不需要绝对 PE 表。

Llama 3.2 还使用很大的 \`rope_theta\`（\`500000\`），以及长上下文的 **scale factor**（留到后续章节）。
`,
    latex:
      'q_m^{\\prime} = R_{\\Theta,m} q_m,\\quad k_n^{\\prime} = R_{\\Theta,n} k_n',
  },
  {
    id: 'gqa',
    ep: '06',
    title: 'GQA 注意力',
    summary: 'KV 头少于 query 头，仍然因果。',
    markdown: `## Grouped-Query Attention

Llama 3.2 1B：**32** 个 query 头，**8** 个 KV 头 → 每个 KV 头被 4 个 query 头共享。

收益主要在**解码**阶段：KV cache 变小，质量仍接近完整多头注意力。

模块内流程：投影 Q/K/V → RoPE(Q,K) → 重复 KV → 因果 SDPA → 输出投影。
`,
  },
  {
    id: 'swiglu',
    ep: '07',
    title: 'SwiGLU',
    summary: 'gate × up，再 down——三条无偏置线性层。',
    markdown: `## 现代 FFN

\`\`\`
down(silu(gate(x)) * up(x))
\`\`\`

这取代了旧的「一层 Linear → GELU → 一层 Linear」作为 Llama / Qwen dense 模型的默认 FFN。
`,
    latex: '\\mathrm{FFN}(x)=W_d\\big(\\mathrm{SiLU}(W_g x)\\odot W_u x\\big)',
  },
  {
    id: 'block',
    ep: '08',
    title: 'Transformer 块',
    summary: '可堆叠的 pre-norm 残差三明治。',
    markdown: `## 一个 block

\`\`\`
x → RMSNorm → Attn → +x → RMSNorm → SwiGLU → +x
\`\`\`

Llama 3.2 1B 堆 **16** 次（3B：28 次）。进出租同形状——所以深度只是一个循环。
`,
  },
  {
    id: 'stack',
    ep: '09',
    title: '堆叠 + LM head',
    summary: 'embed → blocks → 最终 RMSNorm → 绑定的 lm_head。',
    markdown: `## 完整前向

\`\`\`
ids → Embedding → [Block × N] → RMSNorm → Linear(vocab)
\`\`\`

Llama 3.2 1B/3B 会把 \`lm_head.weight\` 与 \`embed_tokens.weight\` **绑在一起**。

运行：

\`\`\`bash
python chapters/01-inference/ep09_stack.py
\`\`\`
`,
  },
  {
    id: 'weights',
    ep: '10',
    title: '加载权重',
    summary: '把 Hugging Face Llama 3.2 张量映射进 arche.Llama。',
    markdown: `## 官方权重，手写计算图

\`from_pretrained\` 搭建我们的模块，下载 HF 权重，再按一小张 key 对照表拷贝张量（\`model.layers.*\` → \`layers.*\`）。

需要通过门禁并登录：

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
\`\`\`
`,
  },
  {
    id: 'generate',
    ep: '11',
    title: '文本生成',
    summary: '用 temperature 与 top-k 采样后续 token。',
    markdown: `## 解码循环

反复取最后一步 logits，可选 top-k，采样（或贪心），拼到序列上，遇 EOS 停。

\`\`\`bash
python chapters/01-inference/ep11_generate.py
\`\`\`

能力来自加载的 1B 权重；代码路径仍全部是我们自己的。
`,
  },
];
