import type { Chapter } from './types';
import { PART0_CN } from './part0-cn';

export const CHAPTERS_CN: Chapter[] = [
  ...PART0_CN,
  {
    id: 'overview',
    ep: '01',
    title: '总览',
    summary: '现代 dense 主干——Llama 3.2 谱系，不做 GPT-2 考古。',
    script: 'chapters/01-inference/ep01_overview.py',
    partId: 'trunk',
    estimatedMinutes: 12,
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
    script: 'chapters/01-inference/ep02_tokenizer.py',
    partId: 'trunk',
    estimatedMinutes: 15,
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
    script: 'chapters/01-inference/ep03_embedding.py',
    partId: 'trunk',
    estimatedMinutes: 12,
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
    script: 'chapters/01-inference/ep04_rmsnorm.py',
    partId: 'trunk',
    estimatedMinutes: 12,
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
    script: 'chapters/01-inference/ep05_rope.py',
    partId: 'trunk',
    estimatedMinutes: 15,
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
    script: 'chapters/01-inference/ep06_gqa.py',
    partId: 'trunk',
    estimatedMinutes: 18,
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
    script: 'chapters/01-inference/ep07_swiglu.py',
    partId: 'trunk',
    estimatedMinutes: 12,
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
    script: 'chapters/01-inference/ep08_block.py',
    partId: 'trunk',
    estimatedMinutes: 12,
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
    script: 'chapters/01-inference/ep09_stack.py',
    partId: 'trunk',
    estimatedMinutes: 12,
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
    script: 'chapters/01-inference/ep10_load_weights.py',
    partId: 'trunk',
    estimatedMinutes: 20,
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
    script: 'chapters/01-inference/ep11_generate.py',
    partId: 'trunk',
    estimatedMinutes: 15,
    markdown: `## 解码循环

反复取最后一步 logits，可选 top-k，采样（或贪心），拼到序列上，遇 EOS 停。

\`\`\`bash
python chapters/01-inference/ep11_generate.py
\`\`\`

能力来自加载的 1B 权重；代码路径仍全部是我们自己的。
`,
  },
  {
    id: 'kv-cache',
    ep: '12',
    title: 'KV 缓存',
    summary: '先 prefill 一次，再用缓存的 K/V 逐步解码。',
    script: 'chapters/01-inference/ep12_kv_cache.py',
    partId: 'infer',
    estimatedMinutes: 15,
    markdown: `## 为什么要缓存

没有 cache 时，每生成一个新 token 都要对整段前缀重算注意力。

有了 cache：只投影新 token，把 K/V 追加进去，再对已存的过去做注意力。

\`\`\`bash
python chapters/01-inference/ep12_kv_cache.py
\`\`\`

\`generate(..., use_cache=True)\` 是默认行为。
`,
  },
  {
    id: 'rope-scaling',
    ep: '13',
    title: 'RoPE 缩放',
    summary: 'Llama 3.2 的波长平滑，面向长上下文。',
    script: 'chapters/01-inference/ep13_rope_scaling.py',
    partId: 'infer',
    estimatedMinutes: 15,
    markdown: `## 超出 8k 窗口

Llama 3.2 保留 \`rope_theta=500000\`，再用 **factor / low / high** 调度把长波长拉向 128k 上下文。

\`\`\`bash
python chapters/01-inference/ep13_rope_scaling.py
\`\`\`

短波长不动；长波长除以 \`scale_factor\`；中间带平滑——与 Hugging Face 的 llama3 RoPE 同一思路。
`,
  },
  {
    id: 'moe',
    ep: '14',
    title: 'MoE',
    summary: 'Top-k 路由的 SwiGLU 专家——稀疏容量。',
    script: 'chapters/01-inference/ep14_moe.py',
    partId: 'infer',
    estimatedMinutes: 15,
    markdown: `## Mixture of Experts

线性路由器给每个 token 打分；只跑 \`top_k\` 个专家。参数量可以很大，但每 token 算力仍接近稠密 MLP。

\`\`\`bash
python chapters/01-inference/ep14_moe.py
\`\`\`

这就是 Mixtral、DeepSeek 类前沿堆叠背后的稀疏 FFN 思路。
`,
  },
  {
    id: 'mla',
    ep: '15',
    title: 'MLA',
    summary: '把 KV 压进 latent 缓存（DeepSeek 风格）。',
    script: 'chapters/01-inference/ep15_mla.py',
    partId: 'infer',
    estimatedMinutes: 18,
    markdown: `## Multi-head Latent Attention

不缓存每个 head 的 K/V，MLA 存低秩 latent（外加一小段 RoPE 流），用时再展开。

\`\`\`bash
python chapters/01-inference/ep15_mla.py
\`\`\`

这就是 DeepSeek 推广的解码显存优势——注意力数学还在，缓存更省。
`,
  },
  {
    id: 'device-quant-chat',
    ep: '16',
    title: '设备 · INT8 · 对话模板',
    summary: '选择 CUDA/MPS/CPU、权重量化 INT8，以及 chat template。',
    script: 'chapters/01-inference/ep16_device_quant_chat.py',
    partId: 'infer',
    estimatedMinutes: 15,
    demo: 'chat',
    markdown: `## 推理工程入门

\`pick_device\` 优先 CUDA → MPS → CPU。\`quantize_llama_int8\` 把 Linear 换成按通道 int8 + scale（教学用，不是 bitsandbytes）。

对话走 \`apply_chat_template\`——有 HF 模板就用官方的，否则回退到 role/content 纯文本。

\`\`\`bash
python chapters/01-inference/ep16_device_quant_chat.py
\`\`\`

与 HF logits 对齐见 \`tests/test_align_hf.py\`（无门禁权重时跳过；\`ARCH_HF_ALIGN=1\` 可强制跑）。
`,
  },
  {
    id: 'continuous-batching',
    ep: '17',
    title: '连续批处理',
    summary: '在一个步进循环里调度多段短解码。',
    script: 'chapters/01-inference/ep17_continuous_batching.py',
    partId: 'infer',
    estimatedMinutes: 15,
    demo: 'batch',
    markdown: `## 教学版调度器

每步打包仍活跃的请求，采样下一 token，打到 EOS 或 \`max_new_tokens\` 就退役。

这**不是** paged attention 或生产引擎——只讲清 continuous batching 的控制流形状。

\`\`\`bash
python chapters/01-inference/ep17_continuous_batching.py
\`\`\`
`,
  },
  {
    id: 'pretrain',
    ep: '18',
    title: '预训练',
    summary: '玩具 batch 上的 next-token AdamW 小循环。',
    script: 'chapters/02-training/ep18_pretrain.py',
    partId: 'train',
    estimatedMinutes: 15,
    demo: 'train',
    markdown: `## 从前向到学习

因果 LM 损失：用位置 \`0..T-1\` 预测 \`t+1\`。随机 id 上跑几步 AdamW，就能摸到训练 API。

\`\`\`bash
python chapters/02-training/ep18_pretrain.py
\`\`\`
`,
  },
  {
    id: 'sft-dpo',
    ep: '19',
    title: 'SFT · DPO',
    summary: 'SFT 掩掉 prompt；DPO 跑一步偏好更新。',
    script: 'chapters/02-training/ep19_sft_dpo.py',
    partId: 'train',
    estimatedMinutes: 18,
    demo: 'train',
    markdown: `## 预训练之后的对齐

**SFT** 只监督回复段。**DPO** 用冻结的 reference，比较 chosen / rejected 的 log-prob 比（Rafailov 等）。

\`\`\`bash
python chapters/02-training/ep19_sft_dpo.py
\`\`\`
`,
  },
  {
    id: 'frontier-ckpt',
    ep: '20',
    title: '前沿检查点',
    summary: 'MoE/MLA 往返加载 + Mixtral 风格专家 key 映射。',
    script: 'chapters/02-training/ep20_frontier_ckpt.py',
    partId: 'frontier',
    estimatedMinutes: 15,
    markdown: `## 把引物接上权重

用 \`save_checkpoint\` / \`load_checkpoint\` 存取 MoE 与 MLA。把 Mixtral 的 \`w1/w2/w3\` expert key 映射进 \`MoESwiGLU\`。

分片版 Mixtral / DeepSeek 布局见 ep22。

\`\`\`bash
python chapters/02-training/ep20_frontier_ckpt.py
\`\`\`
`,
  },
  {
    id: 'hf-align',
    ep: '21',
    title: 'HF logits 对齐',
    summary: '在短 prompt 上对比 arche 与 Hugging Face Llama 3.2。',
    script: 'chapters/01-inference/ep21_hf_align.py',
    partId: 'frontier',
    estimatedMinutes: 20,
    markdown: `## 验证手写计算图

\`compare_logits_to_hf\` 把门禁 1B 权重载入两边，检查 argmax 与 max-abs 漂移。

无登录 / 无缓存时章节会干净退出；CI 仍跑合成权重的 bit-exact 映射测试。

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
# 或：ARCH_HF_ALIGN=1 pytest tests/test_align_hf.py -q
\`\`\`
`,
  },
  {
    id: 'sharded-frontier',
    ep: '22',
    title: '分片 MoE / MLA',
    summary: 'index.json + 分片文件装进 Mixtral MoE 与 DeepSeek MLA。',
    script: 'chapters/02-training/ep22_sharded_frontier.py',
    partId: 'frontier',
    estimatedMinutes: 15,
    markdown: `## 与 HF index 同形

写出合成 Mixtral / DeepSeek 分片，再装回 \`MoESwiGLU\` 与 \`MultiHeadLatentAttention\`，用 allclose 校验。

\`\`\`bash
python chapters/02-training/ep22_sharded_frontier.py
\`\`\`
`,
  },
  {
    id: 'paged-cuda-graph',
    ep: '23',
    title: 'Paged KV · CUDA Graph',
    summary: '块表 KV 池 + 固定 shape 的 CUDA graph 捕获。',
    script: 'chapters/01-inference/ep23_paged_cuda_graph.py',
    partId: 'frontier',
    estimatedMinutes: 18,
    markdown: `## 走向生产解码

**Paged KV** 把 token 放进固定 page，再 gather 成连续 cache 给 SDPA。

**CUDA graphs** 捕获固定 shape 的模块步进；在 KV 变成页内原地写之前，Llama 解码仍走 eager——这也是引擎下一步会 graph 的部分。

\`\`\`bash
python chapters/01-inference/ep23_paged_cuda_graph.py
\`\`\`
`,
  },
];
