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
    markdown: `## 要点

手写 **Llama 3.2 风格** dense 解码器：RMSNorm、RoPE、SwiGLU、GQA。教学目标尺寸：**1B**。

\`\`\`mermaid
flowchart LR
  tok[分词] --> emb[嵌入]
  emb --> stack[堆叠]
  stack --> head[LM head]
\`\`\`

## 动手

\`\`\`bash
python chapters/01-inference/ep01_overview.py
\`\`\`

打印层数/头数/维度等配置，不是聊天。

## 下一章

**分词器**（\`?c=tokenizer\`）。

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
    markdown: `## 要点

文字经官方 Llama 3.2 BPE 变成 **token id**（词表 \`128256\`）。不重造 BPE；跑实验需要 HF 门禁登录。

## 动手

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep02_tokenizer.py
\`\`\`

## 下一章

**嵌入层**（\`?c=embedding\`）。

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
    markdown: `## 要点

\`TokenEmbedding\`：id → 向量 \`(vocab, dim)\`。这里不加绝对位置表——位置靠后面的 **RoPE**。Llama 3.2 与 LM head 绑权。

## 动手

\`\`\`bash
python chapters/01-inference/ep03_embedding.py
\`\`\`

## 下一章

**RMSNorm**（\`?c=rmsnorm\`）。

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
    markdown: `## 要点

**RMSNorm** 用均方根缩放（不减均值）。**Pre-norm**：注意力前、MLP 前各一次。

$$\\mathrm{RMSNorm}(x)=x/\\sqrt{\\mathrm{mean}(x^2)+\\varepsilon}\\odot\\gamma$$

## 动手

\`\`\`bash
python chapters/01-inference/ep04_rmsnorm.py
\`\`\`

## 下一章

**RoPE**（\`?c=rope\`）。

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
    demo: 'rope',
    markdown: `## 要点

**RoPE** 按位置旋转 Q/K，注意力自然带上相对位置。本页可拖互动图。

## 动手

\`\`\`bash
python chapters/01-inference/ep05_rope.py
\`\`\`

## 下一章

**GQA**（\`?c=gqa\`）。

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
    demo: 'gqa',
    markdown: `## 要点

**GQA**：KV 头少于 query 头（1B：32 Q / 8 KV），解码 cache 更小。见本页互动图。

流程：投影 → RoPE → 重复 KV → 因果注意力 → 输出投影。

## 动手

\`\`\`bash
python chapters/01-inference/ep06_gqa.py
\`\`\`

## 下一章

**SwiGLU**（\`?c=swiglu\`）。

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
    markdown: `## 要点

现代 FFN：\`down(silu(gate(x)) * up(x))\`——三条无偏置线性层。

## 动手

\`\`\`bash
python chapters/01-inference/ep07_swiglu.py
\`\`\`

## 下一章

**Transformer 块**（\`?c=block\`）。

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
    markdown: `## 要点

\`\`\`
x → RMSNorm → Attn → +x → RMSNorm → SwiGLU → +x
\`\`\`

进出同形状，循环堆叠（1B：16 层）。

## 动手

\`\`\`bash
python chapters/01-inference/ep08_block.py
\`\`\`

## 下一章

**堆叠 + LM head**（\`?c=stack\`）。

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
    markdown: `## 要点

\`ids → Embed → [Block×N] → RMSNorm → Linear(vocab)\`。最后一层分数即下一 token logits；与嵌入绑权。

## 动手

\`\`\`bash
python chapters/01-inference/ep09_stack.py
\`\`\`

## 下一章

**加载权重**（\`?c=weights\`）。

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
    markdown: `## 要点

\`from_pretrained\` 搭好我们的图，再按 key 映射拷入 HF Llama 3.2 权重。需门禁登录。

## 动手

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
\`\`\`

## 下一章

**文本生成**（\`?c=generate\`）。

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
    markdown: `## 要点

循环：取最后一步 logits →（可选 temperature / top-k）→ 采样或贪心 → 拼上 → EOS/长度停。

## 动手

\`\`\`bash
python chapters/01-inference/ep11_generate.py
\`\`\`

## 下一章

**KV 缓存**（\`?c=kv-cache\`）。

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
    demo: 'kv',
    markdown: `## 要点

Prompt 的 K/V 算一次；之后每步只追加。见本页互动条。

## 动手

\`\`\`bash
python chapters/01-inference/ep12_kv_cache.py
\`\`\`

## 下一章

**RoPE 缩放**（\`?c=rope-scaling\`）。

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
    markdown: `## 要点

Llama 3.2 长上下文：大 \`rope_theta\`，按 factor/low/high 拉伸长波长。

## 动手

\`\`\`bash
python chapters/01-inference/ep13_rope_scaling.py
\`\`\`

## 下一章

**MoE**（\`?c=moe\`）。

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
    markdown: `## 要点

路由器每 token 只跑 **top-k** 个 SwiGLU 专家——参数可以很多，算力稀疏（Mixtral 思路）。

## 动手

\`\`\`bash
python chapters/01-inference/ep14_moe.py
\`\`\`

## 下一章

**MLA**（\`?c=mla\`）。

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
    markdown: `## 要点

**MLA** 缓存低秩 latent（加一小段 RoPE 流），而不是每头完整 K/V——解码更省内存。

## 动手

\`\`\`bash
python chapters/01-inference/ep15_mla.py
\`\`\`

## 下一章

**设备 · INT8 · 对话**（\`?c=device-quant-chat\`）。

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
    markdown: `## 要点

\`pick_device\`（CUDA→MPS→CPU）、教学用 INT8 权重、chat template（HF 或纯文本回退）。

## 动手

\`\`\`bash
python chapters/01-inference/ep16_device_quant_chat.py
\`\`\`

## 下一章

**连续批处理**（\`?c=continuous-batching\`）。

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
    markdown: `## 要点

教学调度器：多个短任务一起步进，结束就退役。不是 paged attention。

## 动手

\`\`\`bash
python chapters/01-inference/ep17_continuous_batching.py
\`\`\`

## 下一章

**预训练**（\`?c=pretrain\`）。

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
    markdown: `## 要点

玩具 AdamW：预测下一 token、交叉熵、一步更新。体会 API，不是全网训练。

## 动手

\`\`\`bash
python chapters/02-training/ep18_pretrain.py
\`\`\`

## 下一章

**SFT · DPO**（\`?c=sft-dpo\`）。

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
    markdown: `## 要点

**SFT**：只监督回复段。**DPO**：相对冻结 reference，偏好 chosen（教学一步即可）。

## 动手

\`\`\`bash
python chapters/02-training/ep19_sft_dpo.py
\`\`\`

## 下一章

**前沿检查点**（\`?c=frontier-ckpt\`）。

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
    markdown: `## 要点

存取 MoE/MLA；把 Mixtral \`w1/w2/w3\` 映射进我们的模块。

## 动手

\`\`\`bash
python chapters/02-training/ep20_frontier_ckpt.py
\`\`\`

## 下一章

**HF 对齐**（\`?c=hf-align\`）。

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
    markdown: `## 要点

有门禁权重时对比 arche 与 HF logits（argmax + max-abs）。离线则干净退出；CI 仍测合成映射。

## 动手

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
\`\`\`

## 下一章

**分片 MoE/MLA**（\`?c=sharded-frontier\`）。

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
    markdown: `## 要点

合成 \`index.json\` + \`.pt\` 分片 → 装回 Mixtral 形 MoE / DeepSeek 形 MLA（allclose）。

## 动手

\`\`\`bash
python chapters/02-training/ep22_sharded_frontier.py
\`\`\`

## 下一章

**Paged KV · CUDA Graph**（\`?c=paged-cuda-graph\`）。

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
    markdown: `## 要点

**Paged KV**：块表 + gather。**CUDA graph**：有 CUDA 时捕获固定 shape 计算（KV 原地写之前解码仍 eager）。

## 动手

\`\`\`bash
python chapters/01-inference/ep23_paged_cuda_graph.py
\`\`\`

## 下一章

路径走完。可回看薄弱章，或有权重时跑 HF 对齐。

`,
  },
];
