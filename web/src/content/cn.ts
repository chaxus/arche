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
    markdown: `## 一句话
先看清现代 dense 语言模型的全貌——以及为什么本课不从 GPT-2 考古开始。

## 开始之前
- 第 0 部分（尤其是旅程地图）已完成。
- 约 12 分钟。
- 读完可跑配套脚本。

## 用人话讲
**Dense 解码器**从左到右读 token，预测下一个。内部是同一种积木重复很多层。现代做法更偏向：

- **RMSNorm** 稳住数值
- **RoPE** 用旋转表达位置
- **SwiGLU** 作为前馈网络配方
- **GQA** 让解码时的 KV 更省

**Llama 3.2 谱系**就是按这张零件清单走（教学目标是 1B）。能力以后来自官方权重；理解来自自己写前向。

\`\`\`mermaid
flowchart LR
  tok[分词] --> emb[嵌入]
  emb --> stack[Block_x_N]
  stack --> head[LM_head]
\`\`\`

## 图或公式
先记住上面的流程图；公式会在需要时出现。

## 对应代码
- 配置：\`arche/config.py\`（\`llama32_1b\`）
- 配套：\`chapters/01-inference/ep01_overview.py\` 打印层数、头数、维度——不是聊天。

## 现在就做
\`\`\`bash
python chapters/01-inference/ep01_overview.py
\`\`\`
应看到词表大小、层数、头数等数字。

## 如果翻车了
- \`No module named arche\`：激活 \`.venv\`，在仓库根目录运行。
- 找不到脚本：先 \`cd\` 到 \`arche\` 目录。

## 先别深挖
训练超大模型、线上服务、所有 Llama 变体。

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
    markdown: `## 一句话
文字要先变成整数 **token id**，模型才能用向量计算。

## 开始之前
- 已读总览。
- 约 15 分钟。
- 官方 Llama 3.2 分词器在 Hugging Face 上是**门禁**的——跑本章实验时再登录即可。

## 用人话讲
模型不像人那样直接读字母。**分词器**把文本切成 token，并映射到固定词表里的编号（Llama 3.2：128256）。

我们用**官方**分词器。本课重点是神经网络主干，不重造 BPE。

## 对应代码
- \`arche/tokenizer.py\`
- 配套：\`chapters/01-inference/ep02_tokenizer.py\`

## 现在就做
\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep02_tokenizer.py
\`\`\`
会打印示例字符串的 token id。若暂时不登录，可先继续读，稍后再回来跑。

## 如果翻车了
- 401/403：先在网页接受模型协议，再 login。
- 下载慢：可设 \`export HF_ENDPOINT=https://hf-mirror.com\`。

## 先别深挖
自己训练分词器，或纠结所有 Unicode 边角。

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
    markdown: `## 一句话
每个 token id 变成一串学来的数字——大查找表里的一行。

## 开始之前
- 理解「先有 token id」。
- 约 12 分钟。

## 用人话讲
把它想成词典：token id → 长度为 \`embed_dim\` 的向量。这张表就是 **TokenEmbedding**。

现代 dense 模型通常**不会**在这里再加一张绝对位置表。位置稍后由注意力里的 **RoPE** 提供。

Llama 3.2 还会把嵌入与最终 LM head **绑权**（共用同一份数字）。

## 图或公式
$$X = \\mathrm{Embed}(t_{1:L}) \\in \\mathbb{R}^{L \\times d}$$

## 对应代码
- \`arche/embedding.py\`
- 配套：\`chapters/01-inference/ep03_embedding.py\`

## 现在就做
\`\`\`bash
python chapters/01-inference/ep03_embedding.py
\`\`\`
关注形状是否像 \`(batch, seq, embed_dim)\`。

## 如果翻车了
- 形状对不上：多半改乱了示例尺寸——用 git 恢复脚本。

## 先别深挖
嵌入如何训练、segment id 等花样。

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
    markdown: `## 一句话
RMSNorm 让层输入保持合适的尺度，又不必像 LayerNorm 那样先减均值。

## 开始之前
- 嵌入章已完成。约 12 分钟。

## 用人话讲
网络很深时，数字容易爆掉或消失。**RMSNorm** 用均方根大小去除尺度，再乘可学习增益；不做减均值，更便宜，对现代 LLM 也够稳。
每个 block 里它出现在注意力**之前**和 MLP**之前**（pre-norm）。

## 图或公式
$$\\mathrm{RMSNorm}(x) = \\frac{x}{\\sqrt{\\mathrm{mean}(x^2)+\\varepsilon}} \\odot \\gamma$$

## 对应代码
- \`arche/norms.py\`
- \`chapters/01-inference/ep04_rmsnorm.py\`

## 现在就做
\`\`\`bash
python chapters/01-inference/ep04_rmsnorm.py
\`\`\`

## 如果翻车了
- 实验里出现 NaN：多半改了 epsilon——用 git 恢复脚本。

## 先别深挖
文献里所有 LayerNorm 变体。

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
    markdown: `## 一句话
RoPE 旋转 Q/K，让注意力带上相对位置——不需要绝对位置表。

## 开始之前
- 已读 RMSNorm。约 15 分钟。

## 用人话讲
不靠「加一个位置向量」，**RoPE** 按 token 下标旋转成对维度；相对距离会从点积里出现。
Llama 3.2 还有很大的 \`rope_theta\`，以及后面的长上下文 **scale**（单独一章）。

## 图或公式
$$q_m' = R_{\\Theta,m} q_m,\\quad k_n' = R_{\\Theta,n} k_n$$

## 对应代码
- \`arche/rope.py\`
- \`chapters/01-inference/ep05_rope.py\`

## 现在就做
\`\`\`bash
python chapters/01-inference/ep05_rope.py
\`\`\`

## 如果翻车了
- head 维度为奇数会破坏成对旋转——请用脚本默认值。

## 先别深挖
所有 RoPE 变体；长上下文缩放到 ep13。

## 下一章
**GQA 注意力**（\`?c=gqa\`）。

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
    markdown: `## 一句话
GQA 让多个 query 头共享 KV 头，从而减小解码时的显存/内存。

## 开始之前
- 理解 RoPE。约 18 分钟。

## 用人话讲
经典多头：每个 query 头自带 K/V，准确但 **KV cache** 很大。
**GQA** 减少 KV 头，一组 query 共用。Llama 3.2 1B：32 个 query 头、8 个 KV 头。
流程：投影 → RoPE → 重复 KV → 因果注意力 → 输出投影。

## 对应代码
- \`arche/attention.py\`
- \`chapters/01-inference/ep06_gqa.py\`

## 现在就做
\`\`\`bash
python chapters/01-inference/ep06_gqa.py
\`\`\`

## 如果翻车了
- 必须满足 \`num_heads % num_kv_heads == 0\`。

## 先别深挖
FlashAttention；MLA 稍后讲。

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
    markdown: `## 一句话
SwiGLU 是现代前馈块：gate × up，再 down——三条无偏置线性层。

## 开始之前
- GQA 已完成。约 12 分钟。

## 用人话讲
老教程常见 Linear → GELU → Linear。Llama 风格更常用：
\`down(silu(gate(x)) * up(x))\`。

## 图或公式
$$\\mathrm{FFN}(x)=W_d\\big(\\mathrm{SiLU}(W_g x)\\odot W_u x\\big)$$

## 对应代码
- \`arche/mlp.py\`
- \`chapters/01-inference/ep07_swiglu.py\`

## 现在就做
\`\`\`bash
python chapters/01-inference/ep07_swiglu.py
\`\`\`

## 如果翻车了
- 形状不对：检查 intermediate_dim 是否与三条线性层一致。

## 先别深挖
MoE（多专家）在第 2 部分。

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
    markdown: `## 一句话
一个 Transformer 块就是可堆叠的 pre-norm 残差三明治。

## 开始之前
- 注意力与 SwiGLU 已完成。约 12 分钟。

## 用人话讲
\`\`\`
x → RMSNorm → Attn → +x → RMSNorm → SwiGLU → +x
\`\`\`
进出同形状，所以深度只是循环。Llama 3.2 1B 堆 16 次（3B：28）。

## 对应代码
- \`arche/block.py\`
- \`chapters/01-inference/ep08_block.py\`

## 现在就做
\`\`\`bash
python chapters/01-inference/ep08_block.py
\`\`\`

## 如果翻车了
- 残差相加要求形状一致。

## 先别深挖
某些巨型模型的并行 Attention/MLP 布局。

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
    markdown: `## 一句话
Embed → N 个 block → 最终 RMSNorm → 线性到词表分数（常与嵌入绑权）。

## 开始之前
- block 章已完成。约 12 分钟。

## 用人话讲
\`ids → Embedding → [Block × N] → RMSNorm → Linear(vocab)\`。
最后线性输出下一个 token 的分数（logits）。Llama 3.2 把 \`lm_head\` 与嵌入绑在一起。

## 对应代码
- \`arche/model.py\`
- \`chapters/01-inference/ep09_stack.py\`

## 现在就做
\`\`\`bash
python chapters/01-inference/ep09_stack.py
\`\`\`

## 如果翻车了
- 词表大小不一致：用仓库提供的配置助手。

## 先别深挖
加载 Meta 权重（下一章）与采样技巧。

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
    markdown: `## 一句话
把官方 Hugging Face Llama 3.2 张量映射进手写的 \`arche.Llama\`。

## 开始之前
- 堆叠章已完成。约 20 分钟。
- 需要 HF 登录并接受协议（见第 0 部分地图）。

## 用人话讲
模块建好后还是空的。\`from_pretrained\` 搭好我们的图，下载 HF 权重，再按 key 对照表拷贝。

## 对应代码
- \`arche/load_weights.py\`
- \`chapters/01-inference/ep10_load_weights.py\`

## 现在就做
\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
\`\`\`

## 如果翻车了
- 门禁错误：接受协议并重新登录；网络问题可试镜像。
- 磁盘/内存不够：关掉其它应用；目标是 1B。

## 先别深挖
量化部署格式、多卡分片。

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
    markdown: `## 一句话
把 logits 变成 token：采样（或贪心）选出下一个 id，拼上，重复直到结束。

## 开始之前
- 理解权重章（即使下载延后）。约 15 分钟。

## 用人话讲
每步看最后位置的分数。**temperature** 调节尖锐程度；**top-k** 只保留前 k 个。拼上 id；遇到 EOS 或长度上限就停。

## 对应代码
- \`arche/generate.py\`
- \`chapters/01-inference/ep11_generate.py\`

## 现在就做
\`\`\`bash
python chapters/01-inference/ep11_generate.py
\`\`\`
没有权重时，小 demo 仍能展示循环形状。

## 如果翻车了
- 随机权重输出乱码是正常的——流畅性来自 1B 权重。

## 先别深挖
KV cache（下一部分）与 chat template（更后）。

## 下一章
第 2 部分开始：**KV 缓存**（\`?c=kv-cache\`）。

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
    markdown: `## 一句话
先对 prompt 做一次 prefill，再解码时复用已存的 K/V。

## 开始之前
- 生成章已完成。约 15 分钟。

## 用人话讲
没有 cache 时，每来一个新 token 都要对整段前缀重算注意力。
有了 **KV cache**：prompt 的 K/V 算一次；之后每个新 token 只追加自己的 K/V。

## 对应代码
- \`arche/cache.py\`；\`generate(..., use_cache=True)\`
- \`chapters/01-inference/ep12_kv_cache.py\`

## 现在就做
\`\`\`bash
python chapters/01-inference/ep12_kv_cache.py
\`\`\`

## 如果翻车了
- \`start_pos\` 乱了：恢复章节脚本。

## 先别深挖
Paged KV（ep23）与生产调度器。

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
    markdown: `## 一句话
Llama 3.2 拉伸长波长，让上下文能远超原始窗口。

## 开始之前
- RoPE 与 KV cache。约 15 分钟。

## 用人话讲
按短上下文训练的 RoPE，突然拉到很长会出问题。
Llama 3.2 保留大 \`rope_theta\`，再用 **factor / low / high** 调度：短波长不动，长波长拉伸，中间平滑。

## 对应代码
- \`arche/rope.py\` / attention 配置
- \`chapters/01-inference/ep13_rope_scaling.py\`

## 现在就做
\`\`\`bash
python chapters/01-inference/ep13_rope_scaling.py
\`\`\`

## 如果翻车了
- 小 demo 看不到 128k：关注打印的频率变化即可。

## 先别深挖
YaRN/NTK 争论；这里教本仓库用的 Llama 3.2 日程。

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
    markdown: `## 一句话
MoE 每个 token 只跑 top-k 个专家 MLP——算力稀疏，参数可以很多。

## 开始之前
- 理解 SwiGLU。约 15 分钟。

## 用人话讲
**路由器**给专家打分；每个 token 选 \`top_k\` 个再加权混合。
这是 Mixtral 类堆叠背后的稀疏 FFN 思路。

## 对应代码
- \`arche/moe.py\`
- \`chapters/01-inference/ep14_moe.py\`

## 现在就做
\`\`\`bash
python chapters/01-inference/ep14_moe.py
\`\`\`

## 如果翻车了
- \`top_k\` 不能大于专家数。

## 先别深挖
负载均衡损失与专家并行。

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
    markdown: `## 一句话
MLA 把 KV 压进低秩 latent 缓存（DeepSeek 风格）。

## 开始之前
- 有 KV cache 直觉。约 18 分钟。

## 用人话讲
不缓存每个 head 的完整 K/V，**MLA** 存 latent（外加一小段 RoPE 流），用时再展开。
注意力思路还在，解码更省内存。

## 对应代码
- \`arche/mla.py\`
- \`chapters/01-inference/ep15_mla.py\`

## 现在就做
\`\`\`bash
python chapters/01-inference/ep15_mla.py
\`\`\`

## 如果翻车了
- rank/head_dim 不一致：用章节默认构造参数。

## 先别深挖
DeepSeek 生产内核——key 映射在第 4 部分。

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
    markdown: `## 一句话
选择设备，可选 INT8 存权重，并用模板渲染对话。

## 开始之前
- 会跑脚本。约 15 分钟。

## 用人话讲
\`pick_device\` 优先 CUDA → MPS → CPU。
**INT8 权重量化**更省空间（教学级）。
**Chat template** 格式化角色对话；有 HF 模板用官方的，否则回退纯文本。

## 对应代码
- \`arche/device.py\`、\`quantize.py\`、\`chat.py\`
- \`chapters/01-inference/ep16_device_quant_chat.py\`

## 现在就做
\`\`\`bash
python chapters/01-inference/ep16_device_quant_chat.py
\`\`\`
也可点页面上的小 demo。

## 如果翻车了
- 设备错误：强制 \`"cpu"\`。

## 先别深挖
GPTQ/AWQ 生产量化。

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
    markdown: `## 一句话
教学调度器让多个短解码任务一起步进，直到各自结束。

## 开始之前
- 生成与 KV。约 15 分钟。

## 用人话讲
线上服务不会永远「一个接一个」。它们维护活跃序列池，一起步进，完成的退役。
我们的版本每步 pad——**不是** paged attention，但控制流同形。

## 对应代码
- \`arche/batching.py\`
- \`chapters/01-inference/ep17_continuous_batching.py\`

## 现在就做
\`\`\`bash
python chapters/01-inference/ep17_continuous_batching.py
\`\`\`

## 如果翻车了
- 空输出：保证 \`max_new_tokens >= 1\`。

## 先别深挖
CUDA graph 与 paged 融合（ep23）。

## 下一章
第 3 部分：**预训练**（\`?c=pretrain\`）。

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
