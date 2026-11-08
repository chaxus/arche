import type { Chapter } from './types';
import { PART0_CN } from './part0-cn';

export const CHAPTERS_CN: Chapter[] = [
  ...PART0_CN,
  {
    id: 'overview',
    ep: '01',
    title: '总览',
    summary: '为什么用 Llama 3.2 当标本；整条「猜下一个词」链路长什么样。',
    script: 'chapters/01-inference/ep01_overview.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    markdown: `## 要点

我们要亲手写出一个小 **Llama 3.2**（约 10 亿参数）：够小能装进脑子，权重能下载，零件和生产里常见的一样。业界常把这种「层层都算、猜下一个词」的结构叫做 **dense 解码器**——名字后面混熟即可。目标：前向每一块都自己写过。

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
    summary: '为什么分词、业界怎么切、如何评判，以及为何必须用官方表。',
    script: 'chapters/01-inference/ep02_tokenizer.py',
    partId: 'trunk',
    estimatedMinutes: 22,
    demo: 'tokenizer',
    markdown: `## 要点

### 1. 为什么要分词

模型只会算数字，不会直接读「你好世界」。

- 整句当一个符号 → 词表炸、没见过的话无法表示  
- 拆成纯字节 → 序列太长，又慢又占上下文  

**分词**找中间态：稳定、可学习、长度可控。  
一句话：**同一套规则，进模型前编码，出模型后再解码。**

### 2. 分词器做什么

1. **切** — 文字 → 小段（**token**）  
2. **编号** — 查固定词表 → **token id**（整数）  

之后整条链路只认 id；生成结束用**同一张表**解码。开始/结束等特殊符也是表里的固定 id。

\`\`\`mermaid
flowchart LR
  text[文字] --> cut[切成 token]
  cut --> ids[token id]
  ids --> model[模型]
  model --> ids2[新 id]
  ids2 --> back[解码成文字]
\`\`\`

本页上方可先玩**教学小 BPE**（看切法直觉）；下方脚本才用**官方 Llama 3.2 表**。

### 3. 常见切法，以及三个名字

| 切法 | 人话 | 今天 |
| --- | --- | --- |
| 按词 / 空格 | 英文空格切；中文还要另分 | 大模型几乎不作主方案 |
| 按字符 / 字节 | 一字或一字节一个 id | 当保底层；纯用太长 |
| **BPE** | 按频率合并相邻片 | **开放解码器主流** |
| **WordPiece** | 也按子词，合并规则不同 | BERT 一代多 |
| **Unigram**（常经 **SentencePiece** 训） | 从大词表删片 | 多语言/部分栈仍用 |

**BPE**：从字节/字符起，谁俩最常挨着就合并谁，反复合并成词表。常见片段容易变成单独 token（如 \`the\`、\`ing\`）。

**WordPiece**：也是子词，但更看「合起来是否提高语料似然」。和 BPE **外表像、规则不同，表不能混用**。

**Unigram / SentencePiece**：Unigram 是「先大后删」的训练目标；SentencePiece 多半是**训/存的工具**（常直接切原文，方便中文）。有人说「SentencePiece 分词」≈ 用该工具训出的 Unigram（或 BPE）。

三者都是**子词**：不像整词那么脆，不像纯字节那么碎。产品差异更常出在**语料、预分词正则、词表大小**。

### 4. 业界实际用啥

| 系 | 引擎 | 约词表 | 备注 |
| --- | --- | --- | --- |
| GPT-4 / 3.5 | tiktoken BPE \`cl100k\` | ~100K | 切法库开源 |
| GPT-4o / o | tiktoken BPE \`o200k\` | ~200K | 更大，多语言常更省 |
| Llama 3 / 3.2 | 字节 BPE | 128K+ | **本课官方表** |
| DeepSeek-V3 | 字节 BPE | ~128K | 调预分词/语料做多语言压缩 |
| Kimi K2 | tiktoken 风 BPE | ~163K | 汉字更友好的正则 |
| Qwen 等 | BPE | ~150K | 同路线，表各家自训 |

结论：大家都在 **字节级 BPE / tiktoken 系**里抠细节，不是有的按词、有的按字分家。

### 5. 怎样算「好」（含「权重」是什么）

| 标准 | 人话 |
| --- | --- |
| **压缩 / fertility** | 同样文字几个 token；越少越省。fertility = 平均一词切几片，**越大越碎** |
| **语言公平 / parity** | 中英等同义句 token 数别差太多，否则多语言偏科、更贵 |
| **结构** | 别切坏缩进、换行、URL、数字 |
| **与检查点配对** | 见下例——这是硬约束 |
| **词表大小 / 速度 / 可逆 / 特殊符 / 下游分数** | 越大常越省但嵌入更胖；要快、能还原、标记清晰；最终仍看训完好不好 |

**「必须配对同一套权重」——以及：平时为什么好像不会换表？**

下一章：每个 id 去**嵌入表**取一行数字（这些数是模型**权重**的一部分）。  
正常用法本来就是：**怎么切，就怎么取**——一份检查点配它出厂的那张分词表，你不会中途随便换。

那为什么还强调「换切法会乱」？因为**世界上本来就有很多张表**，乱通常出在**配错对**，不是同一次推理里改主意：

| 何时会「表不一样」 | 例子 |
| --- | --- |
| **换模型家族** | Llama 表 ≠ Qwen 表 ≠ GPT 的 \`cl100k\` / \`o200k\` |
| **同厂换代** | GPT-3.5 ~100K → GPT-4o ~200K（为多语言等重训/扩表） |
| **自己从头训** | 按语料新训一张 BPE；旧表装不上新嵌入 |
| **加特殊符** | 微调/工具调用往表尾加标记；合并规则多半不变，id 空间变了 |
| **操作失误** | 权重是 Llama，却加载了别家 \`tokenizer.json\` |

所以：

- **用同一个模型时**：怎么分就怎么取，不会无故换表。  
- **强调配对**：避免「这份权重 + 那家的分词」；也解释为何各厂表不同、换代会扩表。  

举例：训练时 \`the\` → **65** → 第 65 行是「the」。若误用另一张表变成 \`the\` → **902**，模型仍读 902 行 → 乱。

\`\`\`mermaid
flowchart LR
  pair[正确: Llama权重 + Llama分词表] --> ok[编号对上]
  mismatch[错误: Llama权重 + 别家分词表] --> bad[编号错位 → 乱]
\`\`\`

论文口径：压缩、公平**有用但不是唯一分数**；多语言要综合看。

### 6. 本课怎么做

用官方 **Llama 3.2** BPE（\`128256\`）：不是因为「我们可能换表」，而是**加载这份权重时只能配这张表**。教学面板只演示 BPE 直觉，不替代官方表。

## 动手

先接受 Hugging Face 上 \`meta-llama/Llama-3.2-1B\` 许可，再：

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep02_tokenizer.py
\`\`\`

预期：token id、解码回原句、bos / eos / pad。大陆网络可：\`export HF_ENDPOINT=https://hf-mirror.com\`

## 翻车

本章要下门禁分词器，卡住再看：

- 未许可 / 未 login → 看脚本提示  
- \`401\` → 模型页 Agree 后重新 login  
- 下载慢 → HF 镜像  

## 下一章

**嵌入层**（\`?c=embedding\`）：id 怎样变成那一行向量。

`,
  },
  {
    id: 'embedding',
    ep: '03',
    title: '嵌入层',
    summary: 'token id 变成向量——第一层可学习查找。',
    script: 'chapters/01-inference/ep03_embedding.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    markdown: `## 要点

token id 还只是整数。模型要能做乘法，需要每个 id 对应一串数字——**向量**。

**嵌入（embedding）**就是一张查找表，形状 \`(词表大小，维度)\`：第 \`i\` 行是 id \`i\` 的向量。前向一次 = 按 id 把行取出来。这里还不「理解」语义，只是符号到几何的第一层可学习映射。

Llama 3.2 风格里两点要记住：

- **不加**单独的绝对位置向量表；**位置后面用 RoPE** 加在 Q/K 上。
- 这张表常常和最后的 LM head **绑权**（同一套权重：id→向量，向量→词表分数）。

实验用很小的假词表，免下载权重；形状和真模型一样：\`(batch, seq) → (batch, seq, dim)\`。

## 动手

\`\`\`bash
python chapters/01-inference/ep03_embedding.py
\`\`\`

预期：打印 ids / 输出形状，并提醒位置交给 RoPE。

## 下一章

**RMSNorm**（\`?c=rmsnorm\`）：进重层前先稳住向量尺度。

`,
    latex: 'X = \\mathrm{Embed}(t_{1:L}) \\in \\mathbb{R}^{L \\times d}',
  },
  {
    id: 'rmsnorm',
    ep: '04',
    title: 'RMSNorm',
    summary: '注意力和 MLP 前，把向量尺度稳住。',
    script: 'chapters/01-inference/ep04_rmsnorm.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    markdown: `## 要点

层一深，数值容易炸或塌。**归一化**把向量尺度拉回可控范围。

**RMSNorm**（均方根归一化）是这里的现代默认：

- 对每个 token 向量，在最后一维算 RMS  
- 除以 RMS（再加一点 \`ε\`）  
- 再乘可学习缩放 \`γ\`  
- **不减均值**（和经典 LayerNorm 不同）——更省，实践里也够稳  

我们用 **pre-norm**：注意力前一次、MLP 前一次。Llama / Qwen 一类 dense 栈就是这样。

$$\\mathrm{RMSNorm}(x)=x/\\sqrt{\\mathrm{mean}(x^2)+\\varepsilon}\\odot\\gamma$$

## 动手

\`\`\`bash
python chapters/01-inference/ep04_rmsnorm.py
\`\`\`

预期：进出同形状；\`γ\` 初始为 1 时，归一化后每 token 的 RMS 约等于 1。

## 下一章

**RoPE**（\`?c=rope\`）：用旋转把位置写进 Q/K。

`,
    latex:
      '\\mathrm{RMSNorm}(x) = \\frac{x}{\\sqrt{\\mathrm{mean}(x^2)+\\varepsilon}} \\odot \\gamma',
  },
  {
    id: 'rope',
    ep: '05',
    title: 'RoPE',
    summary: '用旋转把位置写进 Q/K，而不是贴在嵌入上。',
    script: 'chapters/01-inference/ep05_rope.py',
    partId: 'trunk',
    estimatedMinutes: 15,
    demo: 'rope',
    markdown: `## 要点

注意力靠 Q 和 K 做点积。若完全没有位置，「先 A 后 B」和「先 B 后 A」看起来一样。

**RoPE**（旋转位置编码）按**位置**给每个 Q/K 转一个角。相对距离会自然进点积——不必在嵌入上再贴一张大正弦位置表。

Llama 3.2 用很大的基频（\`rope_theta\`）照顾长文；后面还有**波长缩放**专讲更长上下文。

本页可拖互动图。代码里：预计算 cos/sin，\`rotate_half\`，再 \`q' = q·cos + rotate(q)·sin\`（K 同理）。

## 动手

\`\`\`bash
python chapters/01-inference/ep05_rope.py
\`\`\`

预期：演示一次 RoPE 应用并核对形状。

## 下一章

**GQA**（\`?c=gqa\`）：KV 头更少的注意力。

`,
    latex:
      'q_m^{\\prime} = R_{\\Theta,m} q_m,\\quad k_n^{\\prime} = R_{\\Theta,n} k_n',
  },
  {
    id: 'gqa',
    ep: '06',
    title: 'GQA 注意力',
    summary: '多组 query 共用 KV，解码缓存更小。',
    script: 'chapters/01-inference/ep06_gqa.py',
    partId: 'trunk',
    estimatedMinutes: 18,
    demo: 'gqa',
    markdown: `## 要点

经典多头：每个 query 头配自己的 K/V 头。解码时 **KV 缓存**（后面章节）会很大。

**GQA**（分组查询注意力）：KV 头少于 query 头，多个 query **共用**一组 K/V。Llama 3.2 1B 是 **32 个 Q / 8 个 KV**——质量差不多，缓存更小。

仍然是**因果**：位置 \`t\` 只能看见 \`≤ t\`。

代码流程：投影 Q/K/V → 拆头 → RoPE → \`repeat_interleave\` 扩 KV → 缩放点积注意力 → 输出投影。

## 动手

\`\`\`bash
python chapters/01-inference/ep06_gqa.py
\`\`\`

预期：一次小前向，打印头数与输出形状。可看本页互动图。

## 下一章

**SwiGLU**（\`?c=swiglu\`）：每一块里的前馈半边。

`,
  },
  {
    id: 'swiglu',
    ep: '07',
    title: 'SwiGLU',
    summary: '现代前馈：门控 SiLU × up，再 down。',
    script: 'chapters/01-inference/ep07_swiglu.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    markdown: `## 要点

注意力混合过各个位置之后，每个位置还要过一个**按位置的 MLP** 变换特征。

现代栈用 **SwiGLU**，而不是简单的 GELU MLP：

$$\\mathrm{FFN}(x)=W_d\\big(\\mathrm{SiLU}(W_g x)\\odot W_u x\\big)$$

三条**无偏置**线性层：**gate**、**up**、**down**。gate 走 SiLU，再与 up 相乘，最后 down 投回。宽度思路与 Llama / Qwen dense 一致。

## 动手

\`\`\`bash
python chapters/01-inference/ep07_swiglu.py
\`\`\`

预期：进出同为 \`embed_dim\`，中间层更宽。

## 下一章

**Transformer 块**（\`?c=block\`）：Norm + 注意力 + SwiGLU + 残差拼起来。

`,
    latex: '\\mathrm{FFN}(x)=W_d\\big(\\mathrm{SiLU}(W_g x)\\odot W_u x\\big)',
  },
  {
    id: 'block',
    ep: '08',
    title: 'Transformer 块',
    summary: '可反复堆叠的一层残差三明治。',
    script: 'chapters/01-inference/ep08_block.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    markdown: `## 要点

一个 **Transformer 块**就是可重复的三明治：

\`\`\`
x → RMSNorm → 注意力 → + 残差
  → RMSNorm → SwiGLU → + 残差
\`\`\`

进出同形状，所以叠 \`N\` 次（Llama 3.2 1B：**16** 层）。现代 dense 用 **pre-norm**（子层**前**归一化）；残差让梯度还能流。

注意力可带 KV cache，解码时用（马上会用到）。

## 动手

\`\`\`bash
python chapters/01-inference/ep08_block.py
\`\`\`

预期：一块前向；残差路径上形状不变。

## 下一章

**堆叠 + LM head**（\`?c=stack\`）：嵌入 → 多层块 → 词表分数。

`,
  },
  {
    id: 'stack',
    ep: '09',
    title: '堆叠 + LM head',
    summary: '整条路：嵌入 → 多层块 → 词表分数。',
    script: 'chapters/01-inference/ep09_stack.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    markdown: `## 要点

整模路径：

\`token id → 嵌入 → Block×N → 最终 RMSNorm → Linear(词表)\`

最后的线性层是 **LM head**：每个位置给词表里每个词一个分数（**logits**）。取最大（或采样）就是下一个 token。

Llama 3.2 常把 LM head 与嵌入表 **绑权**——一张矩阵两个用途。

## 动手

\`\`\`bash
python chapters/01-inference/ep09_stack.py
\`\`\`

预期：随机小模型前向，logits 形状 \`(batch, seq, vocab)\`。

## 下一章

**加载权重**（\`?c=weights\`）：用真 Llama 张量替换随机初始化。

`,
  },
  {
    id: 'weights',
    ep: '10',
    title: '加载权重',
    summary: '把真 Llama 3.2 张量拷进我们的模块。',
    script: 'chapters/01-inference/ep10_load_weights.py',
    partId: 'trunk',
    estimatedMinutes: 20,
    markdown: `## 要点

到目前图是对的，但权重是随机的——输出没有意义。

**\`from_pretrained\`** 先搭好我们的 \`Llama\` 模块，再按 **key 映射**（HF 名 → arche 名）把官方 Hugging Face Llama 3.2 检查点拷进来。之后前向应与发布模型一致（差在数值误差）。

需要接受模型许可并 \`huggingface-cli login\`（和分词器同一道门）。

## 动手

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
\`\`\`

预期：确认权重已装入（参数量 / 简短自检）。镜像：\`export HF_ENDPOINT=https://hf-mirror.com\`。

## 翻车

与分词器相同：许可、登录、网络。

## 下一章

**文本生成**（\`?c=generate\`）：把 logits 一个个变成字。

`,
  },
  {
    id: 'generate',
    ep: '11',
    title: '文本生成',
    summary: '循环：logits → 采样 → 拼上 → 再来。',
    script: 'chapters/01-inference/ep11_generate.py',
    partId: 'trunk',
    estimatedMinutes: 15,
    markdown: `## 要点

生成是一个**循环**，不是算一次就完：

1. 用当前 id 跑模型 → 取**最后位置**的 logits  
2. 可选：**temperature** 软化，**top-k** 截断  
3. **贪心**（\`argmax\`）或按分布**采样**  
4. 把新 id 拼上；遇到 EOS 或达到长度就停  

有了 KV 缓存（下一章），第 1 步在首次 prefill 之后会便宜很多。本章先把采样循环本身搞清。

## 动手

\`\`\`bash
python chapters/01-inference/ep11_generate.py
\`\`\`

预期：一段短解码文本（真文案需要上一章的权重；否则看脚本说明）。

## 下一章

**KV 缓存**（\`?c=kv-cache\`）：别每步都把整段 prompt 重算一遍。

`,
  },
  {
    id: 'kv-cache',
    ep: '12',
    title: 'KV 缓存',
    summary: 'Prompt 算一次；解码只追加新的 K/V。',
    script: 'chapters/01-inference/ep12_kv_cache.py',
    partId: 'infer',
    estimatedMinutes: 15,
    demo: 'kv',
    markdown: `## 要点

朴素生成每一步都对**全部**历史做注意力——prompt 的 K/V 被反复重算。

**KV 缓存：** 第一次（**prefill**）存下每层的 K/V；之后每个新 token（**decode**）只算自己的 K/V 并**追加**，再对整段缓存做注意力。

结果（精确算术下）一样，回复越长越省。看本页互动条。

## 动手

\`\`\`bash
python chapters/01-inference/ep12_kv_cache.py
\`\`\`

预期：演示带缓存的逐步解码。

## 下一章

**RoPE 缩放**（\`?c=rope-scaling\`）：长 prompt 下 RoPE 仍好用。

`,
  },
  {
    id: 'rope-scaling',
    ep: '13',
    title: 'RoPE 缩放',
    summary: '拉长波长，让更长的 prompt 仍好用。',
    script: 'chapters/01-inference/ep13_rope_scaling.py',
    partId: 'infer',
    estimatedMinutes: 15,
    markdown: `## 要点

RoPE 的角度是按训练长度调的。上下文拉太长，相对位置可能「绕坏」。

**Llama 3.2 风格 RoPE 缩放**：保持很大的 \`rope_theta\`，用 factor / low / high 频段平滑拉长**长波长**（与 HF 公式兼容）。近距离行为仍熟悉，远距离被拉伸，长 prompt 才好用。

本章讲的是**调度表**，不是另造一种注意力。

## 动手

\`\`\`bash
python chapters/01-inference/ep13_rope_scaling.py
\`\`\`

预期：对比未缩放 / 已缩放的频率或形状打印。

## 下一章

**MoE**（\`?c=moe\`）：稀疏专家，而不是一个巨大 MLP。

`,
  },
  {
    id: 'moe',
    ep: '14',
    title: 'MoE',
    summary: '很多专家；每个 token 只跑 top-k。',
    script: 'chapters/01-inference/ep14_moe.py',
    partId: 'infer',
    estimatedMinutes: 15,
    markdown: `## 要点

Dense MLP 对每个 token 跑**同一套**前馈。**混合专家（MoE）**准备很多专家 MLP（这里是 SwiGLU 专家），再用**路由器**给每个 token 只选 **top-k** 个专家来算。

总参数可以很大，但每个 token 只付少数专家的算力——**稀疏计算**（Mixtral 思路）。本章是教学用 MoE 块，不是完整生产 Mixtral。

## 动手

\`\`\`bash
python chapters/01-inference/ep14_moe.py
\`\`\`

预期：一次小 MoE 前向的路由 / 形状。

## 下一章

**MLA**（\`?c=mla\`）：压缩解码时要缓存的东西。

`,
  },
  {
    id: 'mla',
    ep: '15',
    title: 'MLA',
    summary: '缓存一小段 latent，而不是完整 K/V。',
    script: 'chapters/01-inference/ep15_mla.py',
    partId: 'infer',
    estimatedMinutes: 18,
    markdown: `## 要点

GQA 已经减少了 KV 头。**MLA**（Multi-head Latent Attention，DeepSeek 思路）为了解码显存走得更远：缓存**低秩 latent**（外加一小段 RoPE 流），而不是每头完整 K/V，注意力需要时再展开。

教学目标：体会 latent KV **为什么**省内存——不是比特级复刻 DeepSeek 权重。

## 动手

\`\`\`bash
python chapters/01-inference/ep15_mla.py
\`\`\`

预期：紧凑前向，展示 latent 缓存形状。

## 下一章

**设备 · INT8 · 对话**（\`?c=device-quant-chat\`）：跑在哪、权重怎么变小、对话怎么排版。

`,
  },
  {
    id: 'device-quant-chat',
    ep: '16',
    title: '设备 · INT8 · 对话模板',
    summary: '选设备、压权重、排对话格式。',
    script: 'chapters/01-inference/ep16_device_quant_chat.py',
    partId: 'infer',
    estimatedMinutes: 15,
    demo: 'chat',
    markdown: `## 要点

围着算子核心的三块实用东西：

1. **设备** — 有 CUDA 用 CUDA，否则 Apple MPS，再否则 CPU（\`pick_device\`）  
2. **INT8（教学）** — 权重存成 int8 省内存；计算走简单反量化路径，方便学  
3. **对话模板** — 把 system/user/assistant 消息排成模型认识的字符串；没有 HF 模板时回退成可读纯文本（可试本页 demo）

## 动手

\`\`\`bash
python chapters/01-inference/ep16_device_quant_chat.py
\`\`\`

预期：选中的设备、量化说明、渲染后的对话串。

## 下一章

**连续批处理**（\`?c=continuous-batching\`）：多段短生成共享步进。

`,
  },
  {
    id: 'continuous-batching',
    ep: '17',
    title: '连续批处理',
    summary: '多个短任务共享一个步进循环。',
    script: 'chapters/01-inference/ep17_continuous_batching.py',
    partId: 'infer',
    estimatedMinutes: 15,
    demo: 'batch',
    markdown: `## 要点

朴素服务：每人一个生成循环——人一多 GPU 经常空转。

**连续批处理**（这里是教学调度器）：多个短解码任务放进**同一个**步进循环；谁先打满长度（或 EOS）就退役，必要时再进新任务。大家共享同一拍「时间步」。

这**不是**完整 paged attention / vLLM——只是调度形状。可试本页 demo。

## 动手

\`\`\`bash
python chapters/01-inference/ep17_continuous_batching.py
\`\`\`

预期：几个任务一起推进直到各自结束。

## 下一章

**预训练**（\`?c=pretrain\`）：一个极小的 next-token 训练循环。

`,
  },
  {
    id: 'pretrain',
    ep: '18',
    title: '预训练',
    summary: '玩具 next-token 训练——感受 API。',
    script: 'chapters/02-training/ep18_pretrain.py',
    partId: 'train',
    estimatedMinutes: 15,
    demo: 'pretrain',
    markdown: `## 要点

**预训练** = 在大量文本上学习猜下一个 token。损失通常是 logits 与真实下一 id 的交叉熵；优化器（这里是 **AdamW**）更新权重。

循环故意**极小**（玩具 batch、很少步）：感受 API、看 loss 动——不是真去训 1B 全网数据。本页可调学习率/步数，在浏览器里真跑一小段。

## 动手

\`\`\`bash
python chapters/02-training/ep18_pretrain.py
\`\`\`

预期：几步玩具 loss。本页 demo 用示意数字演同一思路。

## 下一章

**SFT · DPO**（\`?c=sft-dpo\`）：跟指令、跟偏好。

`,
  },
  {
    id: 'sft-dpo',
    ep: '19',
    title: 'SFT · DPO',
    summary: 'SFT 掩掉提示；DPO 跑一步偏好。',
    script: 'chapters/02-training/ep19_sft_dpo.py',
    partId: 'train',
    estimatedMinutes: 18,
    demo: 'train',
    markdown: `## 要点

预训练之后，还要会**跟对话**、**更偏好好回答**。

- **SFT**（监督微调）：用「提示→回答」样本训练，但**掩掉提示**，损失只打在回复 token 上。  
- **DPO**（直接偏好优化）：给定更好 / 更差回答，相对**冻结的 reference** 把模型推向更好那一侧——基础形式不需要单独奖励模型。我们只跑**教学用的一步**。

## 动手

\`\`\`bash
python chapters/02-training/ep19_sft_dpo.py
\`\`\`

预期：看到 SFT 掩码行为，以及一次 DPO 更新打印。

## 下一章

**前沿检查点**（\`?c=frontier-ckpt\`）：存取 MoE / MLA 形状的权重。

`,
  },
  {
    id: 'frontier-ckpt',
    ep: '20',
    title: '前沿检查点',
    summary: '存取 MoE/MLA；映射 Mixtral 专家 key。',
    script: 'chapters/02-training/ep20_frontier_ckpt.py',
    partId: 'frontier',
    estimatedMinutes: 15,
    markdown: `## 要点

前沿模块（MoE、MLA）和 dense 一样要会**存取**，不能静默形状错乱。

本章往返教学用 MoE/MLA 检查点，并演示 **Mixtral 风格**专家 key（\`w1/w2/w3\`）如何映射到我们的模块名——读真实开放权重时少猜一点。

## 动手

\`\`\`bash
python chapters/02-training/ep20_frontier_ckpt.py
\`\`\`

预期：存→载成功，并看到 key 映射演示。

## 下一章

**HF 对齐**（\`?c=hf-align\`）：有权重时证明 dense logits 能对上 Hugging Face。

`,
  },
  {
    id: 'hf-align',
    ep: '21',
    title: 'HF logits 对齐',
    summary: '短 prompt 上对照 HF 的 logits。',
    script: 'chapters/01-inference/ep21_hf_align.py',
    partId: 'frontier',
    estimatedMinutes: 20,
    markdown: `## 要点

信任但要验证：\`from_pretrained\` 之后，用**同一句短 prompt**分别跑 arche 与 Hugging Face Llama 3.2，比 **argmax** 和 **logits 最大绝对差**。

若没有门禁权重，本章干净退出（CI 仍有合成映射测试）。这是「dense 主干接线对不对」的关卡。

## 动手

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
\`\`\`

有权重时预期：一份很小的差异报告。

## 翻车

无权重 / 未登录 → 脚本干净跳过或给出 Fix；不等于 arche 逻辑坏了。

## 下一章

**分片 MoE/MLA**（\`?c=sharded-frontier\`）：大检查点在磁盘上怎么拆。

`,
  },
  {
    id: 'sharded-frontier',
    ep: '22',
    title: '分片 MoE / MLA',
    summary: 'index.json + 分片 → 装回 MoE/MLA。',
    script: 'chapters/02-training/ep22_sharded_frontier.py',
    partId: 'frontier',
    estimatedMinutes: 15,
    markdown: `## 要点

超大模型很少一个文件搞定。Hugging Face 风格常用 **\`index.json\`** + 许多 **\`.safetensors\` / \`.pt\` 分片**。

我们造一份 **合成** index + 分片（Mixtral 形 MoE、DeepSeek 形 MLA），再装回来做 \`allclose\`——看清打包/拆包，不必真下几十 GB。

## 动手

\`\`\`bash
python chapters/02-training/ep22_sharded_frontier.py
\`\`\`

预期：分片写入/读回，张量对齐。

## 下一章

**Paged KV · CUDA Graph**（\`?c=paged-cuda-graph\`）：服务端显存与捕获的思路。

`,
  },
  {
    id: 'paged-cuda-graph',
    ep: '23',
    title: 'Paged KV · CUDA Graph',
    summary: '块表 KV 池与固定 shape 的 CUDA graph。',
    script: 'chapters/01-inference/ep23_paged_cuda_graph.py',
    partId: 'frontier',
    estimatedMinutes: 18,
    markdown: `## 要点

两个服务端思路（教学体量）：

1. **Paged KV** — K/V 放进固定大小的**块** + **块表**，注意力时再 gather（vLLM 启发的形状）。避免「一条超长序列必须整段连续缓存」的故事。  
2. **CUDA Graph** — 有 CUDA 时**捕获**固定 shape 计算，减少启动开销；在 KV 写入安全变为原地之前，解码仍可 eager。

实验展示机制，不是可替换生产引擎。

## 动手

\`\`\`bash
python chapters/01-inference/ep23_paged_cuda_graph.py
\`\`\`

预期：分页 gather；仅在有 CUDA 时做 graph 捕获。

## 下一章

路径走完。可回看薄弱章，或有权重时跑 HF 对齐。

`,
  },
];
