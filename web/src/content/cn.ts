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

### 1. 为什么选这个标本

需要一个够小、开放、像生产的模型，端到端手写。

### 2. 「模型」在这里指什么

**Llama 3.2 风格**的 next-token 栈：分词 → 嵌入 → 块堆叠 → LM head。人们常叫它 **dense 解码器**——名字后面混熟即可。

\`\`\`mermaid
flowchart LR
  tok[分词] --> emb[嵌入]
  emb --> stack[堆叠]
  stack --> head[LM head]
\`\`\`

### 3–5. 为什么是 Llama 3.2

约 1B、权重可下、积木与很多生产 dense 一致（RMSNorm、RoPE、SwiGLU、GQA）。后面还有推理、小训练、前沿引物。

### 6. 本课怎么做

实验打印配置尺寸——还不是聊天。

## 动手

\`\`\`bash
python chapters/01-inference/ep01_overview.py
\`\`\`

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

### 1. 为什么要嵌入

token id 只是整数。网络要能做乘加，需要每个 id 对应一串数字——**向量**。

### 2. 做什么

**嵌入**是一张查找表，形状 \`(词表大小, 维度)\`：第 \`i\` 行是 id \`i\` 的向量。前向 = 按 id 取行。这里还不「理解」语义，只是符号 → 几何的第一层可学习映射。

\`\`\`mermaid
flowchart LR
  ids[token id] --> table[嵌入表]
  table --> vecs["向量 (B, T, dim)"]
\`\`\`

### 3. 你会听到的设计

| 选择 | 人话 | 今天 |
| --- | --- | --- |
| 嵌入上再加绝对位置表 | 每个位置学一个向量 | Llama 风格里不作主线 |
| **这里不加位置；RoPE 稍后** | 位置写在 Q/K 上 | **主流 dense** |
| 与 LM head 绑权 | 同一矩阵：id→向量、向量→词表分 | 小 Llama 常见 |

### 4. 业界实际

Llama 3.2 / Qwen 一类：只做查表；**RoPE** 管位置；常 **绑** 嵌入 ↔ LM head。

### 5. 怎样算对

- 形状 \`(B, T) → (B, T, dim)\`  
- 与分词器同一词表（第 \`i\` 行 ↔ id \`i\`）  
- 若后面假定 RoPE，就别在这里偷偷加回绝对 PE  

### 6. 本课怎么做

\`arche/\` 里的 \`TokenEmbedding\`。实验用很小假词表，免下载；形状与真模型一致。

## 动手

\`\`\`bash
python chapters/01-inference/ep03_embedding.py
\`\`\`

预期：ids / 输出形状，并提醒位置交给 RoPE。

## 下一章

**RMSNorm**（\`?c=rmsnorm\`）：进重层前先稳住尺度。

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

### 1. 为什么要归一化

层一深，数值容易炸或塌。归一化把尺度拉回可控。

### 2. RMSNorm 做什么

对每个 token 向量（最后一维）：

1. 算 RMS  
2. 除以 RMS（+ 很小的 \`ε\`）  
3. 乘可学习 \`γ\`  
4. **不减均值**（和 LayerNorm 不同）

我们用 **pre-norm**：注意力前一次、MLP 前一次。

$$\\mathrm{RMSNorm}(x)=x/\\sqrt{\\mathrm{mean}(x^2)+\\varepsilon}\\odot\\gamma$$

### 3. 你会听到的名字

| 名字 | 人话 | 今天 |
| --- | --- | --- |
| LayerNorm | 减均值再缩放 | 偏 GPT-2 老故事 |
| **RMSNorm** | 只按 RMS 缩放 | **Llama / Qwen 默认** |
| Pre-norm / post-norm | 子层前还是后归一化 | 现代栈：pre-norm |

### 4. 业界实际

Llama 3.2、Qwen dense、多数新开放解码器：RMSNorm + pre-norm。

### 5. 怎样算对

- 进出同形状  
- \`γ=1\` 时，归一化后 RMS ≈ 1  
- 稳住训练，但不等于整模型的全部故事  

### 6. 本课怎么做

\`arche/norms.py\` 的 \`RMSNorm\`。实验打印形状与每 token RMS。

## 动手

\`\`\`bash
python chapters/01-inference/ep04_rmsnorm.py
\`\`\`

## 下一章

**RoPE**（\`?c=rope\`）：把位置写进 Q/K。

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

### 1. 为什么要位置

没有位置，「先 A 后 B」和「先 B 后 A」对注意力长一样。

### 2. RoPE 做什么

**RoPE** 按**位置**旋转每个 Q/K。相对距离进点积——不必在嵌入上贴大位置表。

本页可拖互动图。代码：cos/sin 缓存、\`rotate_half\`，再 \`q' = q·cos + rotate(q)·sin\`。

### 3. 其他做法

| 做法 | 人话 | 今天 |
| --- | --- | --- |
| 绝对可学习 PE | 每个下标一个向量 | 偏老栈 |
| 正弦 PE | 固定 sin/cos | 原版 Transformer |
| **RoPE** | 旋转 Q/K | **主流解码器** |
| ALiBi 等 | 在分数上加距离偏置 | 部分长上下文路线 |

### 4. 业界实际

Llama 3 / 3.2：很大的 \`rope_theta\`；后面专章讲**波长缩放**。

### 5. 怎样算对

- 相对位置能进注意力  
- Q/K 形状不变  
- 更长上下文可能要缩放（后面专章）  

### 6. 本课怎么做

\`arche/rope.py\`。实验应用 RoPE 并核对形状。

## 动手

\`\`\`bash
python chapters/01-inference/ep05_rope.py
\`\`\`

## 下一章

**GQA**（\`?c=gqa\`）：更少的 KV 头。

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

### 1. 为什么要 GQA

完整 MHA：每个 Q 头自带 KV → 解码时 **KV 缓存**涨得快。

### 2. GQA 做什么

**分组查询注意力**：KV 头少于 Q 头，多组 Q **共用**一组 KV。仍是**因果**（\`t\` 只看 \`≤ t\`）。

流程：投影 → 拆头 → RoPE → 扩 KV → SDPA → 输出投影。见本页图。

### 3. 注意力布局家族

| 布局 | 人话 | 今天 |
| --- | --- | --- |
| MHA | Q 头数 = KV 头数 | 经典 |
| MQA | 全体 Q 共用一套 KV | 极致省 |
| **GQA** | 分组共用 | **Llama 3.2 1B：32Q / 8KV** |
| MLA | 把 KV 压成 latent | DeepSeek 风（后面章） |

### 4. 业界实际

Llama 3.2 1B 用 GQA；很多开放 7B–70B 为了解码显存也用 GQA。

### 5. 怎样算对

- \`n_rep = n_q / n_kv\` 正确  
- prefill / decode 的因果设置正确  
- 缓存比满 MHA 小，质量别塌光  

### 6. 本课怎么做

\`arche/attention.py\` 的 \`CausalSelfAttention\`。实验打印头数与形状。

## 动手

\`\`\`bash
python chapters/01-inference/ep06_gqa.py
\`\`\`

## 下一章

**SwiGLU**（\`?c=swiglu\`）：前馈那半边。

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

### 1. 为什么注意力后还要 MLP

注意力混合各位置；每个位置还要**各自**变换特征。

### 2. SwiGLU 做什么

$$\\mathrm{FFN}(x)=W_d\\big(\\mathrm{SiLU}(W_g x)\\odot W_u x\\big)$$

三条无偏置线性：**gate**、**up**、**down**。

### 3. 其他做法

| FFN | 人话 | 今天 |
| --- | --- | --- |
| GELU MLP（GPT-2 风） | 一层隐层 + GELU | 偏考古教学 |
| **SwiGLU** | 门控 SiLU × up | **Llama / Qwen dense** |
| SwiGLU 专家 MoE | 路由 top-k | Mixtral 风（后面） |

### 4. 业界实际

现代 dense 解码器：SwiGLU（或近亲门控）。

### 5. 怎样算对

- 进出 \`embed_dim\`；中间更宽  
- 三层无 bias（Llama 风格）  

### 6. 本课怎么做

\`arche/mlp.py\` 的 \`SwiGLU\`。

## 动手

\`\`\`bash
python chapters/01-inference/ep07_swiglu.py
\`\`\`

## 下一章

**Transformer 块**（\`?c=block\`）：拼成三明治。

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

### 1. 为什么要「块」

一个可重复单元，叠 \`N\` 次。

### 2. 长什么样

\`\`\`
x → RMSNorm → 注意力 → + 残差
  → RMSNorm → SwiGLU → + 残差
\`\`\`

进出同形状。注意力可带 KV cache。

### 3. 布局选择

| 选择 | 人话 | 今天 |
| --- | --- | --- |
| Post-norm | 残差后再归一化 | 偏老 |
| **Pre-norm** | 子层前归一化 | **现代默认** |
| 注意力∥MLP 并行 | 非 Llama 布局 | 不是本课主干 |

### 4. 业界实际

Llama 3.2 1B：**16** 个这样的块。

### 5. 怎样算对

- 残差路径形状不变  
- cache 别漏层  

### 6. 本课怎么做

\`arche/block.py\` 的 \`TransformerBlock\`。

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
    summary: '整条路：嵌入 → 多层块 → 词表分数。',
    script: 'chapters/01-inference/ep09_stack.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    markdown: `## 要点

### 1. 为什么要堆叠

一块还不是语言模型——多层 + 词表头才是。

### 2. 整条路

\`ids → 嵌入 → Block×N → 最终 RMSNorm → Linear(词表)\`

最后分数 = 下一 token 的 **logits**。常 **绑** LM head ↔ 嵌入。

### 3. 零件

| 零件 | 作用 |
| --- | --- |
| 嵌入 | id → 向量 |
| 块 | 上下文混合 |
| 最终 norm | 进 head 前稳住 |
| LM head | 向量 → 词表分 |

### 4. 业界实际

Llama 风格 dense 端到端（下一章前仍是随机权重）。

### 5. 怎样算对

- logits \`(B, T, vocab)\`  
- 绑权与声称的配置一致  

### 6. 本课怎么做

\`arche/model.py\` 的 \`Llama\`。

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
    summary: '把真 Llama 3.2 张量拷进我们的模块。',
    script: 'chapters/01-inference/ep10_load_weights.py',
    partId: 'trunk',
    estimatedMinutes: 20,
    markdown: `## 要点

### 1. 为什么要加载权重

图对了但权重随机 → 输出无意义。

### 2. 做什么

\`from_pretrained\` 搭好模块，再按 **key 映射**拷入 HF Llama 3.2 张量。

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

\`arche/load_weights.py\` + 本章脚本。

## 动手

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
\`\`\`

## 翻车

与分词器相同：许可、登录、镜像。

## 下一章

**文本生成**（\`?c=generate\`）。

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

### 1. 为什么要循环

一次前向只给出「下一个」的分数，不是整段回复。

### 2. 生成做什么

1. 取**最后位置** logits  
2. 可选 temperature / top-k  
3. 贪心或采样  
4. 拼上；EOS 或长度则停  

KV 缓存（下一章）让第 1 步在 prefill 后变便宜。

### 3. 旋钮

| 旋钮 | 效果 |
| --- | --- |
| temperature → 0 | 更贪心/更尖 |
| temperature 升高 | 更随机 |
| top-k | 只保留前 k 个 |

### 4. 业界实际

所有聊天栈同一类循环；采样细节各家不同。

### 5. 怎样算对

- 能干净停下  
- 有 EOS 时尊重它  
- 有真权重时文字连贯（不是本章唯一目标）  

### 6. 本课怎么做

\`arche/generate.py\` 的 \`generate\`。

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
    summary: 'Prompt 算一次；解码只追加新的 K/V。',
    script: 'chapters/01-inference/ep12_kv_cache.py',
    partId: 'infer',
    estimatedMinutes: 15,
    demo: 'kv',
    markdown: `## 要点

### 1. 为什么要缓存

朴素解码每步都对整段 prompt 重算 K/V。

### 2. KV 缓存做什么

**Prefill** 一次 → 存每层 K/V。每次 **decode** 只追加一步再对缓存做注意力。见本页条带图。

### 3. 相关想法

| 想法 | 作用 |
| --- | --- |
| 连续缓存 | 沿时间 \`cat\`（本课教学缓存） |
| Paged KV | 块表 + gather（后面引物） |
| GQA/MLA | 缩小要存的东西 |

### 4. 业界实际

生产解码器都有 KV 缓存；分页/布局各引擎不同。

### 5. 怎样算对

- 与全量重算结果一致（差在数值）  
- 算量随新 token 涨，而不是每步整段 prompt  

### 6. 本课怎么做

\`arche/cache.py\` 的 \`KVCache\`。

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
    summary: '拉长波长，让更长的 prompt 仍好用。',
    script: 'chapters/01-inference/ep13_rope_scaling.py',
    partId: 'infer',
    estimatedMinutes: 15,
    markdown: `## 要点

### 1. 为什么要缩放 RoPE

角度按训练长度调的；上下文拉太长可能「绕坏」。

### 2. Llama 风格缩放做什么

保持大 \`rope_theta\`；按 factor/low/high **平滑拉长波长**（与 HF 兼容）。

### 3. 版图

| 方法 | 人话 |
| --- | --- |
| NTK / YaRN / … | 其他长上下文 RoPE 配方 |
| **Llama 3 波长平滑** | 本课对照的 |
| 更长数据直接训 | 不只靠公式 |

### 4. 业界实际

Llama 3.2 官方配置里的长上下文调度。

### 5. 怎样算对

- 近距离仍熟悉  
- 远距离被拉伸而不是乱绕  

### 6. 本课怎么做

\`arche/rope.py\` 的缩放辅助 + 本章脚本。

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
    summary: '很多专家；每个 token 只跑 top-k。',
    script: 'chapters/01-inference/ep14_moe.py',
    partId: 'infer',
    estimatedMinutes: 15,
    markdown: `## 要点

### 1. 为什么要 MoE

每个 token 都跑同一套 dense MLP 简单，但容量要人人付算力。

### 2. MoE 做什么

很多专家 MLP（这里是 SwiGLU）+ **路由器**：每 token 只跑 **top-k** → 稀疏计算。

### 3. 版图

| 想法 | 备注 |
| --- | --- |
| Dense FFN | 始终全开 |
| **MoE（Mixtral 风）** | top-k 专家 |
| 共享专家等变体 | 产品各异 |

### 4. 业界实际

本课是教学 MoE 块——不是完整 Mixtral 生产栈。

### 5. 怎样算对

- 路由到合法专家  
- 负载别极端倾斜（真训练很在意；这里只尝 API）  

### 6. 本课怎么做

\`arche/moe.py\` 引物。

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
    summary: '缓存一小段 latent，而不是完整 K/V。',
    script: 'chapters/01-inference/ep15_mla.py',
    partId: 'infer',
    estimatedMinutes: 18,
    markdown: `## 要点

### 1. 为什么要 MLA

GQA 已减 KV 头；解码显存仍可能占大头。

### 2. MLA 做什么

缓存**低秩 latent**（+ 小段 RoPE 流），而不是每头完整 K/V；需要时再展开。

### 3. 版图

| 想法 | 作用 |
| --- | --- |
| GQA | 更少 KV 头 |
| **MLA** | latent KV（DeepSeek 思路） |
| Paged KV | 显存怎么切块 |

### 4. 业界实际

教学形状——不是比特级 DeepSeek 权重。

### 5. 怎样算对

- latent 缓存比满 KV 小  
- 前向能跑通  

### 6. 本课怎么做

\`arche/mla.py\` 引物。

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
    summary: '选设备、压权重、排对话格式。',
    script: 'chapters/01-inference/ep16_device_quant_chat.py',
    partId: 'infer',
    estimatedMinutes: 15,
    demo: 'chat',
    markdown: `## 要点

### 1. 为什么要这三块

只有算子不够跑聊天：设备、体积、消息格式都要。

### 2. 覆盖什么

1. **设备** — CUDA → MPS → CPU  
2. **INT8（教学）** — 权重更小；简单反量化  
3. **对话模板** — 角色 → 模型要的字符串（HF 或纯文本回退；可试 demo）

### 3. 版图

| 主题 | 生产备注 |
| --- | --- |
| 量化 | INT8/INT4/FP8… 配方很多 |
| 模板 | 模型相关；别随便发明 |
| 设备 | 同一图，不同后端 |

### 4–6. 本课怎么做

教学辅助 + 脚本——不是完整量化产品。

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
    summary: '多个短任务共享一个步进循环。',
    script: 'chapters/01-inference/ep17_continuous_batching.py',
    partId: 'infer',
    estimatedMinutes: 15,
    demo: 'batch',
    markdown: `## 要点

### 1. 为什么

每人一个生成循环 → 结束节奏不同时 GPU 常空转。

### 2. 这里的连续批是什么

多个短解码任务共享一个步进循环；结束就退役。**不是**完整 paged / vLLM。可试 demo。

### 3. 版图

| 层 | 作用 |
| --- | --- |
| 调度 | 这一步谁还在跑 |
| 分页 / 内核 | 显存与算力怎么跑（后面引物） |

### 4–6. 本课怎么做

教学调度形状 + 本章脚本。

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
    summary: '玩具 next-token 训练——感受 API。',
    script: 'chapters/02-training/ep18_pretrain.py',
    partId: 'train',
    estimatedMinutes: 15,
    demo: 'pretrain',
    markdown: `## 要点

### 1. 为什么要预训练

先在大量文本上学会猜下一个 token，再谈跟对话。

### 2. 我们跑什么

玩具 AdamW：logits 对下一 id、交叉熵、一步更新。**本页**可拖学习率/步数，在浏览器里真跑一小段。

### 3. 版图

| 规模 | 含义 |
| --- | --- |
| 本实验 | 感受 API |
| 真预训练 | 海量数据、多卡、很久 |

### 4–6. 本课怎么做

\`arche/train/pretrain.py\` + 页内试跑。不是全网训练。

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
    summary: 'SFT 掩掉提示；DPO 跑一步偏好。',
    script: 'chapters/02-training/ep19_sft_dpo.py',
    partId: 'train',
    estimatedMinutes: 18,
    demo: 'train',
    markdown: `## 要点

### 1. 为什么

预训练后仍要会跟对话、更偏好好回答。

### 2. 做什么

- **SFT：** 提示→回答；**掩掉提示**；损失只打在回复  
- **DPO：** 相对冻结 reference，偏好 chosen（教学一步）

### 3. 版图

RLHF / RLAIF / DPO / … 配方很多；这里尝两个 API。

### 4–6. 本课怎么做

\`arche/train/sft.py\`（及 DPO）+ 本章脚本。

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
    summary: '存取 MoE/MLA；映射 Mixtral 专家 key。',
    script: 'chapters/02-training/ep20_frontier_ckpt.py',
    partId: 'frontier',
    estimatedMinutes: 15,
    markdown: `## 要点

### 1. 为什么

MoE/MLA 也要能存取，不能静默形状错乱。

### 2. 做什么

往返教学检查点；把 Mixtral 风 \`w1/w2/w3\` 映射到我们的模块名。

### 3–6. 本课怎么做

读开放 MoE 检查点的引物——不是完整训练器。

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
    summary: '短 prompt 上对照 HF 的 logits。',
    script: 'chapters/01-inference/ep21_hf_align.py',
    partId: 'frontier',
    estimatedMinutes: 20,
    markdown: `## 要点

### 1. 为什么

信任 dense 主干接线，但要验证。

### 2. 做什么

同一短 prompt 跑 arche 与 HF Llama 3.2；比 argmax 与 logits 最大绝对差。无权重 → 干净退出。

### 3–6. 本课怎么做

本章脚本 + CI 合成映射测试。

## 动手

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
\`\`\`

## 翻车

无权重/未登录 → 跳过或提示，不等于逻辑坏了。

## 下一章

**分片 MoE/MLA**（\`?c=sharded-frontier\`）。

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

### 1. 为什么

超大模型用 \`index.json\` + 许多分片文件。

### 2. 做什么

合成 index + 分片（Mixtral 形 MoE / DeepSeek 形 MLA）；装回并 \`allclose\`。

### 3–6. 本课怎么做

打包/拆包形状，不必真下几十 GB。

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
    summary: '块表 KV 池与固定 shape 的 CUDA graph。',
    script: 'chapters/01-inference/ep23_paged_cuda_graph.py',
    partId: 'frontier',
    estimatedMinutes: 18,
    markdown: `## 要点

### 1. 为什么

服务端需要比玩具循环更聪明的显存与更低的启动开销。

### 2. 做什么

1. **Paged KV** — 定长块 + 块表 + gather  
2. **CUDA Graph** — 有 CUDA 时捕获固定 shape 计算  

教学机制——不是可替换的 vLLM。

### 3–6. 本课怎么做

\`arche/paged_kv.py\`（及 graph）+ 本章脚本。

## 动手

\`\`\`bash
python chapters/01-inference/ep23_paged_cuda_graph.py
\`\`\`

## 下一章

路径走完——可回看薄弱章，或有权重时跑 HF 对齐。

`,
  },
];
