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

本课是**从零训到能对接业界前沿**：先选一个与生产同构的标本，再逐块手写、再谈加速与训练。标本错了，后面所有「业界怎么做」都会对不上。

### 1. 有什么作用

定标本：**Llama 3.2 风格** next-token 栈——分词 → 嵌入 → 块×N → LM head（**dense 解码器**）。后续章都挂在这条主干上。

\`\`\`mermaid
flowchart LR
  tok[分词表] --> emb[嵌入表]
  emb --> stack[块堆叠]
  stack --> head[LM head]
\`\`\`

### 2. 从零开始：怎么「生成」这条路线

| 步骤 | 做什么 |
| --- | --- |
| **1. 选开放、可下、够小的生产同构模型** | 本课：Llama 3.2 ~1B |
| **2. 冻结其分词 / 记下 V、D、层数、头数** | 配置即契约 |
| **3. 按同积木手写模块** | RMSNorm、RoPE、GQA、SwiGLU… |
| **4. 先随机权重跑通形状，再加载/自训权重** | 图对 → 数对 |
| **5. 再叠推理与对齐** | KV、批处理、SFT… |

### 3. 怎么样算选对了

| 类型 | 标准 |
| --- | --- |
| **硬** | 配置自洽；积木与声称的开源配方一致 |
| **硬** | 能支撑「从零」叙事：表可训、网可叠、权可装 |
| **软** | 社区文档全、权重可申请 |
| **软** | 后续前沿引物（MoE/MLA）有地方挂 |

### 4. 如何改进 · 对整模影响

换标本（如改 Qwen）→ 分词/嵌入/块配置全换，课程映射要重做。加深/加宽 → 全面重训。本课保持 1B 标本，用引物看更大模型。

### 5. 业界 · 开源怎么做

Llama / Qwen / DeepSeek 等均以「公开配置 + 检查点 + 分词」发布；教学与复现都先对齐一份配方。vLLM 等引擎假设你已有这类 dense/MoE 检查点。

### 6. 本课

\`\`\`bash
python chapters/01-inference/ep01_overview.py
\`\`\`

打印配置尺寸，还不是聊天。

## 动手

\`\`\`bash
python chapters/01-inference/ep01_overview.py
\`\`\`

## 下一章

**分词器**（\`?c=tokenizer\`）：从零生成分词表。

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

本课路线是**从零搭到能对接业界前沿**：分词表不是「用现成的就完了」，而是你自己训模型时要**选定/训出**的第一张表；后面嵌入、预训练、推理成本都绑在它上面。

### 1. 分词表有什么作用

分词表 = **切分规则 + 词表（片段 ↔ id）**。

1. **切** — 文字 → token  
2. **编号** — token → 固定整数 id  

进模型前编码，出模型后用**同一张表**解码。BOS/EOS/工具调用等特殊符也是表里的 id。

\`\`\`mermaid
flowchart LR
  text[文字] --> cut[切]
  cut --> ids[token id]
  ids --> emb[嵌入第 i 行]
  emb --> stack[后面整网]
  stack --> ids2[新 id]
  ids2 --> back[解码]
\`\`\`

没有这张表，嵌入没有合法行号，预训练目标（下一 token）也没有词表可对。

### 2. 从零开始：分词表怎么生成

开放 decoder 主流是 **字节级 BPE**（思路与 GPT/Llama 同族）。从零时顺序大致是：

| 步骤 | 做什么 | 决定了什么 |
| --- | --- | --- |
| **1. 定语料配比** | 代码 / 中英 / 数学 / 网页…各多少 | 哪些片段会进词表、哪种语言更「省 token」 |
| **2. 预分词正则** | 空格、数字、标点、CJK 怎么先切开 | 结构是否被切坏；中文体验差不差 |
| **3. 底层符号** | UTF-8 **256 字节**起（或等价） | 任意文本可表示，可逆 |
| **4. BPE 合并** | 反复合并语料上最常见的相邻 pair，直到目标词表大小 | 词表内容与压缩能力 |
| **5. 特殊符** | \`<|begin_of_text|>\`、EOS、pad、工具标记… | 对话/代理协议 |
| **6. 冻结并版本化** | 写出 \`tokenizer.json\` / merges，打版本号 | 此后嵌入行数、检查点都按此 id 空间 |
| **7. 再训嵌入与整网** | 词表定了才能定 \`vocab_size\`、建嵌入、开预训练 | **换表 ≈ 重做嵌入与大量训练** |

本课提供两层：

- **教学**：\`arche/teach_bpe.py\` 在小语料上真训一张表（看生成过程）  
- **标本**：官方 Llama 3.2 表（看业界成品长什么样、如何与权重配对）

### 3. 怎么样的分词表算好

先分 **硬约束** 与 **优化目标**。

**硬约束（不满足就不能上前沿管线）**

| 标准 | 人话 |
| --- | --- |
| **可逆** | 正常文本 encode→decode 能还原（允许官方声明的规范化） |
| **覆盖** | 任意字节序列有表示（字节级 BPE 的底线） |
| **稳定** | 同一文本多次编码一致 |
| **与训练一致** | 预训练/微调/推理用同一张表；id 空间不偷偷改 |

**优化目标（决定贵不贵、偏不偏科、好不好训）**

| 标准 | 人话 | 对整模的影响 |
| --- | --- | --- |
| **压缩 / fertility** | 同样文字更少 token 更好；fertility = 平均一词几片，越大越碎 | 上下文更长、训练/推理更省；反之浪费窗口与算力 |
| **语言公平** | 中英等同义句 token 数别差太远 | 不公平 → 某语言有效上下文短、贵、难学好 |
| **领域适配** | 代码、公式、CJK 不被切得稀碎 | 垂直能力上限 |
| **结构保全** | 缩进、JSON、URL、数字 | 工具调用与代码能力 |
| **词表大小** | 常见 32K–200K+ | 越大通常越省 token，但**嵌入参数 ∝ V×D**，softmax 更贵 |
| **特殊符设计** | 对话/工具边界清晰 | 对齐与代理行为 |
| **下游指标** | 同架构下预训练 loss、评测 | 最终裁判之一 |

「好」不是单一分数：常在 **压缩、公平、词表大小、下游** 之间权衡。

### 4. 如何改进 · 对大模型整体有什么影响

| 改进杠杆 | 怎么做 | 整模影响 |
| --- | --- | --- |
| **换/重训语料配比** | 加代码、加中文、加数学 | 对应域 fertility↓，能力更容易涨；别的域可能变差 |
| **调预分词正则** | 数字整段、CJK 更友好 | 结构与公平；常比「盲目加大词表」更划算 |
| **加大词表** | 更多 merges / 更大 V | 序列变短；嵌入与 LM head 变胖；显存↑ |
| **加特殊符** | 表尾追加，旧 id 不动 | 嵌入要**扩行并继续训**；旧检查点要迁移 |
| **换算法族** | BPE ↔ Unigram 等 | **整张表作废**，必须重建嵌入并重训 |
| **继承并扩展** | 在旧表尾加域内高频片 | 迁移成本低于推倒重来 |

从零对接前沿时的实务顺序：

1. 先定**数据与目标语言/域** → 再定词表  
2. 词表冻结后再定 \`vocab_size\`、嵌入、开始预训练  
3. 中途换表成本极高；扩特殊符相对常见，也要改嵌入

### 5. 业界怎么做 · 知名开源怎么做

大家多在 **字节 BPE / tiktoken 系**里抠语料与正则，而不是有的按词、有的按字分家。

| 模型 / 系 | 引擎与规模 | 做法要点 |
| --- | --- | --- |
| **GPT-4 / 3.5** | tiktoken BPE \`cl100k\` ~100K | 切法库开源；表与模型绑定 |
| **GPT-4o** | \`o200k\` ~200K | 换代扩表，多语言更省；**新表 ≠ 旧嵌入** |
| **Llama 3 / 3.2** | 字节 BPE，128K+ | **本课标本**；官方 \`tokenizer\` 与权重出厂一对 |
| **Qwen** | BPE ~150K 级 | 自训表；强调多语言与工具特殊符 |
| **DeepSeek-V3** | 字节 BPE ~128K | 调预分词/语料做多语言压缩 |
| **Kimi K2** | tiktoken 风 ~163K | 汉字更友好的正则 |
| **老 BERT 系** | WordPiece | 编码器时代；与 decoder BPE **表不互通** |
| **部分多语言** | Unigram + SentencePiece | 仍见；和 BPE 规则不同，不能混装 |

业界共识（对人话）：

- **训模型的人**：自己训或继承一张表，写进配方，永不和别家表混用  
- **用模型的人**：只加载该检查点自带的分词器  
- **改进**：新一代常「更大词表 + 更好的多语言/代码正则」，而不是换回按词切

### 6. 代码（本课）

教学训表（从零看生成，无需 HF）：

\`\`\`bash
python chapters/01-inference/ep02_teach_bpe.py
\`\`\`

核心在 \`arche/teach_bpe.py\`：256 字节起 → 合并高频 pair → encode/decode → 打印 fertility。

官方标本表：

\`\`\`python
from arche.tokenizer import load_tokenizer
tok = load_tokenizer()  # meta-llama/Llama-3.2-1B
ids = tok.encode("The cat sat on the mat.")
assert tok.decode(ids)  # 可逆检查
\`\`\`

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep02_tokenizer.py
\`\`\`

## 动手

\`\`\`bash
python chapters/01-inference/ep02_teach_bpe.py
huggingface-cli login
python chapters/01-inference/ep02_tokenizer.py
\`\`\`

大陆网络可：\`export HF_ENDPOINT=https://hf-mirror.com\`

## 翻车

官方表：许可 / login / \`401\` / 镜像。教学 BPE 不需网络。

## 下一章

**嵌入层**（\`?c=embedding\`）：词表冻结后，嵌入表如何从零生成、怎样算好、如何改进及开源怎么做。

`,
  },
  {
    id: 'embedding',
    ep: '03',
    title: '嵌入层',
    summary: 'id 只是编号；嵌入按行取出一串小数，后面才能算。',
    script: 'chapters/01-inference/ep03_embedding.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    demo: 'embedding',
    markdown: `## 要点

分词表冻结之后，从零训模型的下一步是**嵌入表**：它把每个 id 变成可训练的一排小数。表不好，后面所有层都在噪声上学习；表与分词不对齐，整条前沿管线直接废。

### 1. 嵌入表有什么作用

矩阵 \`(V, D)\`：**第 \`i\` 行 = id \`i\` 的小数行**（\`V\` = 词表大小，\`D\` = 模型维度）。

| 作用 | 人话 |
| --- | --- |
| **① 接口** | 离散 id → 可乘加的张量 |
| **② 可学习身份** | 每个 token 的「初始样子」；预训练主要写手 |
| **③ 常绑 LM head** | 小模型：进取行、出打分共用，省参数、常更稳 |

推理 = 按 id **gather**（本页上方可试）。不是词典释义。

\`\`\`mermaid
flowchart LR
  tok[分词表冻结] --> shape["定 V = vocab_size"]
  shape --> init[随机初始化 V×D]
  init --> pretrain[预训练改行]
  pretrain --> ckpt[写入检查点]
\`\`\`

### 2. 从零开始：嵌入表怎么生成

**不是**手填，也**不是**再跑 BPE。BPE 决定「谁是几号」；嵌入决定「几号对应哪串数」。

| 步骤 | 做什么 | 说明 |
| --- | --- | --- |
| **1. 锁定 V** | \`V = 分词词表大小\`（含特殊符） | 与上一章出厂一致 |
| **2. 选定 D** | 如 Llama 3.2 1B 用 **2048** | 架构超参；与隐藏层宽度一致 |
| **3. 建模块** | \`nn.Embedding(V, D)\` 或等价 | 随机初始化 |
| **4. 决定是否绑权** | 小模型常 \`embed ↔ lm_head\` 共享 | 大模型有时解开 |
| **5. 预训练** | next-token CE；反传更新**出现过的行** | 嵌入与整网一起长 |
| **6. 存盘** | \`embed_tokens.weight\` 等进检查点 | 推理只加载、取行 |
| **7. 变更时** | 扩特殊符 → **扩行再训**；换分词表 → **整表作废重来** | 成本差几个数量级 |

初始化只是「空柜壁」；**数是训练写进去的**。本课假表演示取行与一步更新；真表来自官方权重或你自己的预训练。

### 3. 怎么样的嵌入表算好

**硬约束**

| 标准 | 人话 |
| --- | --- |
| **形状** | \`(V, D)\` 与配置、分词一致；\`T\` 个 id → \`T\` 行 |
| **配对** | 行 \`i\` ↔ 分词 id \`i\`（跨模型绝不混表） |
| **数值健康** | 无 NaN/Inf；尺度可被 RMSNorm 等稳住 |
| **契约** | 本课栈位置用 RoPE；嵌入不加绝对 PE（除非整栈改） |

**优化目标（很少单独打「嵌入分」）**

| 标准 | 人话 | 对整模的影响 |
| --- | --- | --- |
| **与预训练目标合拍** | 常用 token 行更新充分 | 底模语言能力 |
| **维度够用** | D 太小欠表达；太大费参、难吃数据 | 容量与稳定性 |
| **稀有 token** | 长尾 id 少更新 | 专有名词/罕见字弱 |
| **绑权是否合适** | 小模型绑常更省 | 参数量与输出校准 |
| **下游 / loss** | 同架构对比 | **主裁判** |

和分词不同：fertility 可离线测；嵌入好坏主要看**同设定下整模**是否更好。

### 4. 如何改进 · 对大模型整体有什么影响

| 改进杠杆 | 怎么做 | 整模影响 |
| --- | --- | --- |
| **更好的预训练** | 数据、步数、稳定性（norm、lr…） | 主路径：嵌入随整网变好 |
| **调 D** | 改架构宽度 | 全局容量；需重训 |
| **绑 / 解绑 LM head** | tie 或拆开再训 | 小模型常 tie；大模型看消融 |
| **扩词表** | 表尾加行，新行初始化（随机/均值/子词组合等）再继续训 | 支持新特殊符/域；旧行可保留 |
| **换分词** | 新 V | **必须重建嵌入并重做预训练**（或昂贵的迁移） |
| **只换嵌入初始化** | 如用旧模型行做热启动 | 有时加快收敛；错误配对会毁模型 |
| **冻结嵌入微调** | 少数据时冻住 | 减过拟合；表达上限也可能降 |

从零对接前沿的顺序：

1. 分词冻结 → 2. 定 \`V,D\` 与是否 tie → 3. 随机嵌入 + 整网预训练 → 4. 需要协议符再**扩行**继续训  

**不要**把 Llama 嵌入塞进 Qwen 分词，或反过来。

### 5. 业界怎么做 · 知名开源怎么做

| 模型 / 系 | 嵌入怎么做 | 备注 |
| --- | --- | --- |
| **Llama 3 / 3.2** | \`nn.Embedding\` 风格查表；小规格常 **tie** LM head；**无**嵌入绝对 PE | **本课标本**；RoPE 管位置 |
| **Qwen** | 查表 + RoPE；自训词表决定 V | 与自有分词严格配对 |
| **DeepSeek** | 查表；前沿侧重点在 MLA/MoE 等，嵌入仍是 id→行 | 分词多语言友好则嵌入行更「值」 |
| **GPT 系（公开信息）** | 大词表 → 更大嵌入矩阵；换代扩表则嵌入重做 | \`o200k\` 不是在 \`cl100k\` 嵌入上打补丁那么简单 |
| **BERT 一代** | 常加 **token + segment + 绝对位置** 三路嵌入 | 与现代 decoder「嵌入纯查表 + RoPE」不同 |
| **MoE 大模型** | 嵌入仍多是共享查表；稀疏在 FFN 专家 | 嵌入不是 MoE 的主戏，但 V 仍贵 |

业界共识：

- 嵌入 = **可学习查表**，跟分词 **出厂一对**  
- 改进嵌入 ≈ **改进预训练与词表**，很少单独「精修嵌入矩阵」当产品卖点  
- 开放权重里嵌入就在 safetensors 里；你从零训就把训练结果写进去

### 6. 代码（本课）

\`\`\`python
# arche/embedding.py — 建表与取行
self.weight = nn.Embedding(vocab_size, embed_dim)
return self.weight(token_ids)
\`\`\`

\`\`\`python
# 从零：随机表 → 一步更新（教学）
emb = TokenEmbedding(V, D)
loss = F.cross_entropy(head(emb(ctx)[:, -1]), target)
loss.backward(); opt.step()  # 出现过的行被改写
\`\`\`

\`\`\`bash
python chapters/01-inference/ep03_embedding.py
\`\`\`

预期：形状、取行、一步后某行的平均 |Δ|、有限性检查。真 Llama：\`V≈128256\`，\`D=2048\`（1B），在官方检查点中。

## 动手

\`\`\`bash
python chapters/01-inference/ep03_embedding.py
\`\`\`

## 下一章

**RMSNorm**（\`?c=rmsnorm\`）：嵌入给出的小数进深网前，如何稳住尺度（从零训时的关键）。

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

从零训深网时，没有归一化，数值很容易炸或塌——嵌入之后第一道「稳住尺子」就是它。

### 1. 有什么作用

**RMSNorm**：对每个 token 向量算 RMS，除掉它，再乘可学习 \`γ\`；**不减均值**（异于 LayerNorm）。Llama 风格 **pre-norm**：注意力前、MLP 前各一次。

$$\\\\mathrm{RMSNorm}(x)=x/\\\\sqrt{\\\\mathrm{mean}(x^2)+\\\\varepsilon}\\\\odot\\\\gamma$$

### 2. 从零开始：怎么生成 / 接到哪

| 步骤 | 做什么 |
| --- | --- |
| **1** | 每个需要稳住的入口建一个 \`RMSNorm(D)\`，\`γ\` 长度 = \`D\` |
| **2** | \`γ\` 常初始化为 1 |
| **3** | 按 pre-norm 接到 Attention / MLP 前 |
| **4** | 与整网一起训；推理原样前向 |

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 进出同形状；\`ε\` 防除零；位置符合配方（pre vs post） |
| **软** | \`γ=1\` 时 RMS≈1（公式体检）；训练少炸、少 NaN |
| **软** | 最终看整模 loss——Norm 很少单独打分 |

### 4. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| LayerNorm→RMSNorm | 少算均值，Llama/Qwen 默认；换了要和权重匹配 |
| pre-norm vs post-norm | 现代深网几乎 pre；post 更难训深 |
| 学 \`γ\` / 冻 \`γ\` | 微调策略；乱改易伤稳定性 |
| 与 lr/初始化搭配 | 从零训能否收敛的关键 |

### 5. 业界 · 开源

| 系 | 做法 |
| --- | --- |
| **Llama 3/3.2、Qwen** | RMSNorm + pre-norm |
| **GPT-2 教学老栈** | LayerNorm + 常 post-norm |
| **DeepSeek 等** | 主干仍 RMS/pre 一族（细节看配方） |

### 6. 本课

\`arche/norms.py\`：

\`\`\`bash
python chapters/01-inference/ep04_rmsnorm.py
\`\`\`

## 动手

\`\`\`bash
python chapters/01-inference/ep04_rmsnorm.py
\`\`\`

## 下一章

**RoPE**（\`?c=rope\`）：从零把位置写进 Q/K。

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

从零做 decoder：嵌入只回答「是谁」；**位置**要用 RoPE（或别的 PE）写进注意力，否则顺序无关。

### 1. 有什么作用

**RoPE** 按位置旋转 Q/K，相对距离进入点积——**不必**在嵌入上加绝对位置表。

### 2. 从零开始：怎么生成

| 步骤 | 做什么 |
| --- | --- |
| **1** | 选 \`rope_theta\`（Llama 3 很大） |
| **2** | 预计算各位置 cos/sin 缓存 |
| **3** | 对 Q/K：\`rotate_half\` 后与 cos/sin 组合 |
| **4** | 与注意力一起训；长上下文再考虑缩放（后章） |

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | Q/K 形状不变；相对位置改变则分数变；与检查点 \`theta\`/缩放一致 |
| **软** | 训练长度内稳定；超长靠缩放或继续训 |

### 4. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 更大 \`theta\` / 缩放 | 长文能力；错配置毁长上下文 |
| 绝对 PE → RoPE | 现代默认；换了要整栈一致 |
| ALiBi 等 | 另一路线；与 RoPE 权重不混用 |

### 5. 业界 · 开源

| 系 | 做法 |
| --- | --- |
| **Llama 3/3.2、Qwen、DeepSeek** | RoPE（+ 各自长文配方） |
| **原版 Transformer** | 正弦 PE |
| **部分长文工作** | ALiBi / YaRN / NTK 等 |

本页可拖互动图。

### 6. 本课

\`\`\`bash
python chapters/01-inference/ep05_rope.py
\`\`\`

## 动手

\`\`\`bash
python chapters/01-inference/ep05_rope.py
\`\`\`

## 下一章

**GQA**（\`?c=gqa\`）：从零定头数，省 KV 缓存。

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

从零定架构时，注意力头布局直接决定**解码显存**与质量上限。

### 1. 有什么作用

**GQA**：Q 头多于 KV 头，多组 Q **共用**一组 KV；仍因果。流程：投影→拆头→RoPE→扩 KV→SDPA→输出投影。

### 2. 从零开始：怎么生成

| 步骤 | 做什么 |
| --- | --- |
| **1** | 选 \`n_q\`、\`n_kv\`（\`n_q % n_kv == 0\`） |
| **2** | 实现广播/重复 KV 再与 Q 注意力 |
| **3** | 与整网训；推理接 KV 缓存按 \`n_kv\` 存 |

Llama 3.2 1B 标本：**32 Q / 8 KV**。

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 头数整除；因果正确；与权重头数一致 |
| **软** | 缓存明显小于满 MHA；质量相对 MHA 可接受 |

### 4. 如何改进 · 对整模影响

| 布局 | 影响 |
| --- | --- |
| MHA | 质量上限高，KV 最贵 |
| **GQA** | 折中——开放 7B–70B 常用 |
| MQA | 最省，质量压力大 |
| MLA | 进一步压缓存（后章） |

改头数 = 改架构 = **重训**；不能把 MHA 权重塞进 GQA 形状。

### 5. 业界 · 开源

| 系 | 做法 |
| --- | --- |
| **Llama 3/3.2** | GQA |
| **许多 Qwen/同类** | GQA |
| **DeepSeek-V3** | MLA 路线（不同压缩） |
| **经典论文/老** | 满 MHA |

### 6. 本课

\`\`\`bash
python chapters/01-inference/ep06_gqa.py
\`\`\`

## 动手

\`\`\`bash
python chapters/01-inference/ep06_gqa.py
\`\`\`

## 下一章

**SwiGLU**（\`?c=swiglu\`）：从零接 FFN。

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

从零搭块：注意力混合位置；**FFN** 在每个位置做非线性——现代 dense 默认 SwiGLU。

### 1. 有什么作用

$$\\\\mathrm{FFN}(x)=W_d(\\\\mathrm{SiLU}(W_g x)\\\\odot W_u x)$$

三条无 bias 线性：gate / up / down。

### 2. 从零开始：怎么生成

| 步骤 | 做什么 |
| --- | --- |
| **1** | 定 \`ffn_dim\`（通常 > \`D\`） |
| **2** | 建三套权重，无 bias |
| **3** | 接在块内第二段（pre-norm 后） |
| **4** | 整网训；MoE 时每个专家往往是一个 SwiGLU |

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 进出 \`D\`；无 bias（Llama 契约）；形状与权重一致 |
| **软** | 相对 GELU MLP 的收益看整模 |

### 4. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 加宽 \`ffn_dim\` | 容量↑、算力↑ |
| GELU MLP→SwiGLU | 与 Llama/Qwen 对齐；要重训 |
| 改成 MoE 专家 | 参数总量↑、单 token 稀疏算 |

### 5. 业界 · 开源

| 系 | 做法 |
| --- | --- |
| **Llama / Qwen dense** | SwiGLU |
| **Mixtral 等 MoE** | 专家为 SwiGLU 类 |
| **GPT-2 风教学** | GELU MLP |

### 6. 本课

\`\`\`bash
python chapters/01-inference/ep07_swiglu.py
\`\`\`

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
    summary: '可反复堆叠的一层残差三明治。',
    script: 'chapters/01-inference/ep08_block.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    markdown: `## 要点

从零搭模型：零件齐了要收成**可重复的一层**，才能叠深度。

### 1. 有什么作用

**Transformer 块**（pre-norm 三明治）：

\`\`\`
x → RMSNorm → 注意力 → + 残差
  → RMSNorm → SwiGLU → + 残差
\`\`\`

进出同形状；注意力可接 KV cache。

### 2. 从零开始：怎么生成

| 步骤 | 做什么 |
| --- | --- |
| **1** | 按上图接线（本课布局） |
| **2** | 每块独立一套权重 |
| **3** | 定深度 N（Llama 3.2 1B = **16**） |
| **4** | 堆叠进模型，整网训 |

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 残差可加；pre-norm 位置对；有 cache 时每层隔离 |
| **软** | 加深仍可训（残差+norm）；深度由算力与目标定 |

### 4. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 加深 N | 能力↑、显存/算力↑，需重训 |
| post→pre-norm | 现代默认，利于从零训深 |
| 注意力∥MLP 并行 | 非 Llama 主线；换布局要重训 |

### 5. 业界 · 开源

Llama/Qwen dense：pre-norm 块重复 N 次。DeepSeek 等在块内换注意力/FFN 变体，**块边界**仍在。

### 6. 本课

\`\`\`bash
python chapters/01-inference/ep08_block.py
\`\`\`

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

从零：一块不是语言模型；要 **嵌入 + N 块 + 最终 norm + LM head** 才有 next-token 分布。

### 1. 有什么作用

\`ids → 嵌入 → Block×N → 最终 RMSNorm → Linear(V) → logits\`

小 Llama 常 **tie** LM head ↔ 嵌入。

### 2. 从零开始：怎么生成

| 步骤 | 做什么 |
| --- | --- |
| **1** | 用已定 V、D、N、头数建空壳 |
| **2** | 串嵌入→块→最终 norm→head |
| **3** | 决定 tie 与否 |
| **4** | 随机初始化后要么加载权重，要么开预训练 |

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | logits \`(B,T,V)\`；配置与权重可对上；tie 约定一致 |
| **软** | 装权/训完后文字与指标达标 |

### 4. 如何改进 · 对整模影响

加深加宽、改 tie、改词表 V——都是架构级，需重训或严格迁移。堆叠本身不「单独优化」，优化的是各子模块与训练。

### 5. 业界 · 开源

开放 dense（Llama/Qwen）同一骨架；差异在超参与数据。MoE 则把块内 FFN 换成路由专家，骨架仍在。

### 6. 本课

\`\`\`bash
python chapters/01-inference/ep09_stack.py
\`\`\`

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

从零两条路交汇：**(A) 自己预训练写出检查点**；**(B) 加载开源检查点**对齐业界。加载错 = 前面手写全白费。

### 1. 有什么作用

按 key 映射把张量拷进模块。随机权重图对但无意义；装上 Llama 3.2 权重后，同前向才像语言模型。

### 2. 从零开始：怎么做

| 路径 | 步骤 |
| --- | --- |
| **自训** | 训完 \`state_dict\` / safetensors 存盘；记下分词版本 |
| **加载开源** | 许可+登录 → 下 HF 检查点 → **key 映射**进 arche → 配官方分词 |
| **校验** | 参数量、缺 key、短 prompt 与 HF 比 logits（后章） |

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 形状/数量对；映射无漏；分词配对 |
| **软** | 与 HF 对齐接近；生成可读 |

### 4. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 修映射 | 消除静默错层 |
| 精度（fp16/bf16） | 速度/显存/数值 |
| 从中间检查点恢复 | 续训；步数与 RNG 要一致 |
| 错分词 | 整模语义崩 |

### 5. 业界 · 开源

HF \`transformers\` + safetensors 是事实标准；Llama/Qwen/DeepSeek 均如此分发。引擎（vLLM）读同一类检查点。

### 6. 本课

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
\`\`\`

## 动手

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
\`\`\`

## 翻车

许可、登录、镜像——同分词章。

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

从零：有了 logits 还要**循环采样**，才是生成；产品差异多在停止与采样，不在「有没有循环」。

### 1. 有什么作用

取最后位置 logits → 温度/top-k（可选）→ 贪心或采样 → append → EOS/长度停止。

### 2. 从零开始：怎么生成这套逻辑

| 步骤 | 做什么 |
| --- | --- |
| **1** | Prefill/前向得 logits |
| **2** | 实现采样与停止条件 |
| **3** | 接上 KV 缓存（下章）降本 |
| **4** | 再接对话模板（后章）才像聊天 |

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 能停；只看过去；id 落在词表 |
| **软** | 温度行为符合预期；有真权重时连贯 |

### 4. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| temperature / top-p / top-k | 多样性 vs 可信；不改权重 |
| 停止符 / 最大长度 | 产品体验 |
| 约束解码 | 结构化输出 |
| 无缓存的循环 | 正确但太慢——必须上 KV |

### 5. 业界 · 开源

所有聊天栈同一循环；vLLM/TGI 等加速的是缓存与批处理，不是取消循环。Llama/Qwen 推理 API 都暴露采样旋钮。

### 6. 本课

\`\`\`bash
python chapters/01-inference/ep11_generate.py
\`\`\`

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

### 1. 有什么作用

**KV 缓存**在 prefill 时把每层的 K、V 存下来；之后每生成一个新 token（decode），只算新位置的 K/V，拼进缓存，再对**已有缓存**做注意力——不必每步把整段 prompt 重算一遍。

本页条带图可看时间上如何增长。

### 2. 为什么大模型过程需要它

朴素解码：每步都对「prompt + 已生成」全长重算 K/V → 算力随长度平方级浪费。  
生成本就是逐步循环（上一章）；缓存是让这个循环在工程上**可行、可服务**的关键。没有它，长对话解码会极慢极贵。

### 3. 怎么来的 · 怎么做

| 阶段 | 做什么 |
| --- | --- |
| **Prefill** | 一次前向，写入每层 K/V |
| **Decode** | 只追加一步 K/V，注意力读缓存 |
| **布局** | 本课教学：沿时间 \`cat\`；生产还有 Paged KV 等（后面引物） |
| **变小** | GQA / MLA 减少「每步要存的东西」 |

### 4. 有什么标准 · 怎样算好

| 类型 | 标准 | 人话 |
| --- | --- | --- |
| **硬** | 与全量重算数值等价（允许浮点差） | 缓存不能改语义 |
| **硬** | 逐步追加，不丢层 | 每层各有一份 |
| **软** | 算量随**新 token**涨，而不是每步整段 | 直接收益 |
| **软** | 显存占用可接受 | 再靠 GQA/分页等 |

### 5. 业界实际 · 本课怎么做

生产解码器都有 KV 缓存。  
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

### 1. 有什么作用

**RoPE 缩放**在上下文超过训练长度时，按约定配方（Llama 3：大 \`rope_theta\` + factor/low/high **波长平滑**）调整旋转，减轻「角度绕坏」。

### 2. 为什么大模型过程需要它

RoPE 的角度表是按训练长度调的。直接把位置下标拉到远超训练值，相对距离可能失真，长文质量掉。  
产品要 128K 级上下文时，常在**不改完所有权重**的前提下加缩放调度；也可配合更长数据继续训。

### 3. 怎么来的 · 怎么做

| 路线 | 人话 |
| --- | --- |
| **Llama 3 波长平滑** | 本课对照；与 HF 配置兼容 |
| NTK / YaRN / … | 其他长上下文 RoPE 配方 |
| 更长数据直接训 | 不只靠公式 |

### 4. 有什么标准 · 怎样算好

| 类型 | 标准 | 人话 |
| --- | --- | --- |
| **硬** | 与检查点声明的 scaling 一致 | 错配方 = 长文乱 |
| **软** | 近距离仍「像训练时」 | 短依赖别被毁掉 |
| **软** | 远距离可用而不崩 | 用长文评测，不是单看公式漂亮 |

### 5. 业界实际 · 本课怎么做

Llama 3.2 官方长上下文调度。  
\`arche/rope.py\` 缩放辅助 + 本章脚本。

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

### 1. 有什么作用

**MoE（混合专家）**用很多专家 FFN（这里是 SwiGLU）+ **路由器**：每个 token 只激活 **top-k** 个专家，其余不算——用稀疏计算换更大容量。

### 2. 为什么大模型过程需要它

Dense：每个 token 都跑同一套完整 MLP → 容量↑则人人付全额算力。  
MoE 想要「参数总量很大，但单 token 只走少数专家」，在同等算力下堆更多知识容量（训练与服务都更复杂）。

### 3. 怎么来的 · 怎么做

| 步骤 | 做什么 |
| --- | --- |
| **1** | 建 N 个专家 MLP + 路由线性 |
| **2** | 对每个 token 打分，选 top-k |
| **3** | 只跑选中专家，加权合并 |
| **训练** | 常加负载均衡等辅助损失（真训练很在意） |

本课是教学 MoE 块 API，不是完整 Mixtral 生产栈。

### 4. 有什么标准 · 怎样算好

| 类型 | 标准 | 人话 |
| --- | --- | --- |
| **硬** | 路由下标合法 | 不能指向不存在的专家 |
| **硬** | top-k 与实现一致 | 形状、合并要对 |
| **软** | 负载别极端倾斜 | 真训练核心；本课只尝 API |
| **软** | 同算力下质量 / 容量权衡 | 产品级评测 |

### 5. 业界实际 · 本课怎么做

Mixtral 风 top-k；还有共享专家等变体。  
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

### 1. 有什么作用

**MLA（多头潜在注意力思路）**不把每头完整 K/V 都缓存，而是存**低秩 latent**（常加一小段 RoPE 相关流），需要时再展开——进一步压解码显存。

### 2. 为什么大模型过程需要它

GQA 已减 KV 头，长上下文下 KV 仍可能占显存大头。  
DeepSeek 一类路线用 latent 压缩「要缓存的东西」，换实现与数值上的更多细节。

### 3. 怎么来的 · 怎么做

| 对比 | 作用 |
| --- | --- |
| GQA | 更少 KV 头 |
| **MLA** | latent KV（DeepSeek 思路） |
| Paged KV | 显存怎么切块管理 |

本课给教学形状与 API，**不是**比特级复刻 DeepSeek 权重。

### 4. 有什么标准 · 怎样算好

| 类型 | 标准 | 人话 |
| --- | --- | --- |
| **硬** | latent 缓存比满 KV 小 | 压缩成立 |
| **硬** | 前向能跑通、形状自洽 | 引物底线 |
| **软** | 与满注意力质量接近 | 产品级；本课不宣称对齐某正式版 |

### 5. 业界实际 · 本课怎么做

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

### 1. 有什么作用

把模型**跑起来并像聊天**：选设备、缩小权重体积、按对话模板把角色消息编成模型要的字符串。三块都不是「新算子」，但是上线必备胶水。

1. **设备** — CUDA → MPS → CPU  
2. **INT8（教学）** — 权重更小；简单反量化再算  
3. **对话模板** — 角色 → 模型字符串（HF 模板或纯文本回退）

本页可试对话 demo。

### 2. 为什么大模型过程需要它

主干算对了，仍可能：没 GPU、权重大、消息格式不对 → 没法交互。  
设备决定能不能跑；量化决定能不能塞进内存；模板决定模型是否按「对话」而不是「裸补全」来理解输入。

### 3. 怎么来的 · 怎么做

| 主题 | 做法 | 生产备注 |
| --- | --- | --- |
| 设备 | 探测可用后端，\`.to(device)\` | 同一图，不同后端 |
| 量化 | 教学 INT8；真产品还有 INT4/FP8 等配方 | 精度 / 速度权衡 |
| 模板 | 用模型自带 chat template | **别随便发明**格式 |

### 4. 有什么标准 · 怎样算好

| 类型 | 标准 | 人话 |
| --- | --- | --- |
| **硬** | 设备回退链可用 | 没 CUDA 不该硬崩 |
| **硬** | 模板与模型配对 | 错模板 → 胡话或拒答 |
| **软** | 量化后质量可接受 | 看评测；教学 INT8 ≠ 产品方案 |
| **软** | 多轮角色清晰 | 系统 / 用户 / 助手边界对 |

### 5. 业界实际 · 本课怎么做

教学辅助 + 脚本，不是完整量化产品。

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

### 1. 有什么作用

**连续批处理（教学版）**：多个短解码任务共享一个步进循环；谁先结束谁退役，空位留给新请求。避免「一人一个生成循环、快的在等慢的」。

可试本页 demo。**不是**完整 paged / vLLM。

### 2. 为什么大模型过程需要它

服务端同时服务很多人。每人独占一条生成循环时，结束节奏不同 → GPU 常空转。  
连续批把「这一步还有谁在跑」变成调度问题，提高利用率；真系统还要配合分页 KV、内核融合等。

### 3. 怎么来的 · 怎么做

| 层 | 作用 |
| --- | --- |
| **调度** | 每步谁还在跑（本课重点） |
| **分页 / 内核** | 显存与算力怎么跑（后面引物） |

### 4. 有什么标准 · 怎样算好

| 类型 | 标准 | 人话 |
| --- | --- | --- |
| **硬** | 结束序列正确退役 | 不把已结束的继续步进 |
| **硬** | 各序列结果互不串 | 批在一起 ≠ 语义混在一起 |
| **软** | GPU 更满、延迟更稳 | 服务指标；本课只示意形状 |

### 5. 业界实际 · 本课怎么做

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

### 1. 有什么作用

**预训练**用大量文本训练「猜下一个 token」：前向得 logits → 与真实下一 id 比交叉熵 → 反向更新权重（含嵌入表等）。这是嵌入表和几乎所有主干参数的**主产地**。

### 2. 为什么大模型过程需要它

随机权重没有语言能力。先在海量文本上学会统计与结构，再谈对话、偏好。  
没有预训练，后面的 SFT / DPO 没有像样的起点。

### 3. 怎么来的 · 怎么做

| 规模 | 含义 |
| --- | --- |
| **本实验** | 玩具 AdamW：几步、小数据，感受 API |
| **本页试跑** | 浏览器里拖学习率 / 步数，看 loss |
| **真预训练** | 海量数据、多卡、很久、复杂配方 |

步骤直觉：组 batch → 预测下一 token → loss → \`optimizer.step()\`。

### 4. 有什么标准 · 怎样算好

| 类型 | 标准 | 人话 |
| --- | --- | --- |
| **硬** | loss 对的是「下一个 id」 | 别算成别的任务 |
| **硬** | 梯度更新到该更新的参数 | 含嵌入等 |
| **软** | loss 下降曲线 | 玩具上能看到趋势即可 |
| **软** | 真训练：数据质量、稳定性、下游 | 本课不假装达到 |

### 5. 业界实际 · 本课怎么做

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

### 1. 有什么作用

预训练之后的对齐两步（本课 API）：

- **SFT**：示范「提示 → 回答」；**掩掉提示**，损失只打在回复 token 上  
- **DPO**：相对冻结的 reference，拉高 chosen、压低 rejected（教学一步）

### 2. 为什么大模型过程需要它

预训练会补全网页，不一定会「当助手」。  
SFT 教会格式与跟随；DPO / RLHF 一类方法再把输出往人类偏好靠。没有这类步骤，聊天产品很难用。

### 3. 怎么来的 · 怎么做

| 方法 | 数据 | 目标 |
| --- | --- | --- |
| SFT | 指令–回复对 | 模仿回复 |
| DPO | 同一提示的 chosen / rejected | 偏好排序（无显式 reward 模型的一种） |
| 更大版图 | RLHF / RLAIF / … | 配方很多；本课只尝两个 API |

### 4. 有什么标准 · 怎样算好

| 类型 | 标准 | 人话 |
| --- | --- | --- |
| **硬** | SFT 掩码正确 | 别在提示上算 loss |
| **硬** | DPO 有 reference / 配对样本 | API 形状对 |
| **软** | 更听指令、更少胡话 | 人类/自动评测 |
| **软** | 别把预训练能力洗没 | 数据与学习率权衡 |

### 5. 业界实际 · 本课怎么做

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

### 1. 有什么作用

**前沿检查点引物**：让 MoE / MLA 一类模块也能**存取权重**——往返教学检查点，并把 Mixtral 风 \`w1/w2/w3\` 等名字映射到我们的模块名，避免静默形状错乱。

### 2. 为什么大模型过程需要它

会跑 MoE/MLA 前向不够；开放权重以「别人的 key 名 + 分片文件」形式存在。  
不会加载 = 前沿结构只停留在纸上。

### 3. 怎么来的 · 怎么做

读开放 MoE 检查点的映射与往返；不是完整训练器。

### 4. 有什么标准 · 怎样算好

| 类型 | 标准 | 人话 |
| --- | --- | --- |
| **硬** | 存取后 \`allclose\` / 形状一致 | 别静默装错 |
| **硬** | 名称映射覆盖专家权重 | 漏映射 = 随机专家 |
| **软** | 能接真开放权重 | 产品级工程；本课打形状 |

### 5. 本课怎么做

本章脚本引物。

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

### 1. 有什么作用

**HF 对齐**：同一短 prompt，跑 arche 与官方 HF Llama 3.2，比 argmax 与 logits 最大绝对差。用来验证 dense 主干接线与权重映射可信。

### 2. 为什么大模型过程需要它

手写栈容易「能跑但装错层」。与权威实现对短序列 logits，是信任检查点加载的硬证据。无权重时干净退出，不等于逻辑坏了。

### 3. 怎么来的 · 怎么做

本章脚本 + CI 合成映射测试；需许可与登录才能下真权重。

### 4. 有什么标准 · 怎样算好

| 类型 | 标准 | 人话 |
| --- | --- | --- |
| **硬** | 有权重时 logits 足够接近 | 映射/公式没大错 |
| **硬** | 无权重时跳过并说明 | 别假装通过 |
| **软** | 更长序列、更多算子 | 后续扩展；本课短 prompt |

### 5. 本课怎么做

见动手；卡住看翻车。

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

### 1. 有什么作用

超大模型权重常是 \`index.json\` + 许多分片文件。本章用**合成** index + 分片（Mixtral 形 MoE / DeepSeek 形 MLA）练习打包/拆包，装回后 \`allclose\`——不必真下几十 GB。

### 2. 为什么大模型过程需要它

单文件装不下或不好分发。不会读分片 = 前沿开放权重用不成。教学用合成数据，专练**索引与拼接**，不练下载耐力。

### 3. 怎么来的 · 怎么做

生成合成分片 → 按 index 装回 → 数值对齐检查。

### 4. 有什么标准 · 怎样算好

| 类型 | 标准 | 人话 |
| --- | --- | --- |
| **硬** | index 与分片一致 | 缺片/错 key 要失败得明白 |
| **硬** | 装回 \`allclose\` | 拼接正确 |
| **软** | 接真仓库格式 | 产品工程 |

### 5. 本课怎么做

本章脚本。

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

### 1. 有什么作用

服务端两块常见加速引物：

1. **Paged KV** — 定长块 + 块表 + gather，灵活管理显存碎片  
2. **CUDA Graph** — 有 CUDA 时捕获固定 shape 计算，降低启动开销  

教学机制——**不是**可替换的 vLLM。

### 2. 为什么大模型过程需要它

玩具循环在单人短句上够用；线上高并发、变长请求会把显存切碎、把 kernel 启动打满。分页管显存，Graph 管「同一形状反复跑」的启动税。

### 3. 怎么来的 · 怎么做

\`arche/paged_kv.py\`（及 graph）展示块表 gather 与可选 CUDA Graph 捕获；形状固定时收益明显。

### 4. 有什么标准 · 怎样算好

| 类型 | 标准 | 人话 |
| --- | --- | --- |
| **硬** | gather 结果与逻辑缓存一致 | 分页不能改语义 |
| **硬** | 块表不越界 | 分配/回收正确 |
| **软** | 碎片更少、启动更少 | 服务指标；本课示意 |

### 5. 本课怎么做

本章脚本引物。路径走完可回看薄弱章，或有权重时跑 HF 对齐。

## 动手

\`\`\`bash
python chapters/01-inference/ep23_paged_cuda_graph.py
\`\`\`

## 下一章

路径走完——可回看薄弱章，或有权重时跑 HF 对齐。

`,
  },
];
