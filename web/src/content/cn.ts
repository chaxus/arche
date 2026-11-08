import type { Chapter } from './types';
import { PART0_CN } from './part0-cn';

export const CHAPTERS_CN: Chapter[] = [
  ...PART0_CN,
{
    id: 'overview',
    ep: '01',
    title: '总览',
    summary: '为什么用 Llama 3.2 ~1B 当例子；一条从零的「猜下一个词」主线。',
    script: 'chapters/01-inference/ep01_overview.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    markdown: `## 要点

本课是**一条从零到对接业界前沿**的路。阅读顺序：建表 → **猜词训练** → dense 积木 → 对齐 → 服务 → 前沿（MoE/MLA）。贯穿例子是 **Llama 3.2 ~1B**——例子配方错了，后面「业界怎么做」会对不上。

### 1. 有什么作用

你亲手写出 **Llama 3.2 风格** next-token 栈——分词 → 嵌入 → 块×N → LM head（**dense 解码器**）。后续章都挂在同一条主干上。官方 Llama 文件是这个例子的成品一对，不是第二条课。

\`\`\`mermaid
flowchart LR
  tok[分词表] --> emb[嵌入表]
  emb --> stack[块堆叠]
  stack --> head[LM head]
\`\`\`

### 2. 为什么用 Llama 3.2 ~1B 当例子

**一句话：学当前前沿 dense 架构，用你能跑完的规模。**

Llama 3.2 是 Meta **Llama 3** 系列的开放小规格——与更大 Llama（以及 Qwen 一类 dense）同属一套配方：**RMSNorm + RoPE + GQA + SwiGLU + 字节级 BPE**。这是今天开源 dense 主干的主流积木，不是为课堂简化的玩具网。MoE / MLA、更大引擎放在后面，作为挂在这条主干上的引物。

| 为什么选它 | 人话 |
| --- | --- |
| **前沿同构的积木** | 模块名与契约，和当前开源 dense / 推理栈对得上 |
| **同一家族、可学规模** | ~1B 让 V / D / 层数 / 头数可数、HF 可下——架构仍是 Llama 3，尺寸是为了学得完 |
| **开放成品一对** | 官方分词 + 检查点，「手写 vs 业界」才是真对照 |
| **先 dense** | 先拿下 dense 解码器，再摸 MoE/MLA；不从 GPT-2 博物馆起步 |

所以：**架构 = 当前前沿 dense；规模 = ~1B，好让从零写得完。** 不是「聊天最强」，也不是「在家训 70B」。

### 3. DeepSeek 算什么架构（同一条路，后面的地图）

口语里的 **DeepSeek 前沿开源**（V2 / V3、以及以 V3 为底的 R1）**不是**「另一个 Llama dense」。一句话对照：

| | **Llama 3.2（本课例子）** | **DeepSeek-V2 / V3 系** |
| --- | --- | --- |
| 容量 | 每层 **Dense** FFN | **MoE**——多专家，每次 top-k 激活 |
| 注意力 / KV | **GQA** | **MLA**——用 latent 压 KV |
| 共同底色 | RMSNorm / RoPE / 现代解码器骨架 | 同属 Transformer 家族，升级点不同 |
| 在本课位置 | 第 1–3 部分手写并跑通 | 第 4 部分引物：MoE → MLA → 检查点地图 |

**读法：** Llama 教 dense 主干；DeepSeek 是这条家族上很重要的开源 **MoE + MLA** 分支。不第一天就从 DeepSeek 起手——先要有 dense 栈。

### 4. 从零开始：怎么「生成」这条路线

| 步骤 | 做什么 |
| --- | --- |
| **1. 钉死例子** | Llama 3.2 ~1B |
| **2. 分词 + 嵌入** | 教学 BPE + 查表；记下 V、D、层数、头数 |
| **3. 猜下一个词训练** | shift + 交叉熵 + 反传——**写出**嵌入与权重 |
| **4. 手写积木** | RMSNorm、RoPE、GQA、SwiGLU… 再堆叠 |
| **5. 加载 Llama / 生成 / HF → 对齐 → 服务 → 前沿** | 侧栏一条路 |

### 5. 怎么样算选对了

| 类型 | 标准 |
| --- | --- |
| **硬** | 配置自洽；积木与声称的开源配方一致 |
| **硬** | 能支撑「从零」叙事：表可训、网可叠、权可装 |
| **软** | 社区文档全、权重可申请 |
| **软** | 后续 MoE/MLA 引物有地方挂（DeepSeek 向地图） |

### 6. 如何改进 · 对整模影响

换例子（如改 Qwen）→ 分词/嵌入/块配置全换，课程映射要重做。加深/加宽 → 全面重训。本课保持 1B 例子，用引物看更大模型。

### 7. 业界 · 开源怎么做

Llama / Qwen 主推 dense（也有 MoE 变体）；**DeepSeek-V3** 是开源侧 **大 MoE + MLA** 的旗舰叙事。都以「公开配置 + 检查点 + 分词」发布；vLLM 等引擎假设你已有这类检查点。

### 8. 本课

\`\`\`bash
python chapters/01-inference/ep01_overview.py
\`\`\`

打印配置尺寸，还不是聊天。

## 动手

\`\`\`bash
python chapters/01-inference/ep01_overview.py
\`\`\`

## 下一章

**分词器**（\`/cn/tokenizer/\`）：从零生成分词表。

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

本课（同一条路）提供两层：

- **表怎么长出来**：\`arche/teach_bpe.py\` 在小语料上真训一张表（看生成过程）  
- **Llama 例子的成品表**：官方 Llama 3.2 分词，与权重出厂一对

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
| **Llama 3 / 3.2** | 字节 BPE，128K+ | **本课例子**；官方 \`tokenizer\` 与权重出厂一对 |
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

官方例子表：

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

**嵌入层**（\`/cn/embedding/\`）：词表冻结后，嵌入表如何从零生成、怎样算好、如何改进及开源怎么做。

`,
  },
{
    id: 'embedding',
    ep: '03',
    title: '嵌入层',
    summary: 'id → 查表取出一排小数；为什么要小数，而不是词典释义。',
    script: 'chapters/01-inference/ep03_embedding.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    demo: 'embedding',
    markdown: `## 要点

分词只给出**整数编号**（id）。神经网络不会「理解」\`42\` 这个号码——它只会加减乘矩阵。所以中间必须有一步：**用 id 去一张大表里取出一排小数**。这张表就叫**嵌入表（embedding）**。

### 1. id 怎么变成小数？有什么用？

把它想成**储物柜墙**，不是词典释义：

1. 分词说：这段字切成 id \`7, 3, 7\`
2. 墙上有 \`V\` 个柜子（词表有多大就有多少柜），每个柜子里固定放 **\`D\` 个小数**（Llama 3.2 1B 里 \`D=2048\`）
3. **变成小数 = 打开对应柜门，把那一排放出来**（代码里叫 \`gather\` / \`Embedding\` 查表）  
   - id \`7\` → 第 7 行那 \`D\` 个数  
   - id \`3\` → 第 3 行  
   - 同一个 id **永远**对应同一行（训练前随机填、训练后被改成「有用的」数）

极小例子（\`D=3\`，假数）：

| id | 嵌入行（3 个小数） |
| --- | --- |
| 3 | \`0.2, -0.1, 0.5\` |
| 7 | \`-0.3, 0.8, 0.1\` |

输入 id 序列 \`[7, 3, 7]\` → 输出矩阵三行：先第 7 行，再第 3 行，再第 7 行。**没有公式把 7「算成」小数**；就是查表。

**有什么用（为什么非要小数）？**

| 没有嵌入（只拿着 id） | 有嵌入（一排小数） |
| --- | --- |
| id 只是编号，不能和权重矩阵正经相乘 | 后面的注意力、FFN **只吃小数张量** |
| \`猫=3\` 和 \`狗=9\` 的远近没有几何意义 | 训练会把常一起出现的 token 的行「推近/排开」，给整网当起点 |
| 换分词重编号会乱套 | 行号必须和分词 id **同一套编号** |

一句话：**分词负责编号；嵌入负责给每个号一排可训练、可计算的初始向量。** 表不好，后面全在噪声上练；表和分词错位，整条管线废。

**这些小数一开始从哪来？**  
建表时先**随机填**（几乎没语义）。然后整网做 next-token 训练：猜错了就反传，**顺便改动出现过的那些行**——常用 token 的行被改得最多。训完写入检查点；推理时只加载、取行，不再现场「发明」小数。本课也可直接加载 Llama 官方已经训好的行。

**为什么非要「一词一排小数」，而不是别的？**  
因为后面的层是连续数学（乘矩阵、加、非线性）。整数 id 不能当那种输入；一排小数可以。训练会让经常一起出现、角色相近的 token，其向量变得对「猜下一个词」有用——这是副作用，目标仍是预测，不是手工画词向量图。

本页上方的互动就是在做「按 id 取行」。

\`\`\`mermaid
flowchart LR
  ids[分词得到 id] --> gather[按行号取嵌入]
  gather --> floats[得到 T×D 小数]
  floats --> blocks[后面各层只吃小数]
\`\`\`

### 2. 从零开始：嵌入表怎么生成

**不是**手填释义，也**不是**再跑 BPE。BPE 决定「谁是几号」；嵌入决定「几号柜子里放哪串数」——数主要靠**预训练**写进去。

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
| **Llama 3 / 3.2** | \`nn.Embedding\` 风格查表；小规格常 **tie** LM head；**无**嵌入绝对 PE | **本课例子**；RoPE 管位置 |
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

**预训练**（\`/cn/pretrain/\`）：猜下一个词的训练回路——**写出**那些小数行（以及整棵主干）。

`,
    latex: 'X = \\mathrm{Embed}(t_{1:L}) \\in \\mathbb{R}^{L \\times d}',
  },
{
    id: 'pretrain',
    ep: '18',
    title: '预训练',
    summary: '猜下一个词：错位、交叉熵、反传——嵌入和权重怎样被写出来。',
    script: 'chapters/02-training/ep18_pretrain.py',
    partId: 'trunk',
    estimatedMinutes: 15,
    demo: 'pretrain',
    markdown: `## 要点

这是从零的**核心**：**猜下一个词的训练**（预训练）。嵌入行和几乎全部主干权重，都是这样从随机变成有用的。没有它，你只有空壳——或者只能加载别人训好的小数（后面的 Llama 例子）。

### 1. 有什么作用

在海量文本上，迫使模型根据前面的 id **猜下一个 id**。猜错 → 损失 → 反传 → 改参数（**含嵌入**）。嵌入章里「柜子里的小数」的主要作者，就是这个回路。

### 2. 推理猜词 vs 训练猜词

| | **推理**（聊天） | **训练**（本章） |
| --- | --- | --- |
| 真正的下一个 token | 不知道——从分数里选 | 数据里**已经有** |
| 打分之后 | argmax / 采样 → 拼上 | 和真 id 比 → **算损失** |
| 参数 | 通常冻住 | **每步更新** |

前向形状一样；训练多了「老师信号」和「改写权重」。

### 3. 从零：回路怎么走（人话步骤）

假想句子分词后 ids 是 \`[10, 20, 30]\`（对应 \`The cat sat\` 一类，编号是假的）：

| 位置 | 上下文 ids | 真正的下一个 id |
| --- | --- | --- |
| 0 | \`[10]\` | \`20\`（cat） |
| 1 | \`[10, 20]\` | \`30\`（sat） |

一步训练在干什么：

1. **文本 batch** → 分词 → id 张量  
2. **嵌入** = 按 id 取出小数行  
3. **前向**过主干 → **logits**，形状大约 \`(batch, 时间，V)\`——每个位置对词表里每个 id 打分  
4. **错位（shift）**：用位置 \`t\` 的上下文去预测位置 \`t+1\` 的 id（不能把答案漏进该步输入）  
5. **损失** = logits 与真下一 id 的交叉熵（对位置平均）  
6. **反向** = 梯度流回用过的嵌入行和每一块权重  
7. **优化器**（如 AdamW）走一小步；真跑会重复到海量 token  

\`\`\`mermaid
flowchart LR
  data[文本 batch] --> tok[分词]
  tok --> emb[嵌入取行]
  emb --> fwd[前向主干]
  fwd --> logits[logits]
  logits --> ce[与下一 id 交叉熵]
  ce --> bwd[反传]
  bwd --> upd[更新嵌入行与权重]
\`\`\`

初始嵌入是随机柜壁；**有用的小数是这个回路写进去的。** 后面各章手写 RMSNorm / RoPE / …，就是为了让第 3 步的前向变成*你的*栈，而不是黑盒。

### 4. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 目标真是下一 token；shift 正确；梯度打到嵌入 + 主干 |
| **硬** | 无 NaN；损失算在该算的位置 |
| **软** | loss 趋势下降；抽样不那么胡说 |
| **软** | lr / 裁剪稳住，长跑能跑完 |

### 5. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 数据质量与配比 | 能力上限主因 |
| 规模（参数 × token） | 缩放律 |
| 稳定性（norm、lr、裁剪） | 能不能训完 |
| 训练不足 | 嵌入**和**深层都欠拟合 |

下一部分的 SFT / DPO 是在「已经会猜词」之上再转向；**不能代替**这个基础回路。

### 6. 业界 · 开源

Llama / Qwen / DeepSeek 都是 **先预训练，再对齐**。真栈：Megatron / DeepSpeed / torchtitan 一类。本课是**玩具 API + 页内 playground**——感受形状，不是集群实跑。

### 7. 本课

\`\`\`bash
python chapters/02-training/ep18_pretrain.py
\`\`\`

页内可拖 lr / 步数。然后去写夹在「嵌入」和「logits」之间的那些积木。

## 动手

\`\`\`bash
python chapters/02-training/ep18_pretrain.py
\`\`\`

## 下一章

**RMSNorm**（\`/cn/rmsnorm/\`）：嵌入之后稳住尺度，深度猜词训练才跑得下去。

`,
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

**RoPE**（\`/cn/rope/\`）：从零把位置写进 Q/K。

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

**GQA**（\`/cn/gqa/\`）：从零定头数，省 KV 缓存。

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

Llama 3.2 1B 例子：**32 Q / 8 KV**。

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

**SwiGLU**（\`/cn/swiglu/\`）：从零接 FFN。

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

**Transformer 块**（\`/cn/block/\`）。

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

**堆叠 + LM head**（\`/cn/stack/\`）。

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

**加载权重**（\`/cn/weights/\`）。

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

仍是一条从零主线。本章加载 **Llama 例子**的开源检查点，让手写栈立刻能说话。Part 2 才体会这些小数怎样被**训练写出来**——真正的权重多半不是手填的。加载错 = 前面手写全白费。

### 1. 有什么作用

按 key 映射把张量拷进模块。随机权重图对但无意义；装上 Llama 3.2 权重后，同前向才像语言模型。

### 2. 从零开始：怎么做

| 同一条路上的步骤 | 做什么 |
| --- | --- |
| **加载例子（本章）** | 许可 + 登录 → 下 HF 检查点 → **key 映射**进 arche → 配官方分词 |
| **稍后训练（Part 2）** | 玩具/真预训练写出 \`state_dict\`；记下分词版本 |
| **校验** | 参数量、缺 key；生成后一章 **HF 对齐**比短 prompt logits |

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

**文本生成**（\`/cn/generate/\`）。

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


**HF logits 对齐**（\`/cn/hf-align/\`）：对照官方实现，验收手写栈。

`,
  },
{
    id: 'hf-align',
    ep: '21',
    title: 'HF logits 对齐',
    summary: '短 prompt 上对照 HF 的 logits。',
    script: 'chapters/01-inference/ep21_hf_align.py',
    partId: 'trunk',
    estimatedMinutes: 20,
    markdown: `## 要点

Part 1 收尾验收：**与权威 HF 实现对短 prompt logits**——证明映射与公式没大错。过关后再进入 Part 2，体会权重如何从零训练出来。

### 1. 有什么作用

同 prompt 跑 arche vs HF Llama 3.2；比 argmax 与 max abs diff。

### 2. 从零开始：怎么做

固定种子与精度；加载同一权重；比短序列；无权重则干净跳过。

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 有权时足够接近；无权时明确跳过 |
| **软** | 覆盖更长序列/更多算子 |

### 4. 如何改进 · 对整模影响

对齐失败 → 回查映射/RoPE/norm。通过对齐再谈优化与服务。

### 5. 业界 · 开源

开源复现常把「对上 HF/官方」当 CI 门禁；vLLM 等也有数值回归测。

### 6. 本课

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
\`\`\`

## 动手

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
\`\`\`

## 翻车

无权/未登录 → 跳过 ≠ 逻辑坏。

## 下一章


**SFT · DPO**（\`/cn/sft-dpo/\`）：在会猜词之后做对齐。

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

从零到可服务：生成循环正确还不够——**无 KV 缓存则长解码不可用**。这是对接业界推理的第一刀。

### 1. 有什么作用

Prefill 存每层 K/V；decode 只算新位置并追加，对缓存做注意力。

### 2. 从零开始：怎么生成

| 步骤 | 做什么 |
| --- | --- |
| **1** | 每层预分配/动态 \`cat\` 的 K/V 缓冲 |
| **2** | Prefill 写入；decode 追加 |
| **3** | 注意力改为读缓存 |
| **4** | 再演进分页（后章）与 GQA/MLA 减体积 |

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 与全量重算等价（浮点差内）；不串层、不丢步 |
| **软** | 算力随新 token 涨；显存可接受 |

### 4. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| GQA/MLA | 缓存更小 |
| Paged KV | 并发碎片 |
| 量化 KV | 显存↓、精度权衡 |
| 无缓存 | 仅玩具长度可行 |

### 5. 业界 · 开源

所有生产 decoder（vLLM、TGI、llama.cpp…）必有 KV 缓存；布局各异。Llama/Qwen 推理默认开启。

### 6. 本课

\`\`\`bash
python chapters/01-inference/ep12_kv_cache.py
\`\`\`

## 动手

\`\`\`bash
python chapters/01-inference/ep12_kv_cache.py
\`\`\`

## 下一章

**RoPE 缩放**（\`/cn/rope-scaling/\`）。

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

从零训完短上下文后，产品要 128K 级窗口——**RoPE 缩放**是常见「配方级」扩展，不是随便把下标拉长。

### 1. 有什么作用

按官方调度（Llama 3：大 \`theta\` + factor/low/high 波长平滑）调整旋转，减轻超长位置「绕坏」。

### 2. 从零开始：怎么做

| 步骤 | 做什么 |
| --- | --- |
| **1** | 先按训练长度训好 RoPE |
| **2** | 需要长文时启用与检查点一致的 scaling 配置 |
| **3** | 或继续用更长数据训 / 用 YaRN 等配方 |
| **4** | 用长文评测验证 |

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 与检查点声明一致 |
| **软** | 近距仍稳；远距可用；长文基准达标 |

### 4. 如何改进 · 对整模影响

错缩放 → 长文崩、短文也可能歪。更大窗口抬高 KV 显存与算力。缩放不能替代「没学过的长程依赖数据」。

### 5. 业界 · 开源

Llama 3/3.2 官方长上下文；Qwen/DeepSeek 各有长文声明；YaRN/NTK 常见于开源长文讨论。

### 6. 本课

\`\`\`bash
python chapters/01-inference/ep13_rope_scaling.py
\`\`\`

## 动手

\`\`\`bash
python chapters/01-inference/ep13_rope_scaling.py
\`\`\`

## 下一章


**设备 · INT8 · 对话**（\`/cn/device-quant-chat/\`）。

`,
  },
{
    id: 'moe',
    ep: '14',
    title: 'MoE',
    summary: '很多专家；每个 token 只跑 top-k。',
    script: 'chapters/01-inference/ep14_moe.py',
    partId: 'frontier',
    estimatedMinutes: 15,
    markdown: `## 要点

从零若要「更大容量、可控算力」，dense FFN 会贵——**MoE** 用稀疏专家换参数量。**DeepSeek-V3** 的开源叙事大体是 **MoE（容量）+ MLA（压 KV）**；本章先讲 MoE 这一半。Mixtral 也是开源 MoE；本课 Llama 3.2 例子故意先停在 **dense**。

### 1. 有什么作用

多专家 SwiGLU + 路由：每 token **top-k** 专家，其余不算。替换的是前面手写的 dense FFN——解码器骨架还在，换的是容量手段。

### 2. 从零开始：怎么生成

| 步骤 | 做什么 |
| --- | --- |
| **1** | 定专家数 E、top-k、是否共享专家 |
| **2** | 每专家一套 FFN + 路由线性 |
| **3** | 训练加负载均衡等辅助损失 |
| **4** | 推理按路由稀疏执行；检查点按专家存 |

本课是 API 引物，不是完整 Mixtral 训练栈。

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 路由合法；top-k 合并形状对 |
| **软** | 负载不太塌；同算力下质量/容量权衡达标 |

### 4. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 更多专家 / 更大 k | 容量或算力↑ |
| 均衡损失 | 防专家塌缩 |
| 替换 dense FFN | 架构级；需按 MoE 重训 |
| 通信与落地 | 多卡专家并行是工程主成本 |

### 5. 业界 · 开源

| 系 | 角色 |
| --- | --- |
| **DeepSeek-V3** | 大 **MoE** + **MLA**（下一章）——开源非 dense 旗舰叙事 |
| **Mixtral** | top-2 专家 SwiGLU 的 MoE |
| **Qwen-MoE 等** | 各自 E/k 配方 |
| **Llama 3.2 ~1B（本课）** | 先 dense——再读 DeepSeek 地图 |

### 6. 本课

\`\`\`bash
python chapters/01-inference/ep14_moe.py
\`\`\`

## 动手

\`\`\`bash
python chapters/01-inference/ep14_moe.py
\`\`\`

## 下一章

**MLA**（\`/cn/mla/\`）。

`,
  },
{
    id: 'mla',
    ep: '15',
    title: 'MLA',
    summary: '缓存一小段 latent，而不是完整 K/V。',
    script: 'chapters/01-inference/ep15_mla.py',
    partId: 'frontier',
    estimatedMinutes: 18,
    markdown: `## 要点

从零做长上下文服务：GQA 之后 KV 仍可能爆——**MLA**（Multi-head Latent Attention）用低秩 latent 压缓存。和 **MoE** 合在一起，就是口语里 **DeepSeek-V2 / V3** 的一句话画像：MoE 管容量，MLA 管 KV。

### 1. 有什么作用

缓存 latent（+ 小段 RoPE 流）而非满头 K/V，需要时再展开。Llama / Qwen 主线仍是 **GQA**；DeepSeek 开源前沿线推 **MLA**。

### 2. 从零开始：怎么生成

| 步骤 | 做什么 |
| --- | --- |
| **1** | 定 latent 维与投影 |
| **2** | 训练期按 MLA 前向（与论文/开源实现对齐） |
| **3** | 推理存 latent 缓存 |
| **4** | 本课只给形状引物，不比特级复刻 |

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 缓存更小；前向形状自洽 |
| **软** | 质量接近满 KV；实现与某正式版对齐（产品级） |

### 4. 如何改进 · 对整模影响

更小 latent → 更省、表达风险↑。与 GQA/分页正交可组合。换 MLA = 架构级，需按该配方训。

### 5. 业界 · 开源

| 系 | 注意力 / KV |
| --- | --- |
| **DeepSeek-V2 / V3** | **MLA**（+ MoE 本体） |
| **Llama 3 / Qwen dense** | **GQA** 主线 |
| 引擎（vLLM 等） | 逐步加 MLA 支持 |

### 6. 本课

\`\`\`bash
python chapters/01-inference/ep15_mla.py
\`\`\`

## 动手

\`\`\`bash
python chapters/01-inference/ep15_mla.py
\`\`\`

## 下一章


**前沿检查点**（\`/cn/frontier-ckpt/\`）。

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

从零模型要**给人用**：设备、体积、对话格式是上线胶水——不算新层，但缺一不可。

### 1. 有什么作用

1. **设备** CUDA→MPS→CPU  
2. **量化（教学 INT8）** 缩小权重  
3. **对话模板** 角色→模型字符串  

### 2. 从零开始：怎么做

| 步骤 | 做什么 |
| --- | --- |
| **1** | 探测后端，模块 \`.to(device)\` |
| **2** | 需要落地时选量化配方（本课 INT8 示意；产品常用 GPTQ/AWQ/FP8…） |
| **3** | 使用**该模型自带** chat template，勿自创 |
| **4** | 与生成循环、KV 一起接到服务 |

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 无 CUDA 可回退；模板与模型配对 |
| **软** | 量化后质量可接受；多轮角色清晰 |

### 4. 如何改进 · 对整模影响

更强量化 → 更小更快、质量风险↑。错模板 → 胡话。多后端维护成本在工程不在算法。

### 5. 业界 · 开源

HF \`apply_chat_template\`；vLLM/TGI 内置模板与量化加载；Llama/Qwen 均带官方模板。llama.cpp 系偏 GGML 量化。

### 6. 本课

\`\`\`bash
python chapters/01-inference/ep16_device_quant_chat.py
\`\`\`

## 动手

\`\`\`bash
python chapters/01-inference/ep16_device_quant_chat.py
\`\`\`

## 下一章

**连续批处理**（\`/cn/continuous-batching/\`）。

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

从零单用户循环 → 业界要**多用户共用步进**：连续批是服务利用率的核心调度形状。

### 1. 有什么作用

多任务共享一步；结束退役；新请求填入。教学版**不是**完整 vLLM。

### 2. 从零开始：怎么生成

| 步骤 | 做什么 |
| --- | --- |
| **1** | 维护「活跃序列」集合 |
| **2** | 每步只前向仍活跃者 |
| **3** | 结束移除；接入分页 KV（后章）才接近生产 |

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 结束必退役；序列互不串结果 |
| **软** | GPU 更满、尾延迟更稳 |

### 4. 如何改进 · 对整模影响

更好调度 → 同硬件更高吞吐。与模型权重无关，但与 KV 布局强耦合。做错会串话或漏停。

### 5. 业界 · 开源

vLLM 连续批 + paged attention 事实标准；TGI、TensorRT-LLM、SGLang 等同思路不同实现。

### 6. 本课

\`\`\`bash
python chapters/01-inference/ep17_continuous_batching.py
\`\`\`

## 动手

\`\`\`bash
python chapters/01-inference/ep17_continuous_batching.py
\`\`\`

## 下一章


**Paged KV · CUDA Graph**（\`/cn/paged-cuda-graph/\`）。

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

Part 2 收尾：预训练后仍要**对齐**——SFT 教会跟随；DPO/RLHF 一类推向偏好。之后进入 Part 3，把生成做成可服务的解码。

### 1. 有什么作用

- **SFT**：提示→回答；掩提示；loss 只在回复  
- **DPO**：相对 reference，抬 chosen、压 rejected  

### 2. 从零开始：怎么做

| 步骤 | 做什么 |
| --- | --- |
| **1** | 准备指令数据 / 偏好对 |
| **2** | 从预训练检查点续训 SFT |
| **3** | 再 DPO/RLHF（配方任选） |
| **4** | 评估拒答、跟随、回归 |

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | SFT 掩码对；DPO 样本与 reference 形对 |
| **软** | 更听指令、更安全；少伤预训练能力 |

### 4. 如何改进 · 对整模影响

数据质量 > 盲目步数。过强对齐可能「变傻」。可与扩特殊符、改模板一起做。

### 5. 业界 · 开源

Llama/Qwen 均有 Instruct/Chat 变体；Zephyr 等推 DPO；OpenAI/Anthropic 用 RLHF 变体（细节不全开）。TRL 等库是开源对齐常用工具。

### 6. 本课

\`\`\`bash
python chapters/02-training/ep19_sft_dpo.py
\`\`\`

## 动手

\`\`\`bash
python chapters/02-training/ep19_sft_dpo.py
\`\`\`

## 下一章


**KV 缓存**（\`/cn/kv-cache/\`）：生成要能服务，先把解码成本打下来。

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

从零若走上 MoE/MLA，检查点**名字与形状**必须接得住开放权重——否则前沿结构只停留在纸上。

### 1. 有什么作用

教学往返检查点；Mixtral 风 \`w1/w2/w3\` 映射到本课模块名。

### 2. 从零开始：怎么做

设计模块名 ↔ 开源名映射；保存/加载；缺 key 失败要响。

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 往返对齐；专家权重全覆盖 |
| **软** | 能接真开放权重仓库 |

### 4. 如何改进 · 对整模影响

映射错 → 静默随机专家。统一命名减少工程债。

### 5. 业界 · 开源

HF 分片 + 文档化 key；Mixtral/DeepSeek 各自命名。转换脚本是生态标配。

### 6. 本课

\`\`\`bash
python chapters/02-training/ep20_frontier_ckpt.py
\`\`\`

## 动手

\`\`\`bash
python chapters/02-training/ep20_frontier_ckpt.py
\`\`\`

## 下一章


**分片 MoE / MLA**（\`/cn/sharded-frontier/\`）。

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

从零训到大模型：权重是 **index.json + 分片**。不会拼分片就接不住开放前沿权重。

### 1. 有什么作用

合成分片练习 pack/unpack，装回 \`allclose\`——免下几十 GB。

### 2. 从零开始：怎么做

按 HF 习惯写 index；按张量切分保存；加载时按 index 聚合。

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | index 与分片一致；装回对齐；缺片报错清楚 |
| **软** | 兼容真仓库格式 |

### 4. 如何改进 · 对整模影响

分片策略影响下载并行与内存峰值；错拼 = 静默坏权。

### 5. 业界 · 开源

HF Hub 分片是标准；Llama/Qwen/DeepSeek/Mixtral 均如此。\`safetensors\` + index 是主格式。

### 6. 本课

\`\`\`bash
python chapters/02-training/ep22_sharded_frontier.py
\`\`\`

## 动手

\`\`\`bash
python chapters/02-training/ep22_sharded_frontier.py
\`\`\`

## 下一章


路径走完——可回看薄弱章，或有权重时再跑 HF 对齐。

`,
  },
{
    id: 'paged-cuda-graph',
    ep: '23',
    title: 'Paged KV · CUDA Graph',
    summary: '块表 KV 池与固定 shape 的 CUDA graph。',
    script: 'chapters/01-inference/ep23_paged_cuda_graph.py',
    partId: 'infer',
    estimatedMinutes: 18,
    markdown: `## 要点

Part 3 收尾引物：**分页 KV** 管显存碎片，**CUDA Graph** 砍固定形状启动税——对接 vLLM 一类引擎。下一 Part 进入 MoE/MLA 等前沿结构。

### 1. 有什么作用

1. Paged KV：定长块 + 块表 + gather  
2. CUDA Graph：捕获固定 shape 计算  

教学 ≠ 可替换 vLLM。

### 2. 从零开始：怎么做

先连续批与逻辑 KV → 引入块表分配/回收 → 可选 Graph 捕获 decode 步。

### 3. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | gather 与逻辑缓存一致；块表不越界 |
| **软** | 更少碎片、更少启动；吞吐↑ |

### 4. 如何改进 · 对整模影响

页大小、预分配策略改吞吐与浪费。Graph 要求形状稳；动态 batch 要小心重捕获。不改模型权重，改服务上限。

### 5. 业界 · 开源

vLLM paged attention 标杆；TensorRT-LLM/SGLang 有图与分页变体；与 Llama/Qwen/DeepSeek 权重搭配使用。

### 6. 本课

\`\`\`bash
python chapters/01-inference/ep23_paged_cuda_graph.py
\`\`\`

路径走完：可回看薄弱章，或有权重时跑 HF 对齐。

## 动手

\`\`\`bash
python chapters/01-inference/ep23_paged_cuda_graph.py
\`\`\`

## 下一章


**MoE**（\`/cn/moe/\`）：进入前沿结构。

`,
  },

];
