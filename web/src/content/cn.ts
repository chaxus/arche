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

**是什么（What）：** *因果语言建模* / **下一个 token 预训练**——只看左边的 token，给词表每个 id 打「下一个是谁」的分，再训练让**真的下一个 id**概率变高。

**为什么（Why）：** 这一个自监督目标，就是**写出**嵌入行和几乎全部主干权重的作者。没有它只有随机空壳（或只能加载别人的小数，如后面的 Llama 例子）。

**怎么做（How，一句话）：** 文本分词 → 嵌入 → 前向 → **错位（shift）**让位置 \`t\` 预测 id \`t+1\` → 交叉熵 → 反传 → 优化器一步。在海量语料上重复。

业界同名事物：nanoGPT 的 \`X,Y\` 窗、Hugging Face「causal LM」、Llama 3 / DeepSeek 的「先 pretrain 再 align」。

### 1. What — 训练目标

对序列 \\(x_1,\\ldots,x_T\\)：

\\[
p(x_1,\\ldots,x_T)=\\prod_t p(x_t \\mid x_{<t})
\\]

训练最小化平均**负对数似然** = 逐 token **交叉熵**。标签来自文本自己（自监督）。

| 名字 | 含义 |
| --- | --- |
| **logits** | softmax 前的分数，形状 \`(batch, time, V)\` |
| **next-token CE** | 每个时间步惩罚「真下一 id」概率过低 |
| **因果** | 位置 \`t\` 不能看见 \`t\` 之后的 token |

### 2. Why — 为什么要这样设计

| 选择 | 原因 |
| --- | --- |
| 预测**下一个** id | 任意原文都能变成训练信号 |
| 一次前向、多位置监督 | 长 \`T\` 的序列约贡献 \`T-1\` 个合法猜题（靠因果 mask） |
| 词表上的 CE | 对应「离散选词」；经 softmax 可反传 |
| 更新**嵌入 + 全部积木** | 嵌入章的小数不是手填的——在这里被改写 |
| 先 pretrain 再 SFT/DPO | 对齐是转向，不是从零发明「会猜词」 |

推理 vs 训练：

| | 推理 | 训练（本章） |
| --- | --- | --- |
| 真下一 id | 不知道 | batch 里已知 |
| logits 之后 | 采样 / argmax | 与真 id 做 CE |
| 权重 | 通常冻住 | 每步更新 |

### 3. How — 错位、损失、一步（仓库真码）

假 ids \`[10, 20, 30]\`：

| 位置 | 概念上看到的 | 要预测 |
| --- | --- | --- |
| 0 | \`10\` | \`20\` |
| 1 | \`10,20\` | \`30\` |

**本仓库实现**（[\`arche/train/pretrain.py\`](https://github.com/chaxus/arche/blob/main/arche/train/pretrain.py)）：

\`\`\`python
def lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """用位置 0..T-1 预测 token t+1。"""
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),  # 去掉最后时刻
        targets[:, 1:].reshape(-1),                   # 去掉第一个 id
    )
\`\`\`

同一份 \`batch\` 既当输入，又当标签来源：\`logits[t]\` 对齐 \`targets[t+1]\`。

一步：

\`\`\`python
opt.zero_grad(set_to_none=True)
logits, _ = model(batch)       # (B, T, V)
loss = lm_loss(logits, batch)
loss.backward()
opt.step()
\`\`\`

实验脚本（[\`ep18_pretrain.py\`](https://github.com/chaxus/arche/blob/main/chapters/02-training/ep18_pretrain.py)）：玩具 \`Llama\` + 随机 id batch + \`train_pretrain_steps\` → 打印 loss 起止。

**HF 坑：** \`DataCollatorForLanguageModeling(mlm=False)\` 常把 \`labels = input_ids\` **不预先错位**；Transformers 的 CausalLM 在 \`forward\` **内部** shift。手写回路（本课 / nanoGPT）是显式 shift。**不要 shift 两次。**

\`\`\`mermaid
flowchart LR
  batch[ids_BT] --> fwd[model前向]
  fwd --> logits[logits_BTV]
  logits --> shift[错位对齐]
  shift --> ce[交叉熵]
  ce --> bwd[反传]
  bwd --> adam[AdamW一步]
\`\`\`

随机初始化时，loss 大约 \`log(V)\`（均匀猜词表）。

### 4. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 只错一位；因果（不漏未来） |
| **硬** | 梯度打到嵌入行 + 主干；无 NaN |
| **软** | loss 下降；抽样不那么胡说 |
| **软** | 大规模时对齐公开菜谱（后章） |

### 5. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 更多/更好的 token | 能力上限主因 |
| shift / mask 正确 | 错了会静默训歪 |
| lr / 裁剪 / 累积 | 能不能跑完（下一章） |
| 训练不足 | 嵌入**和**深层都弱 |

后面把**数据窗**、**优化器**、**前沿规模菜谱**拆开，本章专注「目标 + 一步」。

### 6. 业界指针

- Raschka：input/target 窗怎么切  
- nanoGPT \`train.py\`：可读完整环  
- HF LLM Course ch7.6：从零因果 LM  
- Llama 3 / DeepSeek 报告：目标同族，规模与结构不同  

### 7. 本课

\`\`\`bash
python chapters/02-training/ep18_pretrain.py
\`\`\`

页内可拖 lr / 步数。前向积木细节在数据/优化引物之后的 RMSNorm→…。

## 动手

\`\`\`bash
python chapters/02-training/ep18_pretrain.py
\`\`\`

## 下一章

**预训练数据**（\`/cn/pretrain-data/\`）：batch 与滑窗怎么来。

`,
  },
{
    id: 'pretrain-data',
    ep: '18b',
    title: '预训练数据',
    summary: '切窗、打包、组 batch——文本如何变成 CE 能吃的 (B, T)。',
    script: 'chapters/02-training/ep18_pretrain.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    markdown: `## 要点

**是什么（What）：** 预训练如何把原文变成损失能吃的 **\`(batch, time)\` id 张量**——文档 → 分词 → 切窗/打包 → 组 batch →（可选）pad/mask。

**为什么（Why）：** *预训练*章的目标是固定的；**数据形状**决定每步看见什么信号。窗切错（pad 当标签、漏未来、乱拼接）会静默浪费或训歪。

**怎么做（How）：** 固定长度上下文（或把多文档打进一条流），组成 batch，并保证 **labels 的错位规则与损失一致**。

### 1. What — batch 由什么组成

| 零件 | 作用 |
| --- | --- |
| **文档** | 原文（网页、书、代码…） |
| **token ids** | 分词器输出（BPE 等） |
| **窗 / block** | 模型一次看见的长度 \`T\`（上下文） |
| **batch** | \`B\` 个窗叠成 \`(B, T)\` |
| **labels** | 同一串 ids，按 shift（或 HF 内部 shift）解释 |

两种常见布局：

| 布局 | 想法 |
| --- | --- |
| **滑窗 / 切块** | 长 id 流切成不重叠或步长 \`T\` 的块（nanoGPT 风格） |
| **打包 packing** | 多篇短文拼进一条，用分隔/注意力边界，GPU 更满 |

### 2. Why — 设计压力

| 压力 | 后果 |
| --- | --- |
| GPU 喜欢密实 \`(B,T)\` | 倾向 packing，而不是碎短序列 |
| 上下文 \`T\` | 限制「下一个 token」能回看多远 |
| pad | **不能**当真词来训（mask 或 ignore_index） |
| 文档边界 | 盲拼接会跨无关文档造假「下一词」，除非 mask |

### 3. How — 玩具窗（对齐本仓库）

实验 batch 是随机 id，教**回路**，不是真语料：

\`\`\`python
batch = torch.randint(0, V, (B, T))
logits, _ = model(batch)
loss = lm_loss(logits, batch)  # 用 batch[:, 1:] 作目标
\`\`\`

生产草图（概念，类 nanoGPT）：

\`\`\`python
# tokens: 一维长 id 流
ix = torch.randint(0, len(tokens) - T, (B,))
x = torch.stack([tokens[i : i + T] for i in ix])
y = torch.stack([tokens[i + 1 : i + T + 1] for i in ix])
# 再 CE(model(x), y) —— y 已错位；不要再 shift
\`\`\`

**HF：** collator 若 \`labels == input_ids\`，**模型内部** shift；若你自己造 \`y\`，CE **不要**再 shift。

\`\`\`mermaid
flowchart TD
  docs[原文] --> tok[分词]
  tok --> stream[id流]
  stream --> win[切窗或打包]
  win --> bat[叠成_BT]
  bat --> loss[与lm_loss同一套错位]
\`\`\`

### 4. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 每个监督位置有真下一 id（或被 mask） |
| **硬** | label/输入错位与损失实现一致 |
| **软** | token 利用率高（少纯 padding） |
| **软** | 领域混合覆盖你关心的场景 |

### 5. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 更干净/更多 token | 底座更强 |
| 更长 \`T\` | 长程猜词更好（成本↑） |
| packing 质量 | 吞吐与每天有效更新 |
| 领域配比 | 模型「流利」在哪些域 |

### 6. 本课

暂无同一脚本：

\`\`\`bash
python chapters/02-training/ep18_pretrain.py
\`\`\`

把随机 batch 当成「假窗」。真 DataLoader 离开玩具脊柱后再接。

## 动手

\`\`\`bash
python chapters/02-training/ep18_pretrain.py
\`\`\`

## 下一章

**预训练优化**（\`/cn/pretrain-optim/\`）：AdamW、学习率、梯度裁剪、累积。

`,
  },
{
    id: 'pretrain-optim',
    ep: '18c',
    title: '预训练优化',
    summary: 'AdamW、学习率、裁剪、累积——让 CE 一步跑得稳。',
    script: 'chapters/02-training/ep18_pretrain.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    markdown: `## 要点

**是什么（What）：** 预训练一步的**优化器一侧**——AdamW、学习率（与日程）、梯度裁剪、梯度累积——让 CE 目标能稳定改写权重。

**为什么（Why）：** 损失写对了、优化器写烂 → NaN、发散或 loss 卡死。离开玩具规模后，**怎么迈步**决定跑得完跑不完。

**怎么做（How）：** \`zero_grad\` → 前向 → loss → \`backward\` →（可选 clip）→ \`optimizer.step\` →（可选 lr schedule）。\`B\` 必须很小时，用微 batch **累积梯度**。

### 1. What — 旋钮

| 旋钮 | 作用 |
| --- | --- |
| **AdamW** | 自适应动量 + **解耦 weight decay**（Llama/nanoGPT 类菜谱默认） |
| **lr** | 步长；太大易炸，太小爬行 |
| **schedule** | 先 warmup 再衰减（cosine / WSD / …），避免早期撕碎随机嵌入 |
| **grad clip** | 限制全局梯度范数（如 1.0）抗尖峰 |
| **grad accum** | \`K\` 个微 batch 累加梯度 ≈ 更大有效 batch |

本仓库玩具环（[\`train_pretrain_steps\`](https://github.com/chaxus/arche/blob/main/arche/train/pretrain.py)）：

\`\`\`python
opt = torch.optim.AdamW(model.parameters(), lr=lr)
for batch in batches:
    opt.zero_grad(set_to_none=True)
    logits, _ = model(batch)
    loss = lm_loss(logits, batch)
    loss.backward()
    opt.step()
\`\`\`

尚无 schedule/clip——刻意如此。生产在同一骨架外包一层。

### 2. Why — 各自的理由

| 选择 | 原因 |
| --- | --- |
| AdamW 而非纯 SGD | token 梯度偏稀疏 + 参数极多；自适应步长更稳 |
| Weight decay | 轻度正则；AdamW 形式对齐业界 LLM 实践 |
| Warmup | 早期 CE 很大（\`~log V\`）；硬 lr 会打烂初始化 |
| Clip | 一个坏 batch 不该炸掉浮点 |
| 累积 | 显存不够时仍保持有效 batch |

### 3. How — 累积草图

\`\`\`python
opt.zero_grad(set_to_none=True)
for micro in micro_batches:
    loss = lm_loss(*model_and_shift(micro)) / K
    loss.backward()          # 梯度累加
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()
# scheduler.step()  # 若有
\`\`\`

\`\`\`mermaid
flowchart LR
  mb[微batch] --> bwd[反传_x_K]
  bwd --> clip[梯度裁剪]
  clip --> step[AdamW一步]
  step --> sch[lr日程]
\`\`\`

### 4. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | loss/梯度无 NaN/Inf（clip 后） |
| **硬** | 每个**优化器**步才 \`zero_grad\`（累积时别清早了） |
| **软** | 合理窗口内 loss 下行 |
| **软** | 放大时 lr 日程对齐已知菜谱 |

### 5. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| lr / warmup | 深层 + 嵌入是否训得动 |
| 有效 batch（B×accum×卡数） | 更新噪声 vs 稳定 |
| clip 阈值 | 扛尖峰 vs 更新过弱 |
| decay 设错 | 过度收缩或正则不足 |

### 6. 本课

\`\`\`bash
python chapters/02-training/ep18_pretrain.py
\`\`\`

页内把 lr 拧高到 loss 发疯——就是缩小版的优化课。

## 动手

\`\`\`bash
python chapters/02-training/ep18_pretrain.py
\`\`\`

## 下一章

**预训练规模**（\`/cn/pretrain-scale/\`）：前沿菜谱如何拉长同一条环。

`,
  },
{
    id: 'pretrain-scale',
    ep: '18d',
    title: '预训练规模',
    summary: '同一条 next-token 环如何放大到前沿：token、阶段、并行。',
    script: 'chapters/02-training/ep18_pretrain.py',
    partId: 'trunk',
    estimatedMinutes: 10,
    markdown: `## 要点

**是什么（What）：** 把同一条 next-token 环**放大**的入门图——更多 token、分阶段、并行——先不离开初学者脊柱去做完整系统课。

**为什么（Why）：** Llama 3.2 ~1B 与前沿大跑共享 **目标 + 一步**；差别在 **数据量、分阶段菜谱、集群账、以及（有时）结构**（稠密 vs MoE）。有这张地图，才不会「玩具 loss 降了就毕业」。

**怎么做（How）：** 把公开菜谱读成你已有章节的*扩展*：还是 CE，更难的是数据 / 优化 / 工程。

### 1. What — 放大的轴

| 轴 | 玩具（本课） | 前沿形态 |
| --- | --- | --- |
| **Tokens** | 几个随机 batch | 万亿级真 token |
| **步数 / 阶段** | 一条平坦环 | 多段 lr、数据配比、加长上下文 |
| **Batch** | 很小的 \`B,T\` | 累积 + 数据并行撑起巨大有效 batch |
| **模型** | 极小稠密 \`Llama\` | 稠密 1B–70B+ 或 MoE（DeepSeek）+ MLA |
| **硬件** | 单进程 | Megatron/FSDP/流水线式切分 |

Llama 3 类公开报告：长 pretrain + 短高质量 anneal / 长上下文阶段——心里仍是 next-token。

DeepSeek-V2/V3：**同一族因果 LM**，不同的是**块**（MLA + MoE）——见 Frontier；训练仍是 CE + 大数据优化。

### 2. Why — 为什么不能停在 ep18

| 错觉 | 现实 |
| --- | --- |
| 「随机 id 上 loss 降了」 | 只证明**回路**，不是语言能力 |
| 「抄 AdamW 默认就行」 | 放大后要 warmup/日程/clip |
| 「只把模型变大」 | 数据质量 × token 量往往更关键 |
| 「SFT 能救烂底座」 | 对齐改风格；弱 pretrain 仍然弱 |

### 3. How — 四行读懂一篇菜谱

打开 Llama / DeepSeek / Megatron 文时，对照：

1. **目标** → 仍是 next-token CE（预训练）  
2. **数据** → 配比、packing、上下文日程（预训练数据）  
3. **优化** → AdamW、lr 形状、clip、有效 batch（预训练优化）  
4. **系统** → 并行、精度、checkpoint（更后 / 运维）

\`\`\`mermaid
flowchart TD
  obj[同一CE目标] --> data[更多更好的token]
  data --> opt[稳定的大batch优化]
  opt --> sys[多卡切分]
  sys --> align[再SFT_DPO]
\`\`\`

### 4. 怎么样算好（初学者条）

| 类型 | 标准 |
| --- | --- |
| **硬** | 能指出论文动的是哪一根轴 |
| **软** | 不把 MoE**结构**误当成另一种**损失** |
| **软** | 知道「复现 Llama」前还有工程章 |

### 5. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| Token 预算 | 能力主因 |
| 分阶段数据 / LR | 稳定与末期质量 |
| 并行效率 | 预算是否用得起来 |
| 结构（GQA/MoE/MLA） | 每 token 成本与显存——不是新目标 |

### 6. 本课

入口仍是：

\`\`\`bash
python chapters/02-training/ep18_pretrain.py
\`\`\`

脊柱下一步：稠密**积木**（RMSNorm→…），让 \`model(batch)\` 背后的前向不再是黑盒——再 HF 对齐权重，再 **SFT/DPO**。

## 动手

\`\`\`bash
python chapters/02-training/ep18_pretrain.py
\`\`\`

## 下一章

**RMSNorm**（\`/cn/rmsnorm/\`）：你刚在训的那次前向里的第一块积木。

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

**是什么（What）：** **RMSNorm**——按 token 稳住尺度：除以均方根，再乘可学习 \`γ\`。不减均值、无 bias。

**为什么（Why）：** 深 next-token 栈没有归一化容易炸或塌。Llama 风格 **pre-norm**（先 norm 再 attn / MLP）让从零堆深度训得动。

**怎么做（How）：** 每个 pre-norm 位点放 \`RMSNorm(D)\`；\`γ\` 初值约 1；与整网一起训。

### 1. What — 公式

\\[
\\mathrm{RMSNorm}(x)=\\frac{x}{\\sqrt{\\mathrm{mean}(x^2)+\\varepsilon}}\\odot\\gamma
\\]

仓库（[\`arche/norms.py\`](https://github.com/chaxus/arche/blob/main/arche/norms.py)）：

\`\`\`python
variance = x_f.pow(2).mean(dim=-1, keepdim=True)
x_normed = x_f * torch.rsqrt(variance + self.eps)
return (self.weight.float() * x_normed).to(orig_dtype)
\`\`\`

在 float32 里算再 cast 回去——与 HF Llama 同一类稳定性手法。

### 2. Why — 相对 LayerNorm / 位置

| 选择 | 原因 |
| --- | --- |
| 不减均值 | 稍省；对齐 Llama/Qwen 菜谱 |
| 只有可学习 \`γ\` | 单位 RMS 后再按通道缩放 |
| **Pre-norm** | 稳住 attn/MLP *输入*，深度才活 |
| 嵌入之后、积木之前 | 有了向量就要先「稳住尺子」 |

GPT-2 教学常是 LayerNorm + post-norm。本课例子是 **RMSNorm + pre-norm**。

### 3. How — 怎么接

| 步骤 | 做什么 |
| --- | --- |
| 1 | 注意力前、SwiGLU 前各一个（见 block） |
| 2 | LM head 前再一个最终 RMSNorm（见 stack） |
| 3 | \`ε\` 对齐 checkpoint（\`norm_eps\`） |
| 4 | 训或加载；勿在外来权重下 LN↔RMSN 混用 |

### 4. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 形状不变；有 \`ε\`；pre-norm 位点对齐菜谱 |
| **软** | \`γ=1\` 时 RMS≈1；深前向更少 NaN |
| **软** | 整模 loss / HF 对齐才是终审 |

### 5. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| LN → RMSN | 结构级变更 ⇒ 重训 |
| Pre vs post | 深度可训性 |
| \`γ\` / lr | 耦合收敛 |

### 6. 本课

\`\`\`bash
python chapters/01-inference/ep04_rmsnorm.py
\`\`\`

## 动手

\`\`\`bash
python chapters/01-inference/ep04_rmsnorm.py
\`\`\`

## 下一章

**RoPE**（\`/cn/rope/\`）：把位置写进 Q/K。

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

**是什么（What）：** **RoPE**（旋转位置编码）——按 token 位置旋转 Q、K，让**相对距离**进入注意力点积。

**为什么（Why）：** 嵌入说明「是哪个词」；注意力还要「在哪」。嵌绝对位置表是旧派；现代稠密 Llama/Qwen/DeepSeek 在 Q/K 上用 RoPE。

**怎么做（How）：** 选定 \`rope_theta\`，缓存 cos/sin，\`rotate_half\` 后与 Q/K 组合；长上下文缩放是后话。

### 1. What — 施加一步

仓库（[\`arche/rope.py\`](https://github.com/chaxus/arche/blob/main/arche/rope.py)）：

\`\`\`python
def apply_rotary_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)
\`\`\`

\`build_rope_cache\` 预计算 \`0 .. seq_len-1\` 的 cos/sin。带 cache 解码时按 \`start_pos\` 切片。

### 2. Why — 设计理由

| 选择 | 原因 |
| --- | --- |
| 只旋 Q/K | V 保持内容；位置进分数 |
| 旋转带来相对性 | Dot(q_m, k_n) 依赖 \`m-n\` |
| 大 \`theta\`（Llama 3: 50 万） | 比旧 10k 更能撑长上下文 |
| 对齐 checkpoint 的 theta/scale | PE 错了，加载权重后会静默变差 |

### 3. How — 从零

| 步骤 | 做什么 |
| --- | --- |
| 1 | \`head_dim\` 偶数；由 \`theta\` 建 inv-freq |
| 2 | 缓存到最大序列（显存折中） |
| 3 | Q/K 投影之后、SDPA 之前施加 |
| 4 | 长上下文：Llama-3 波长平滑（后章 \`rope-scaling\`） |

### 4. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 形状不变；头维偶数；Q、K 都施加 |
| **硬** | checkpoint 的 \`theta\` / scale 一致 |
| **软** | 近邻与远邻分数有别 |

### 5. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| Theta / scaling | 长上下文表现 |
| 绝对 PE → RoPE | 现代默认；勿与外来权重混家族 |
| 跳过 RoPE | 模型难以可靠用顺序 |

### 6. 本课

\`\`\`bash
python chapters/01-inference/ep05_rope.py
\`\`\`

## 动手

\`\`\`bash
python chapters/01-inference/ep05_rope.py
\`\`\`

## 下一章

**GQA**（\`/cn/gqa/\`）：省 KV cache 的头布局。

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

**是什么（What）：** **分组查询注意力（GQA）**——Q 头多于 KV 头；多组 Q **共享**一组 KV。仍是因果注意力。

**为什么（Why）：** 纯 MHA 每个 Q 头存一对 KV → 解码显存随长度爆。GQA 大幅砍 cache，质量损失通常可控。Llama 3.2 ~1B：**32 Q / 8 KV**。

**怎么做（How）：** 选可整除的 \`n_q\` / \`n_kv\`，投影，RoPE，对 KV **repeat_interleave** 对齐 Q，再 SDPA；cache 只存 \`n_kv\`。

### 1. What — 本仓库形状

[\`CausalSelfAttention\`](https://github.com/chaxus/arche/blob/main/arche/attention.py)：

\`\`\`python
q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
# ... RoPE ...
k_full = new_cache.k.repeat_interleave(self.n_rep, dim=1)
v_full = new_cache.v.repeat_interleave(self.n_rep, dim=1)
attn = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=...)
\`\`\`

\`n_rep = num_heads // num_kv_heads\`。Cache 追加**窄** K/V；只在 matmul 时广播。

### 2. Why — MHA / GQA / MQA / MLA

| 布局 | 想法 | 代价 |
| --- | --- | --- |
| **MHA** | \`n_kv = n_q\` | 最强，cache 最大 |
| **GQA** | \`n_kv\` 整除 \`n_q\` | 开源中型默认 |
| **MQA** | \`n_kv = 1\` | 极致省 cache |
| **MLA** | 压缩潜变量 KV | 前沿（DeepSeek）—后部 |

改头数 ⇒ **重训**（否则对不上 checkpoint）。

### 3. How — 从零

| 步骤 | 做什么 |
| --- | --- |
| 1 | 固定 \`embed_dim = n_q × head_dim\` |
| 2 | 选能整除 \`n_q\` 的 \`n_kv\` |
| 3 | 无 bias 的 Q/K/V/O（Llama 约定） |
| 4 | Prefill 用因果；有 cache 时看全部过去 key |

### 4. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 分组合法；因果；权重头数对齐 config |
| **软** | Cache 字节约 MHA 的 \`n_kv/n_q\` |
| **软** | 同宽度下相对 MHA 质量站得住 |

### 5. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 更少 KV 头 | 解码显存 / 速度 |
| 太少 | 质量断崖 |
| MLA 路线 | 另一套压缩（Frontier） |

### 6. 本课

\`\`\`bash
python chapters/01-inference/ep06_gqa.py
\`\`\`

## 动手

\`\`\`bash
python chapters/01-inference/ep06_gqa.py
\`\`\`

## 下一章

**SwiGLU**（\`/cn/swiglu/\`）：稠密 FFN。

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

**是什么（What）：** **SwiGLU**——门控前馈：\`down(silu(gate(x)) * up(x))\`，三张**无 bias** 线性层。

**为什么（Why）：** 注意力在位置*之间*混合；FFN 在*每个*位置上变换。SwiGLU（而非纯 GELU MLP）是 Llama/Qwen 稠密默认；MoE 的 expert 常常就是同一块。

**怎么做（How）：** 选定 \`intermediate_dim\`，三张矩阵，接在 block 里第二次 pre-norm 之后。

### 1. What — 代码

[\`arche/mlp.py\`](https://github.com/chaxus/arche/blob/main/arche/mlp.py)：

\`\`\`python
class SwiGLU(nn.Module):
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
\`\`\`

gate / up 把 \`D → I\`；down 回到 \`I → D\`。无 bias——对齐 Llama 权重文件。

### 2. Why — 门控 vs 普通 MLP

| 选择 | 原因 |
| --- | --- |
| SiLU(gate)×up | 软门控；LM 上经验强 |
| 三矩阵 | 同 \`I\` 下比两矩阵 GELU MLP 参数略多 |
| 无 bias | 更省，对齐开源 Llama/Qwen 张量 |
| 按位置 | 易并行；配残差 |

### 3. How — 从零

| 步骤 | 做什么 |
| --- | --- |
| 1 | 选 \`I\`（Llama 3.2 1B 由 config 给出） |
| 2 | \`gate_proj\` / \`up_proj\` / \`down_proj\` 均 bias=False |
| 3 | 放在 \`post_attention_layernorm\` 后 |
| 4 | 以后 MoE：许多 SwiGLU + router |

### 4. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 进出维 \`D\`；无 bias；gate 上 SiLU |
| **软** | 用整模 loss / 质量评判，不孤立看 FFN |

### 5. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 更宽 \`I\` | 容量与算力 |
| GELU → SwiGLU | 需重训 |
| MoE experts | 同一砖 × E，稀疏点火 |

### 6. 本课

\`\`\`bash
python chapters/01-inference/ep07_swiglu.py
\`\`\`

## 动手

\`\`\`bash
python chapters/01-inference/ep07_swiglu.py
\`\`\`

## 下一章

**Transformer block**（\`/cn/block/\`）：把零件夹成一层。

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

**是什么（What）：** 一层可重复的 **Transformer block**——pre-RMSNorm → 注意力 → 残差 → pre-RMSNorm → SwiGLU → 残差。

**为什么（Why）：** 积木只有配上**残差**才能堆成深网。进出同形状，才能叠 \`N\` 次（Llama 3.2 ~1B：**N=16**）。

**怎么做（How）：** 三明治接一次；每层自有权重；再堆叠训或加载。

### 1. What — 前向

[\`TransformerBlock\`](https://github.com/chaxus/arche/blob/main/arche/block.py)：

\`\`\`python
h, new_cache = self.self_attn(self.input_layernorm(x), cache=cache, start_pos=start_pos)
x = x + h
x = x + self.mlp(self.post_attention_layernorm(x))
return x, new_cache
\`\`\`

\`\`\`mermaid
flowchart LR
  x[x] --> n1[RMSNorm]
  n1 --> attn[GQA_RoPE]
  attn --> add1[残差加]
  add1 --> n2[RMSNorm]
  n2 --> ffn[SwiGLU]
  ffn --> add2[残差加]
\`\`\`

### 2. Why — 残差 + pre-norm

| 选择 | 原因 |
| --- | --- |
| 残差相加 | 梯度高速路；子层≈0 时近恒等 |
| Pre-norm（非 post） | 现代深训稳定 |
| 先 attn 后 MLP | Llama/Qwen 主线（非并行 attn∥MLP） |
| 每层 KV cache | 解码复用该层过去的 K/V |

### 3. How — 从零

| 步骤 | 做什么 |
| --- | --- |
| 1 | 按 config 建 norm、\`CausalSelfAttention\`、\`SwiGLU\` |
| 2 | 保持进出 \`embed_dim\` |
| 3 | 贯通 \`cache\` / \`start_pos\` 供生成 |
| 4 | 在 stack 里重复 \`N\` 次 |

### 4. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 残差是加；pre-norm 位置对；形状 \`(B,T,D)\` |
| **硬** | 每层 cache 独立 |
| **软** | 深栈仍能训 / 加载且无 NaN |

### 5. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 更大 \`N\` | 能力与成本 |
| Post → pre-norm | 从零深训成为可能 |
| 换内脏（MLA/MoE） | 前沿变体；block 边界仍在 |

### 6. 本课

\`\`\`bash
python chapters/01-inference/ep08_block.py
\`\`\`

## 动手

\`\`\`bash
python chapters/01-inference/ep08_block.py
\`\`\`

## 下一章

**整网 + LM head**（\`/cn/stack/\`）。

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

**是什么（What）：** 完整的 **稠密因果 LM 壳**：\`ids → embed → Block×N → 最终 RMSNorm → Linear(V) → logits\`。

**为什么（Why）：** 单层 block 还不是语言模型。这就是预训练里 \`model(batch)\` 调用的东西——此前是黑盒。

**怎么做（How）：** 按 config（\`V,D,N,heads,…\`）搭建；可选 **绑定** LM head ↔ embed（小 Llama 常绑）；初始化后加载或预训练。

### 1. What — 前向

[\`Llama\`](https://github.com/chaxus/arche/blob/main/arche/model.py)：

\`\`\`python
x = self.embed_tokens(input_ids)
for layer, cache in zip(self.layers, caches):
    x, new_cache = layer(x, cache=cache, start_pos=start_pos)
x = self.norm(x)
return self.lm_head(x), new_caches
\`\`\`

Logits 形状 \`(B, T, V)\`——正是 \`lm_loss\` 错位后吃的东西。

### 2. Why — 绑定与最终 norm

| 选择 | 原因 |
| --- | --- |
| 最终 RMSNorm | 投向词表前再稳住 |
| 绑定 head↔embed | 参数更少；~1B 常见 |
| 不绑定 | 更大模型更自由 |
| 共享 config | HF 映射的唯一真相源 |

### 3. How — 从零

| 步骤 | 做什么 |
| --- | --- |
| 1 | 填好 \`Llama32Config\` |
| 2 | \`ModuleList\` 放 \`N\` 个 block |
| 3 | 在 init/load 前设 \`tie_word_embeddings\` |
| 4 | 用预训练环训，或 \`from_pretrained\` |

### 4. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | Logits \`(B,T,V)\`；config ↔ 权重；绑定一致 |
| **软** | 加载后或足够预训练步后质量可读 |

### 5. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 深度 / 宽度 / V | 结构级；重训或谨慎迁移 |
| 换成 MoE | 替换 block 内 SwiGLU——骨架仍在 |
| 只拧栈 | 没有数据/优化配合意义有限 |

### 6. 本课

\`\`\`bash
python chapters/01-inference/ep09_stack.py
\`\`\`

## 动手

\`\`\`bash
python chapters/01-inference/ep09_stack.py
\`\`\`

## 下一章

**加载权重**（\`/cn/weights/\`）：用 Llama 小数填满空壳。

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

**是什么（What）：** 通过 **键名映射** 把开源 Llama 3.2 检查点 **加载** 进你手写的栈（safetensors / HF → \`state_dict\`）。

**为什么（Why）：** 随机权重 = 图对、字胡说。官方小数让同一次前向立刻像语言模型。你已见过预训练怎样*写出*小数；本章**借用**一次跑完的结果，不必自己上集群也能生成 / 对齐。

**怎么做（How）：** 接受协议 → 下载 → 映射键 → 配对**官方分词器** → 核对形状。

### 1. What — 入口

[\`from_pretrained\`](https://github.com/chaxus/arche/blob/main/arche/load_weights.py) 建 \`Llama(cfg)\`，把 HF 张量拷进对应名字。

| 路径 | 作用 |
| --- | --- |
| **加载（本章）** | 例子 checkpoint → 马上能说话 |
| **预训练（更早）** | 这些小数从零怎么被写出 |
| **HF 对齐（稍后）** | 证明映射 + 公式对齐 HF logits |

### 2. Why — 映射要小心

| 风险 | 后果 |
| --- | --- |
| 键错 / 转置错 | 静默垃圾或崩溃 |
| 分词器错 | id 指到错误嵌入行 |
| dtype / 设备滑 | 慢或 NaN |
| 只加载一部分 | 「能跑」但层是坏的 |

### 3. How — 清单

| 步骤 | 做什么 |
| --- | --- |
| 1 | 若门禁：\`huggingface-cli login\` |
| 2 | 解析模型 id（课默认 Llama 3.2 ~1B） |
| 3 | Config → 空 \`Llama\` → 拷权重 |
| 4 | **永远**用与 checkpoint 同一套分词器 |

### 4. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 无 missing/unexpected；形状对齐 |
| **硬** | 词表大小 = 嵌入行数 |
| **软** | 短生成像人话；随后 HF 对齐 |

### 5. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 修键映射 | 解锁后续全部演示 |
| 恢复自己的 ckpt | 同一 API，你的小数 |
| 量化加载 | 见后部服务章 |

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

协议、登录、镜像——与分词器下载同类问题。

## 下一章

**文本生成**（\`/cn/generate/\`）：在 logits 上循环采样。

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

**是什么（What）：** **自回归生成**——把 logits 变成文本的循环：打分 → 选下一 id → 拼上 → 停止。

**为什么（Why）：** 训好/加载好的栈只吐分数。产品差在采样与停止规则；共同点是这条环。没有它只有「分类器形」前向，没有聊天。

**怎么做（How）：** 最后位置 logits → 可选 temperature / top-k → 贪心或采样 → 追加 → 遇 EOS 或达长度停下。KV cache（下一部分）让同一环变快。

### 1. What — 骨架

[\`arche/generate.py\`](https://github.com/chaxus/arche/blob/main/arche/generate.py) 大致：

\`\`\`python
# 每一步
logits, caches = model(ids_or_last_token, caches=caches, start_pos=...)
next_id = sample(logits[:, -1, :], temperature=..., top_k=...)  # 或 argmax
# 追加；EOS / max_new_tokens 时 break
\`\`\`

Prefill 一次跑完 prompt；开 cache 时 decode 每次只喂新 token。

### 2. Why — 旋钮 ≠ 权重

| 旋钮 | 改变的是 |
| --- | --- |
| Temperature / top-k / top-p | 挑选风格——不是模型小数 |
| 贪心 argmax | 确定、常偏闷 |
| EOS / 最大长度 | 体验与安全边界 |
| Chat template | 角色如何变成 id（后章） |

### 3. How — 从零

| 步骤 | 做什么 |
| --- | --- |
| 1 | 用**配对**分词器编码 prompt |
| 2 | 前向 → 取最后时刻 logits |
| 3 | 采样或 argmax → 追加 id |
| 4 | 直到停止；id 解码回文本 |

### 4. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 会停；只见过去；合法词表 id |
| **软** | 真权重下短补全连贯 |
| **软** | 同 prompt + 贪心 ≈ 可复现 |

### 5. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 只改采样 | 不重训也能改行为 |
| 无 KV cache | 正确但体感 O(T²)—下一部分修 |
| 约束解码 | 结构化输出 |
| 停止规则差 | 跑飞或截断 |

### 6. 本课

\`\`\`bash
python chapters/01-inference/ep11_generate.py
\`\`\`

## 动手

\`\`\`bash
python chapters/01-inference/ep11_generate.py
\`\`\`

## 下一章

**HF 对齐**（\`/cn/hf-align/\`）：证明手写栈对齐 HF。

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

**是什么（What）：** **数值核对**——同一短 prompt 分别过 arche 与 Hugging Face Llama 3.2，比较 logits（最大绝对差 + argmax 是否一致）。

**为什么（Why）：** 第 1 部分收官证明：键映射、RMSNorm、RoPE、GQA、SwiGLU、整网布局不是「形状对、公式错」。别让后面的服务/优化章把 bug 藏起来。

**怎么做（How）：** 固定 dtype；两边加载；编码一条 prompt；前向；出报告。无权重/未登录时干净跳过。

### 1. What — 报告

[\`compare_logits_to_hf\`](https://github.com/chaxus/arche/blob/main/arche/align_hf.py)：

\`\`\`python
hf_logits = hf(ids).logits
arche_logits, _ = arche(ids)
max_abs = (arche_logits - hf_logits).abs().max()
argmax_match = torch.equal(arche_logits.argmax(-1), hf_logits.argmax(-1))
\`\`\`

\`run_align_or_explain\` 在 HF 权重不可达时返回 \`None\`——**跳过 ≠ 课逻辑坏了**。

### 2. Why — 这道门

| 失败形态 | 常见原因 |
| --- | --- |
| \`max_abs\` 很大 | 映射、RoPE theta、norm 或头布局错 |
| Argmax 不一致 | 同上——外加 dtype / 设备不一致 |
| 跳过 | 未登录 / 门禁 / 离线——修环境，不是公式 |

### 3. How — 从零

| 步骤 | 做什么 |
| --- | --- |
| 1 | 需要则登录；模型 id 与加载章相同 |
| 2 | 实验用 float32 CPU 比较更稳 |
| 3 | 短 prompt（如“The cat sat”） |
| 4 | 在容差内断言，或打印跳过原因 |

### 4. 怎么样算好

| 类型 | 标准 |
| --- | --- |
| **硬** | 有权重：argmax 一致；\`max_abs\` 低于实验容差 |
| **硬** | 无权重：显式跳过，不当假通过 |
| **软** | 更长 prompt / 更多位置留给以后 CI |

### 5. 如何改进 · 对整模影响

| 杠杆 | 影响 |
| --- | --- |
| 失败 → 回头查 | 映射 / RoPE / norm，先别「优化」 |
| 通过 | 可进对齐（SFT/DPO）与服务 |
| 引擎 CI | 同类门禁卡 vLLM 级数回归 |

### 6. 本课

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
\`\`\`

第 1 部分至此结束。第 2 部分转向底座猜词能力（SFT/DPO）。

## 动手

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
\`\`\`

## 翻车

无权重/未登录 → 跳过表示「下不下来」，不是「数学错了」。

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
