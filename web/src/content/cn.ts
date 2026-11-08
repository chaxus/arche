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

### 1. 有什么作用

这一章定标本：**Llama 3.2 风格**的 next-token 栈——分词 → 嵌入 → 块堆叠 → LM head。人们常叫它 **dense 解码器**。后面所有章都挂在这条主干上。

\`\`\`mermaid
flowchart LR
  tok[分词] --> emb[嵌入]
  emb --> stack[堆叠]
  stack --> head[LM head]
\`\`\`

### 2. 为什么课程过程需要它

学大模型需要一个够小、开放、又像生产的模型，才能端到端手写。没有统一标本，后面的 RMSNorm、RoPE、GQA 会对不上同一套配置。

### 3. 怎么来的 · 怎么选

| 考量 | Llama 3.2 1B |
| --- | --- |
| 体量 | 约 1B，笔记本可碰 |
| 开放 | 权重可申请下载 |
| 积木 | RMSNorm、RoPE、SwiGLU、GQA——与很多生产 dense 一致 |
| 课程序列 | 先主干，再推理优化、小训练、前沿引物 |

### 4. 有什么标准 · 怎样算「选对了」

| 类型 | 标准 | 人话 |
| --- | --- | --- |
| **硬** | 配置尺寸能打印、自洽 | 词表、层数、维度对得上文档 |
| **软** | 能支撑整门课 | 后面章都能挂上，而不是只讲玩具公式 |
| **软** | 诚实边界 | 本课 dense 主干 + 引物，不是完整 vLLM / 全量预训练 |

### 5. 本课怎么做

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
    summary: 'id 只是编号；嵌入按行取出一串小数，后面才能算。',
    script: 'chapters/01-inference/ep03_embedding.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    demo: 'embedding',
    markdown: `## 要点

### 1. 嵌入表有什么作用

嵌入表是一张大矩阵：形状 \`(词表大小, 维度)\`。  
**第 \`i\` 行 = token id \`i\` 的那一排小数。**

它在流水线里只干一件硬任务，外加两点连带作用：

| 作用 | 人话 |
| --- | --- |
| **① 接口（硬需求）** | 把离散的 id 变成后面层能乘加的小数。没有这一步，注意力 / MLP 没有合法输入 |
| **② 可学习的「身份」** | 每个 id 在表里有固定一行；训练会改这些数，让常用、相关的 token 更好用 |
| **③ 常兼作出门打分（可选）** | 小 Llama 等把同一张表和最后的 LM head 绑在一起：进模型取行、出模型给词表打分共用 |

它**不是**词典释义，也**不是**整句理解。推理时几乎就是：**按 id 取行**。  
（快递柜直觉：格号 = id，纸条 = 那一行；本页上方可试。）

\`\`\`mermaid
flowchart LR
  text[文字] --> tok[分词 → id]
  tok --> emb[嵌入表取行]
  emb --> rest[后面各层只见小数]
\`\`\`

### 2. 为什么大模型过程里需要它

大模型内部是一串对**张量**的运算（乘矩阵、归一化、注意力……）。  
分词只给出整数编号；整数当「类别下标」可以，当「语义数值」不行——\`id=3\` 和 \`id=7\` 相加没有语言意义。

所以需要一个**固定、可学习的翻译层**：

- 输入侧：id → 小数行（嵌入）  
- 中间：只在小数上算  
- 输出侧：再变成词表上的分数 → 新的 id → 分词表解码成字  

没有嵌入表，就得另发明一种「编号 → 第一层输入」的办法；现代 decoder 几乎都选用**查表**，简单、可训、和词表一一对应。

和分词表对比：分词解决「文字怎么变成编号」；嵌入解决「编号怎么进入可计算的世界」。两张表必须配对，缺一不可。

### 3. 嵌入表是怎么来的

**不是**人给每个词手写小数，也**不是**再跑一遍 BPE。

| 阶段 | 发生什么 |
| --- | --- |
| **定规格** | 行数 = 分词词表大小；列数 = 模型维度（超参，如 2048） |
| **初始化** | 建表，填随机小数（此时几乎没意义） |
| **预训练** | 预测下一个 token；损失反传，**和别的层一起改**这些小数 |
| **存盘** | 写入检查点（\`.safetensors\` 等），和注意力、MLP 等权重放一起 |
| **你使用** | 下载官方模型 → 加载已训好的表；推理时只取行，默认不改 |
| **微调（可选）** | 继续改数，或表尾加行（新特殊符）再训 |

本课脚本用的是临时**随机小表**，只为看清取行动作。真 Llama 的嵌入表在官方权重里，和官方分词器出厂配对。

### 4. 有什么标准？怎样的嵌入表算「好」

先分清：**硬约束**（不对就坏了）和 **好不好**（训完才体现）。

**硬约束（必须满足）**

| 标准 | 人话 |
| --- | --- |
| **形状对** | 行数 = 词表大小；列数 = 模型 \`dim\`；\`T\` 个 id → \`T\` 行 |
| **与分词配对** | 第 \`i\` 行永远对应分词器的 id \`i\`；换表等于拿错格 |
| **数值可用** | 无 NaN/Inf；尺度别炸（常靠后面的 RMSNorm 等帮忙） |
| **和架构约定一致** | 若栈假定 RoPE，就不要在嵌入上再偷偷加绝对位置表（除非整栈一起改） |

**「好不好」——和分词表不同**

分词表可以单独看 fertility、语言公平等。  
嵌入表里的数是**和整网一起学出来的**，业界很少给「单张嵌入表」打一个像 fertility 那样的官方分。更常见的是：

| 看什么 | 人话 |
| --- | --- |
| **下游 / 预训练指标** | 同架构下，语言 loss、评测是否更好——这是主裁判 |
| **维度是否够用** | 太小：表达力不够；太大：参数多、更难吃数据。由架构与算力决定，不是越大越好 |
| **稀有 token** | 很少更新的行可能学不饱；词表过大时更明显 |
| **邻域直觉（辅助）** | 有时看「某行最近的其他行」是否像相关子词；只能当体检，不能当唯一分数 |
| **绑权是否合适** | 小模型绑 LM head 常更省、更稳；大模型有时解开更灵活 |

结论（对人话）：

- **合格**：形状对、和分词配对、能稳定训 / 推理。  
- **好**：在这套模型里，预训练与下游表现好——嵌入是其中一块可学习零件，**不单独颁「嵌入金奖」**。  
- 选现成模型时：用官方嵌入 + 官方分词即可，不必自己另造一张「更好的嵌入表」去硬塞。

### 5. 业界实际 · 本课怎么做

Llama 3.2 / Qwen：查表嵌入；位置交给 RoPE；小模型常绑 LM head。

\`arche/\` 的 \`TokenEmbedding\`：假表看取行；真模型约十二万行，道理相同。

## 动手

\`\`\`bash
python chapters/01-inference/ep03_embedding.py
\`\`\`

预期：词表大小、维度、ids → 输出形状；并提示位置不在嵌入里做。

## 下一章

**RMSNorm**（\`?c=rmsnorm\`）：小数进重层前先稳住尺度。

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

### 1. 有什么作用

**RMSNorm** 把每个 token 那一排小数的尺度拉回可控：先算均方根（RMS），再除以它，最后乘可学习的 \`γ\`。和 LayerNorm 不同，**不减均值**。

在 Llama 风格栈里，注意力前、MLP 前各做一次（**pre-norm**）。

### 2. 为什么大模型过程需要它

层一深，乘加会把数值越推越大或越推越小。尺度乱了，训练难收敛，推理也可能 NaN。  
归一化是「每一小段算完、进入下一块重活之前，先把尺子校准」。没有它，后面的注意力 / MLP 很难稳定叠很多层。

### 3. 怎么来的 · 怎么做

| 阶段 | 做什么 |
| --- | --- |
| **定规格** | 对最后一维做；\`γ\` 长度 = 模型维度 |
| **初始化** | \`γ\` 常初始化为 1 |
| **训练** | 和别的权重一起被梯度更新 |
| **推理** | 按公式缩放；进出形状不变 |

公式：

$$\\mathrm{RMSNorm}(x)=x/\\sqrt{\\mathrm{mean}(x^2)+\\varepsilon}\\odot\\gamma$$

### 4. 有什么标准 · 怎样算好

| 类型 | 标准 | 人话 |
| --- | --- | --- |
| **硬** | 进出同形状 | 不能改序列长度或维度 |
| **硬** | \`ε\` 防除零 | 数值稳定 |
| **硬** | 放在约定位置 | 本课是 pre-norm；别和 post-norm 混装 |
| **软** | \`γ=1\` 时每 token RMS ≈ 1 | 公式没写错的体检 |
| **软** | 训练更稳、少炸 | 最终仍看整模 loss；Norm 很少单独打分 |

### 5. 业界实际 · 本课怎么做

Llama 3.2 / Qwen dense：RMSNorm + pre-norm。  
\`arche/norms.py\` 的 \`RMSNorm\`；实验打印形状与每 token RMS。

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

### 1. 有什么作用

**RoPE（旋转位置编码）**给注意力里的 Q、K 按**位置**做旋转。点积里会自然带上相对距离信息，而不必在嵌入上再贴一张大位置表。

### 2. 为什么大模型过程需要它

自注意力本身对顺序不敏感：打乱 token 顺序，若没有位置信号，分数可以一样。语言却依赖顺序（「狗咬人」≠「人咬狗」）。  
嵌入章只回答「是谁」；RoPE 回答「在第几位」。现代 dense 解码器把位置写进 Q/K，而不是加在嵌入上。

### 3. 怎么来的 · 怎么做

| 阶段 | 做什么 |
| --- | --- |
| **定规格** | 选 \`rope_theta\` 等；为各位置缓存 cos/sin |
| **前向** | 对 Q/K 做 \`rotate_half\`，再与 cos/sin 组合 |
| **训练 / 推理** | 同一套旋转；更长上下文可能要缩放（后面专章） |

本页可拖互动图看角度如何随位置变。

### 4. 有什么标准 · 怎样算好

| 类型 | 标准 | 人话 |
| --- | --- | --- |
| **硬** | Q/K 形状不变 | 只改相位，不改头数/维度 |
| **硬** | 相对位置进注意力 | 交换两位置，分数应跟着变 |
| **硬** | 与检查点约定一致 | \`theta\`、是否缩放必须匹配权重 |
| **软** | 训练长度内行为稳定 | 主裁判仍是整模质量 |
| **软** | 超长时是否要缩放 | 另见 RoPE 缩放章 |

### 5. 业界实际 · 本课怎么做

Llama 3 / 3.2：RoPE + 很大的 \`rope_theta\`。绝对可学习 PE、正弦 PE、ALiBi 是其他路线。  
\`arche/rope.py\`；实验应用 RoPE 并核对形状。

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

### 1. 有什么作用

**GQA（分组查询注意力）**让多组 Q 头**共用**更少的 KV 头。仍是因果注意力（位置 \`t\` 只看 \`≤ t\`），但解码时要存的 K/V 更少。

流程：投影 → 拆头 → RoPE → 扩 KV → SDPA → 输出投影。见本页图。

### 2. 为什么大模型过程需要它

完整 MHA：每个 Q 头自带 KV → **KV 缓存**随层数、长度、头数涨得很快，解码显存贵。  
GQA 在「质量」和「缓存体积」之间折中：比 MQA（全体共用一套 KV）更稳，比满 MHA 更省。

### 3. 怎么来的 · 怎么做

| 阶段 | 做什么 |
| --- | --- |
| **定规格** | 选 \`n_q\`、\`n_kv\`（须整除）；\`n_rep = n_q / n_kv\` |
| **实现** | KV 按组广播/重复后再与 Q 做注意力 |
| **训练** | 与整网一起学投影权重 |
| **推理** | Prefill / decode 都走同一套头布局 + 缓存 |

### 4. 有什么标准 · 怎样算好

| 类型 | 标准 | 人话 |
| --- | --- | --- |
| **硬** | \`n_q\` 能被 \`n_kv\` 整除 | 分组合法 |
| **硬** | 因果 mask 正确 | 不能偷看未来 |
| **硬** | 与权重头数一致 | 装错头数会形状炸或静默错 |
| **软** | 缓存比满 MHA 明显更小 | GQA 的直接收益 |
| **软** | 质量别相对 MHA 塌光 | 用评测/loss 看，不是单独「GQA 分」 |

### 5. 业界实际 · 本课怎么做

Llama 3.2 1B：**32 Q / 8 KV**。很多 7B–70B 也为解码显存用 GQA。  
\`arche/attention.py\` 的 \`CausalSelfAttention\`；实验打印头数与形状。

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

### 1. 有什么作用

注意力混合**各位置**的信息；每个位置还要各自做一次非线性变换——这就是 **FFN / MLP**。  
Llama 风格用 **SwiGLU**：三条无偏置线性（gate / up / down），门控后相乘再投回原维度。

$$\\\\mathrm{FFN}(x)=W_d\\\\big(\\\\mathrm{SiLU}(W_g x)\\\\odot W_u x\\\\big)$$

### 2. 为什么大模型过程需要它

只有注意力，每个位置主要在「读别人」。模型还需要在**当前位置**上做丰富的特征变换（记忆事实、模式等常和 FFN 相关的讨论）。  
SwiGLU 是今天 dense 解码器里常见的门控 FFN，替代老栈的 GELU MLP。

### 3. 怎么来的 · 怎么做

| 阶段 | 做什么 |
| --- | --- |
| **定规格** | 隐层更宽（如 \`ffn_dim\`）；三套权重无 bias |
| **初始化** | 随模型初始化 |
| **训练** | 与整网一起更新 |
| **变体** | MoE 里每个专家往往就是一个 SwiGLU（后面章） |

### 4. 有什么标准 · 怎样算好

| 类型 | 标准 | 人话 |
| --- | --- | --- |
| **硬** | 进出 \`embed_dim\` | 中间可变宽，两头对齐 |
| **硬** | 无 bias（Llama 约定） | 与权重布局一致 |
| **软** | 相对 GELU MLP 的收益 | 看整模；不单独给 FFN 打「金奖」 |

### 5. 业界实际 · 本课怎么做

现代 dense：SwiGLU 或近亲门控。  
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

### 1. 有什么作用

**Transformer 块**是可重复的一层「三明治」：把 RMSNorm、注意力、残差、再 RMSNorm、SwiGLU、残差捆成一个单元。进出同形状，方便叠 \`N\` 次。

\`\`\`
x → RMSNorm → 注意力 → + 残差
  → RMSNorm → SwiGLU → + 残差
\`\`\`

### 2. 为什么大模型过程需要它

前面零件单独成立，但语言模型要**深度**：同一套结构重复多次，才能一层层混合上下文、变换特征。  
「块」定义了重复的边界：残差保证有一条「原始信号」通路；pre-norm 让每块入口尺度可控。

### 3. 怎么来的 · 怎么做

| 阶段 | 做什么 |
| --- | --- |
| **组装** | 按上图接线；注意力可接 KV cache |
| **定深度** | Llama 3.2 1B：**16** 块 |
| **训练** | 每块有自己的一套权重，一起更新 |

### 4. 有什么标准 · 怎样算好

| 类型 | 标准 | 人话 |
| --- | --- | --- |
| **硬** | 残差路径形状不变 | \`x\` 与子层输出可相加 |
| **硬** | pre-norm 位置正确 | 与 Llama 约定一致 |
| **硬** | 有 cache 时每层各管各的 | 别串层、别漏层 |
| **软** | 加深后仍可训 | 靠残差 + norm；深度由架构定 |

### 5. 业界实际 · 本课怎么做

Llama 风格 pre-norm 块是开放 dense 主干。  
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

### 1. 有什么作用

**堆叠**把嵌入、\`N\` 个块、最终 RMSNorm、LM head 串成完整 next-token 模型：

\`ids → 嵌入 → Block×N → 最终 RMSNorm → Linear(词表) → logits\`

最后一维分数 = 下一个 token 在词表上的 **logits**。小 Llama 常 **绑** LM head ↔ 嵌入。

### 2. 为什么大模型过程需要它

单块只是一层变换；语言模型要「读完整段上下文 → 给词表打分」。  
堆叠提供深度；LM head 把隐藏状态映射回**与分词器同一词表**的分数，才能采样出下一个 id。

### 3. 怎么来的 · 怎么做

| 零件 | 作用 |
| --- | --- |
| 嵌入 | id → 向量 |
| 块 × N | 上下文混合 + 逐位变换 |
| 最终 norm | 进 head 前稳住尺度 |
| LM head | 向量 → 词表分（可与嵌入绑权） |

配置决定深度、维度、头数；权重随机时图是对的，但输出无意义（下一章加载）。

### 4. 有什么标准 · 怎样算好

| 类型 | 标准 | 人话 |
| --- | --- | --- |
| **硬** | logits 形状 \`(B, T, vocab)\` | 与词表对齐 |
| **硬** | 绑权约定一致 | 声称绑就要共享同一参数 |
| **硬** | 块数 / 维度与配置一致 | 否则装不上官方权重 |
| **软** | 装权重后文字是否通顺 | 整模质量，不是「堆叠分」 |

### 5. 业界实际 · 本课怎么做

Llama 风格 dense 端到端即此骨架。  
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

### 1. 有什么作用

**加载权重**把已经训好的张量，按名字/映射拷进我们搭好的模块。图对了但参数随机 → 输出无意义；装上官方 Llama 3.2 权重，同一条前向才开始像语言模型。

### 2. 为什么大模型过程需要它

训练产生的「知识」存在检查点里（嵌入表、每层投影、norm 的 \`γ\`……）。  
推理 / 对齐 / 微调都从**正确加载**开始。装错 key、装错分词、装错精度，都会静默或炸裂。

### 3. 怎么来的 · 怎么做

| 步骤 | 做什么 |
| --- | --- |
| **1** | \`from_pretrained\` 思路：按配置建空模块 |
| **2** | 下载 HF Llama 3.2 检查点（许可 + 登录） |
| **3** | 按 **key 映射**把张量拷进 arche 模块名 |
| **4** | 与官方分词器配对使用 |

### 4. 有什么标准 · 怎样算好

| 类型 | 标准 | 人话 |
| --- | --- | --- |
| **硬** | 参数量 / 形状对得上 | 少层、错维会立刻暴露或更糟地静默 |
| **硬** | key 映射覆盖该有的层 | 漏映射 = 那一层仍是随机 |
| **硬** | 分词器配对 | 见分词 / 嵌入章 |
| **软** | 与 HF 同 prompt 的 logits 接近 | 对齐章专门验 |

### 5. 业界实际 · 本课怎么做

官方 HF Llama 3.2 检查点。  
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

### 1. 有什么作用

**生成**把「一次前向 → 下一个 id」做成循环：取最后位置 logits →（可选）温度 / top-k → 贪心或采样 → 拼到序列上 → 直到 EOS 或达到长度。

### 2. 为什么大模型过程需要它

模型前向一次只给出**当前位置的下一 token 分布**，不会一次性吐出整段回复。  
聊天、补全、工具调用，底层都是这个循环；产品差异多在停止条件、采样与模板。

### 3. 怎么来的 · 怎么做

| 步骤 | 做什么 |
| --- | --- |
| **1** | Prefill / 前向得到 logits |
| **2** | 取最后位置；可选 temperature、top-k |
| **3** | argmax 或采样得到新 id |
| **4** | append；遇 EOS 或长度上限则停 |

KV 缓存（下一章）让第 1 步在 prefill 后变便宜，但**循环逻辑不变**。

### 4. 有什么标准 · 怎样算好

| 类型 | 标准 | 人话 |
| --- | --- | --- |
| **硬** | 能停 | EOS / 最大长度生效 |
| **硬** | 只基于已生成前缀 | 不偷看未来 |
| **软** | 温度 / top-k 行为符合预期 | 0 更贪心，升高更随机 |
| **软** | 有真权重时文字连贯 | 整模 + 采样；不是「循环代码金奖」 |

### 5. 业界实际 · 本课怎么做

所有聊天栈同一类循环。  
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
