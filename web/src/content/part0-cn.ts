import type { Chapter } from './types';

/** 第 0 部分 — 零基础，刻意写短。 */
export const PART0_CN: Chapter[] = [
  {
    id: 'start-welcome',
    ep: '00a',
    title: '欢迎',
    summary: '从零开始：大模型怎么写出来、怎么用、前沿在忙什么。',
    script: '',
    partId: 'start',
    estimatedMinutes: 5,
    markdown: `## 要点

**你会学到什么：** 把大模型当成**从零训到能对接业界前沿**来学——分词表/嵌入表怎么生成、怎样算好、如何改进及对整模的影响；主干怎么手写到能跑；再学推理加速、一点点训练，以及 MoE/MLA 等前沿地图。不需要先懂术语。

**一条路，一个例子。** 你从零写的是**当前前沿 dense 配方**：**Llama 3.2 ~1B**——Meta Llama 3 系列的开放小规格同胞。大规格用的是同一套架构家族：RMSNorm、RoPE、GQA、SwiGLU、字节 BPE——今天开源 dense（以及不少闭源栈）仍在交付的积木，不是 GPT-2 课堂玩具。选 **~1B** 是为了规模可学、可下；**架构**才是重点。官方分词 + 权重 = 这份配方的成品一对，用来验收你的手写栈。

**整条路（五段）：**

| 阶段 | 人话 | 你会碰到的名字（后面再熟） |
| --- | --- | --- |
| **0 开场** | 装好环境，学会读一章、跑一条命令 | — |
| **1 写出 dense 本体** | 自训小分词/嵌入 → 手写 RMSNorm、RoPE、GQA、SwiGLU… → 生成 → 用 Llama 例子对照（HF） | dense 解码器 |
| **2 从零训练** | 用玩具数据感受「预训练 / SFT / DPO」怎样写出权重 | pretrain、SFT、DPO |
| **3 生成与服务加速** | KV 缓存、长上下文、设备与量化、连续批、分页 | KV、RoPE 缩放、Paged KV… |
| **4 前沿结构** | MoE + MLA 地图（DeepSeek-V3 向开源栈）、检查点 | MoE、MLA、分片 |

**学完你大概能：** 自己跑通实验；看懂同类开源代码大概在哪一层；再去读论文或 vLLM 一类仓库时知道从哪切入。课内是**能跑的原理 + 前沿地图**；完整工业引擎、超大集群、多模态是下一层，带着地图去接，不用从零猜。

站点：左侧章节；底部上一章 / 下一章；「标记已读」只在本机。仓库：[github.com/chaxus/arche](https://github.com/chaxus/arche)

## 下一章

**大模型在干什么**（\`/cn/start-llm-idea/\`）。
`,
  },
  {
    id: 'start-llm-idea',
    ep: '00b',
    title: '大模型在干什么',
    summary: '一句话：反复猜下一个小词块。',
    script: '',
    partId: 'start',
    estimatedMinutes: 4,
    markdown: `## 要点

大模型（LLM）根据已经写出的内容，反复猜**下一个小词块**：

1. 给词表里每个候选打分  
2. 选出一个（总选最高分，或按概率抽）  
3. 拼上，再重复  

后面会用到的三个名字：**token** = 文本切成的小块；**embedding** = 小块变成的一串数字（向量）；**logits** = 下一个小块的分数。

\`\`\`mermaid
flowchart LR
  text[文字] --> tok[Tokens]
  tok --> vec[向量]
  vec --> stack[层层计算]
  stack --> scores[分数]
  scores --> pick[下一个]
  pick --> text
\`\`\`

## 下一章

**配置电脑**（\`/cn/start-setup/\`）。
`,
  },
  {
    id: 'start-setup',
    ep: '00c',
    title: '配置电脑',
    summary: 'Python 3.10+、venv、装依赖、跑检查。',
    script: 'chapters/00-start/ep00_check_setup.py',
    partId: 'start',
    estimatedMinutes: 15,
    markdown: `## 要点

需要 Python ≥ 3.10 和虚拟环境。不需要显卡。

## 动手

\`\`\`bash
git clone https://github.com/chaxus/arche.git && cd arche
python3 -m venv .venv
source .venv/bin/activate          # Windows: .\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python chapters/00-start/ep00_check_setup.py
\`\`\`

最后一行应是 \`OK\`。没有 git 就从 GitHub 下 ZIP。

## 翻车

\`No module named torch\` → 没激活 venv 或没 pip。网络失败 → 重试 / PyPI 镜像。

## 下一章

**终端保命**（\`/cn/start-terminal/\`）。
`,
  },
  {
    id: 'start-terminal',
    ep: '00d',
    title: '终端保命',
    summary: 'cd、列目录、先看报错最后几行。',
    script: 'chapters/00-start/ep00_check_setup.py',
    partId: 'start',
    estimatedMinutes: 5,
    markdown: `## 要点

| 目的 | macOS / Linux | Windows |
| --- | --- | --- |
| 在哪 | \`pwd\` | \`pwd\` |
| 列表 | \`ls\` | \`dir\` |
| 进目录 | \`cd arche\` | \`cd arche\` |

报错先看 traceback **最后一段**。

## 动手

\`\`\`bash
cd arche && source .venv/bin/activate
python chapters/00-start/ep00_check_setup.py
python chapters/01-inference/ep01_overview.py   # 可选
\`\`\`

## 下一章

**怎么读脚本**（\`/cn/start-reading-code/\`）。
`,
  },
  {
    id: 'start-reading-code',
    ep: '00e',
    title: '怎么读脚本',
    summary: '认出 import / main / print 即可。',
    script: 'chapters/00-start/ep00_check_setup.py',
    partId: 'start',
    estimatedMinutes: 4,
    markdown: `## 要点

本页下方会直接展示实验脚本。后面章节看 **代码** 卡片：

1. **库实现** — 正在学的 \`arche/\` 模块  
2. **实验脚本** — 你要跑的 \`chapters/...\`  

在脚本里认出：说明 → \`import\` → \`def main\` → \`print\` → \`if __name__\`。

## 下一章

**旅程地图**（\`/cn/start-map/\`）。
`,
  },
  {
    id: 'start-map',
    ep: '00f',
    title: '旅程地图',
    summary: '一条从零主线，Llama 作例子；何时需要 HF 登录。',
    script: '',
    partId: 'start',
    estimatedMinutes: 3,
    markdown: `## 要点

\`\`\`mermaid
flowchart LR
  p0[0 开场] --> p1[1 写出本体]
  p1 --> p2[2 从零训练]
  p2 --> p3[3 服务加速]
  p3 --> p4[4 前沿结构]
\`\`\`

**一条路：** 从零 → 手写主干 → 训练（体会权重怎么写出）→ 服务加速 → 前沿地图。

- **贯穿例子：** Llama 3.2 ~1B（配置数字；官方分词/权重是这个例子的成品一对）。  
- **教学 BPE / 玩具训练：** 表和小数怎么来——不是第二条课。  
- **HF 登录**在加载或对齐官方 Llama 文件时需要；形状实验与教学 BPE 不需要。

## 下一章

第 1 部分：**总览**（\`/cn/overview/\`）。
`,
  },
];
