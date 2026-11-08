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

我们用 **Llama 3.2** 约 1B 当标本：零件与生产 dense 一致；官方表与权重演示「业界成品一对」；教学脚本则演示「从零生成」时你自己要做的事。

**整条路（五段）：**

| 阶段 | 人话 | 你会碰到的名字（后面再熟） |
| --- | --- | --- |
| **0 开场** | 装好环境，学会读一章、跑一条命令 | — |
| **1 模型本体** | 把「读词 → 算一层层 → 猜下一个词」每一块自己写出来 | 人们常说的 dense 解码器：RMSNorm、RoPE、注意力、SwiGLU… |
| **2 算得更快** | 别每次从头算；一批请求怎么排；模型怎么压小一点 | KV 缓存、连续批、量化、对话模板 |
| **3 练一练** | 用玩具数据感受「训练 / 跟人对齐」在优化什么 | pretrain、SFT、DPO |
| **4 更前沿的招** | 专家混合、压缩注意力、分页显存等——先摸形状当地图 | MoE、MLA、Paged KV、CUDA Graph |

**学完你大概能：** 自己跑通实验；看懂同类开源代码大概在哪一层；再去读论文或 vLLM 一类仓库时知道从哪切入。课内是**能跑的原理 + 前沿地图**；完整工业引擎、超大集群、多模态是下一层，带着地图去接，不用从零猜。

站点：左侧章节；底部上一章 / 下一章；「标记已读」只在本机。仓库：[github.com/chaxus/arche](https://github.com/chaxus/arche)

## 下一章

**大模型在干什么**（\`?c=start-llm-idea\`）。
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

**配置电脑**（\`?c=start-setup\`）。
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

**终端保命**（\`?c=start-terminal\`）。
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

**怎么读脚本**（\`?c=start-reading-code\`）。
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

**旅程地图**（\`?c=start-map\`）。
`,
  },
  {
    id: 'start-map',
    ep: '00f',
    title: '旅程地图',
    summary: '第 1–4 部分；何时需要 HF 登录。',
    script: '',
    partId: 'start',
    estimatedMinutes: 3,
    markdown: `## 要点

\`\`\`mermaid
flowchart LR
  p0[0 开场] --> p1[1 写出模型]
  p1 --> p2[2 算得更快]
  p2 --> p3[3 练一练]
  p3 --> p4[4 更新的招]
\`\`\`

- **目标：** 写出「下一个词怎么来」→ 学会加速与一点点训练 → 摸到更前沿招数的形状。  
- **标本：** Llama 3.2 小版本（约 10 亿参数）；零件名后面再熟。  
- **HF 登录**只在要下载官方分词器/权重时需要；多数实验用随机小数就行。

## 下一章

第 1 部分：**总览**（\`?c=overview\`）。
`,
  },
];
