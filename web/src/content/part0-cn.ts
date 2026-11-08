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

**是什么（What）：** 本课——**一条从零对接前沿的路**，以 Llama 3.2 ~1B 为稠密例子。

**为什么（Why）：** 术语先行容易迷路。需要可走完的脊柱：建表 → 猜词训练 → 手写主干 → 对齐 → 服务 → MoE/MLA 地图。

**怎么做（How）：** 读左侧章节；跑每个实验；「标记已读」只在本机。仓库：[github.com/chaxus/arche](https://github.com/chaxus/arche)

**例子：** Llama 3.2 ~1B = 前沿 dense 积木（RMSNorm、RoPE、GQA、SwiGLU、字节 BPE）× 可学规模。官方分词 + 权重 = 验收用手写栈的成品一对。

| 阶段 | 人话 |
| --- | --- |
| **0 开场** | 环境；读一章；跑一条命令 |
| **1 表 + 训练 + 积木** | 分词 → 嵌入 → 目标→数据→优化→规模 → RMSNorm…HF |
| **2 对齐** | SFT / DPO |
| **3 服务加速** | KV、长上下文、设备/量化、连续批、分页 |
| **4 前沿** | MoE + MLA 地图、检查点、分片 |

课内 = **能跑的原理 + 前沿地图**。完整引擎 / 超大集群 / 多模态 = 带着地图再爬。

## 下一章

**大模型在干什么**（\`/cn/start-llm-idea/\`）。
`,
  },
  {
    id: 'start-llm-idea',
    ep: '00b',
    title: '大模型在干什么',
    summary: '推理在猜下一个词；训练用对错改写小数——这才是权重从哪来。',
    script: '',
    partId: 'start',
    estimatedMinutes: 5,
    markdown: `## 要点

**是什么（What）：** 推理时大模型在**猜下一个 token**；训练时用对错**改写小数**学会这件事。

**为什么（Why）：** 聊天界面把回路藏起来了。权重不是魔法文件——是海量 next-token 练习（再可选对齐）的结果。

**怎么做（推理）：** 给词表每个 id 打分 → 选出（argmax/采样）→ 拼上 → 重复。

**怎么做（训练）：** 真下一 id 已知 → 交叉熵 → 反传 → 嵌入行和主干权重移动。嵌入之后：[\`/cn/pretrain/\`](/cn/pretrain/) → 数据 → 优化 → 规模 → 再写 dense 积木。

三个名字：**token** = 小块；**embedding** = 一排小数；**logits** = 下一 id 的分数。

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

**是什么（What）：** 一套能跑后续实验的 Python 环境。

**为什么（Why）：** 没有 venv + 依赖，「跑本章」在碰到模型想法之前就会失败。初学路径不需要显卡。

**怎么做（How）：** Python ≥ 3.10 → venv → \`pip install -r requirements.txt\` → 检查脚本打印 \`OK\`。

## 动手

\`\`\`bash
git clone https://github.com/chaxus/arche.git && cd arche
python3 -m venv .venv
source .venv/bin/activate          # Windows: .\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python chapters/00-start/ep00_check_setup.py
\`\`\`

没有 git 就从 GitHub 下 ZIP。

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

**是什么（What）：** 每个实验默认你会的几条壳命令：在哪、列文件、进仓库、激活 venv。

**为什么（Why）：** 早期翻车多半是「目录错了」或「没开 venv」，不是数学错了。

**怎么做（How）：**

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

**是什么（What）：** 站点怎么展示代码——库实现（\`arche/\`）vs 实验脚本（\`chapters/...\`）。

**为什么（Why）：** 第一天靠跑和读学，不靠默写 Python。知道往哪看，胜过背 API。

**怎么做（How）：** 每章打开 **代码** 卡片：

1. **库实现** — 本课对应的模块  
2. **实验脚本** — 你要跑的文件  

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

**是什么（What）：** 一页看清五段路，以及何时需要 Hugging Face 登录。

**为什么（Why）：** 没有地图，第 1 部分会像随机积木。登录只在加载/对齐官方 Llama 时需要——教学 BPE 与多数形状实验不需要。

**怎么做（How）：** 跟 Previous/Next；心里留着这张图：

\`\`\`mermaid
flowchart LR
  p0[0 开场] --> p1[1 表_训练_积木]
  p1 --> p2[2 对齐]
  p2 --> p3[3 服务加速]
  p3 --> p4[4 前沿]
\`\`\`

**一条路：** 分词 → 嵌入 → **猜词训练**（目标→数据→优化→规模）→ dense 积木 → 加载/对照 Llama → 对齐 → 服务 → 前沿地图。

- **贯穿例子：** Llama 3.2 ~1B。  
- **早点抓住的核心：** 预训练写出嵌入行和主干权重。  
- **HF 登录**仅用于官方 Llama 加载 / HF 对齐。

## 下一章

第 1 部分：**总览**（\`/cn/overview/\`）。
`,
  },
];
