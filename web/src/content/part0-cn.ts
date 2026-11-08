import type { Chapter } from './types';

/** 第 0 部分 — 零基础，刻意写短。 */
export const PART0_CN: Chapter[] = [
  {
    id: 'start-welcome',
    ep: '00a',
    title: '欢迎',
    summary: '这课是什么、怎么往下走。',
    script: '',
    partId: 'start',
    estimatedMinutes: 2,
    markdown: `## 要点

**arche** 用手写讲清 **Llama 3.2 风格**的 dense 解码器（不做 GPT-2 考古）。

- 左侧是章节（第 0→4 部分）；底部「上一章 / 下一章」。
- 「标记已读」只存在本机浏览器。
- 实验在 \`chapters/\` 的 Python 脚本里——先走完第 0 部分再跑。

仓库：[github.com/chaxus/arche](https://github.com/chaxus/arche)

## 下一章

**大模型在干什么**（\`?c=start-llm-idea\`）。
`,
  },
  {
    id: 'start-llm-idea',
    ep: '00b',
    title: '大模型在干什么',
    summary: '一句话：预测下一个 token。',
    script: '',
    partId: 'start',
    estimatedMinutes: 4,
    markdown: `## 要点

LLM 根据前文反复选**下一个 token**：

1. 给词表里每个候选打分  
2. 选出一个（贪心或采样）  
3. 拼上，再重复  

**Token** = 文本块；**Embedding** = 向量；**Logits** = 下一 token 的分数。

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

**翻车：** \`No module named torch\` → 没激活 venv 或没 pip。网络失败 → 重试 / PyPI 镜像。

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

打开 \`chapters/00-start/ep00_check_setup.py\`，找到：

1. 顶部说明 — 怎么跑  
2. \`import\` — 借用工具  
3. \`def main\` — 步骤  
4. \`print\` — 输出  
5. \`if __name__ == "__main__"\` — 入口  

先会读、会跑就够。

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
  p0[0 开场] --> p1[1 Dense 主干]
  p1 --> p2[2 更快推理]
  p2 --> p3[3 小训练]
  p3 --> p4[4 前沿]
\`\`\`

- **Llama 3.2 谱系** = RMSNorm、RoPE、SwiGLU、GQA（目标 1B）。
- **HF 登录**只在门禁分词器/权重章需要；多数 demo 用随机小张量。

## 下一章

第 1 部分：**总览**（\`?c=overview\`）。
`,
  },
];
