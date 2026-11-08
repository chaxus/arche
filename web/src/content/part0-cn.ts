import type { Chapter } from './types';

/** 第 0 部分 — 零基础（中文） */
export const PART0_CN: Chapter[] = [
  {
    id: 'start-welcome',
    ep: '00a',
    title: '欢迎',
    summary: 'arche 是什么、网站怎么用、整条路大概要多久。',
    script: '',
    partId: 'start',
    estimatedMinutes: 8,
    markdown: `## 一句话

你会通过「人话讲解 + 短 Python 脚本」搞清现代语言模型怎么工作——**现在还不需要会写 Python**。

## 开始之前

- 阅读约 8 分钟。
- 需要一台能上网的电脑，后面愿意复制粘贴命令即可。
- 不要求机器学习或编程课基础。

## 用人话讲

**arche**（ἀρχή）意为「本原 / 第一原理」。这个项目手写的是**当下**开源模型族（Llama 3.2 风格）里真正重要的零件，而不是 GPT-2 考古之旅。

**网站**是教室；\`chapters/\` 下的 **Python 文件**是实验单。右上角可切换中/英；顶部按钮可「标记已读」（只存在你自己的浏览器里）。

底部有「上一章 / 下一章」，请尽量按顺序走。

## 对应代码

现在还不用跑代码。仓库：[github.com/chaxus/arche](https://github.com/chaxus/arche)。

## 现在就做

1. 看左侧导航：最上面应是 **第 0 部分 — 从这里开始**。
2. 看懂后点 **标记已读**。
3. 用 **下一章** 继续。

## 如果翻车了

- 页面空白：刷新一次；进度在本地浏览器存储里。
- GitHub 要登录：公开文件不登录也能看。

## 先别深挖

论文、显卡、公式、「从零训练全网数据」。我们走的是教学路径。

## 下一章

**大模型在干什么**（\`?c=start-llm-idea\`）。
`,
  },
  {
    id: 'start-llm-idea',
    ep: '00b',
    title: '大模型在干什么',
    summary: '用人话讲「预测下一个 token」——暂时不公式。',
    script: '',
    partId: 'start',
    estimatedMinutes: 12,
    markdown: `## 一句话

大语言模型（LLM）会根据已经出现的内容，反复猜测**下一块文本**。

## 开始之前

- 先读完「欢迎」。
- 约 12 分钟。

## 用人话讲

把它想成特别有耐心的输入法联想：

1. 已有一段文字。
2. 模型给词表里每个候选打分：「下一个更像谁？」
3. 选出一个（有时取最高分，有时在高分里随机）。
4. 拼到后面，再回到第 2 步。

三个以后会反复出现的词：

- **Token**：模型眼里的「文本积木」（可能是词，也可能是词的一段）。
- **Embedding（嵌入）**：用一串数字表示一个 token。
- **Logits / 分数**：对「下一个 token」的原始偏好。

现在**不需要公式**。先记住循环：*看前文 → 打分 → 选一个 → 拼上 → 重复*。

\`\`\`mermaid
flowchart LR
  text[文字] --> tok[Tokens]
  tok --> vec[向量]
  vec --> stack[一层层计算]
  stack --> scores[下一个的分数]
  scores --> pick[选出一个]
  pick --> text
\`\`\`

## 对应代码

后面每一章实现图上的一块。此刻只要有图就够。

## 现在就做

对自己说一句：「LLM 是在根据前文预测下一个 token。」觉得说得通，就可以继续。

## 如果翻车了

- token 仍抽象：想成乐高积木，不是地铁票。
- 图没渲染：上面的文字本身已完整。

## 先别深挖

训练数据、RLHF、Agent、聊天产品。它们都架在这个循环之上。

## 下一章

**配置你的电脑**（\`?c=start-setup\`）。
`,
  },
  {
    id: 'start-setup',
    ep: '00c',
    title: '配置你的电脑',
    summary: '安装 Python、克隆仓库、建虚拟环境、装依赖。',
    script: 'chapters/00-start/ep00_check_setup.py',
    partId: 'start',
    estimatedMinutes: 25,
    markdown: `## 一句话

安装 Python 3.10+，下载本项目，建隔离环境，再安装依赖库。

## 开始之前

- 约 15–40 分钟（看网速）。
- macOS / Windows / Linux 都行；**不需要**显卡。

## 用人话讲

Python 是实验脚本的语言。**venv（虚拟环境）**像一个私人工具箱，避免和电脑上别的项目抢包装。

## 对应代码

装好后运行：

\`\`\`bash
python chapters/00-start/ep00_check_setup.py
\`\`\`

应看到 \`OK — continue...\`。

## 现在就做

### 1) 安装 Python 3.10 或更新

- **macOS：** [python.org](https://www.python.org/downloads/) 或 \`brew install python\`
- **Windows：** [python.org](https://www.python.org/downloads/)，勾选 **Add python.exe to PATH**
- **Linux：** 包管理器安装（如 \`sudo apt install python3 python3-venv python3-pip\`）

检查：

\`\`\`bash
python3 --version
\`\`\`

（Windows 上可能是 \`python\`。）

### 2) 打开终端

- **macOS：**「终端」或 iTerm
- **Windows：** Windows Terminal 或 PowerShell
- **Linux：** 常用终端即可

### 3) 克隆仓库

\`\`\`bash
git clone https://github.com/chaxus/arche.git
cd arche
\`\`\`

没有 git：从 GitHub 下 ZIP，解压后 \`cd\` 进去。

### 4) 创建并激活 venv

macOS / Linux：

\`\`\`bash
python3 -m venv .venv
source .venv/bin/activate
\`\`\`

Windows（PowerShell）：

\`\`\`powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
\`\`\`

提示符前应出现 \`(.venv)\`。

### 5) 安装依赖

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 6) 信心检查

\`\`\`bash
python chapters/00-start/ep00_check_setup.py
\`\`\`

## 如果翻车了

- 找不到 \`python3\`：装好 Python 后重开终端。
- Windows 激活被策略拦截：按微软文档放行，或用 \`cmd\` + \`.venv\\Scripts\\activate.bat\`。
- \`pip\` 因网络失败：重试；国内可换 PyPI 镜像。
- torch 很大：等它下完；CPU 版本够用。

## 先别深挖

CUDA、conda、Docker、Hugging Face 登录（要用官方权重的章节再讲）。

## 下一章

**终端保命**（\`?c=start-terminal\`）。
`,
  },
  {
    id: 'start-terminal',
    ep: '00d',
    title: '终端保命',
    summary: 'cd、列目录、读报错，并跑通第一个检查脚本。',
    script: 'chapters/00-start/ep00_check_setup.py',
    partId: 'start',
    estimatedMinutes: 15,
    markdown: `## 一句话

终端是打字下命令的窗口；学 arche 只需其中很少几条。

## 开始之前

- 已能激活 \`.venv\`，且 \`ep00_check_setup.py\` 打印 OK。
- 约 15 分钟。

## 用人话讲

常用命令：

| 目的 | macOS / Linux | Windows（PowerShell） |
| --- | --- | --- |
| 我在哪？ | \`pwd\` | \`pwd\` |
| 有哪些文件 | \`ls\` | \`dir\` 或 \`ls\` |
| 进入目录 | \`cd arche\` | \`cd arche\` |
| 返回上层 | \`cd ..\` | \`cd ..\` |

**traceback（堆栈）**是 Python 说「我崩了」。先看**最后一段**，那里通常才是真正原因。

## 对应代码

\`\`\`bash
python chapters/01-inference/ep01_overview.py
\`\`\`

（也可等到第 1 部分再跑；提前跑能增加信心。）

## 现在就做

1. \`cd\` 到 \`arche\` 目录。
2. 激活 \`.venv\`。
3. 再跑一次 \`python chapters/00-start/ep00_check_setup.py\`。
4. 可选：跑 \`ep01_overview.py\`，它打印的是配置数字，不是聊天机器人。

## 如果翻车了

- \`No module named torch\`：没激活 venv，或没装 requirements。
- \`can't open file\`：目录不对；\`ls\`/\`dir\` 应能看到 \`chapters\` 和 \`arche\`。
- 求助时请复制**最后几行**报错。

## 先别深挖

Shell 脚本、git 分支、复杂编辑器配置。

## 下一章

**怎么读章节脚本**（\`?c=start-reading-code\`）。
`,
  },
  {
    id: 'start-reading-code',
    ep: '00e',
    title: '怎么读章节脚本',
    summary: '认出 import、main、print——先不用会写。',
    script: 'chapters/00-start/ep00_check_setup.py',
    partId: 'start',
    estimatedMinutes: 12,
    markdown: `## 一句话

在会写 Python 之前，你只要**认出脚本里的路标**就能跟上 arche。

## 开始之前

- 终端保命已完成。
- 约 12 分钟。
- 用任意文本编辑器打开 \`chapters/00-start/ep00_check_setup.py\`。

## 用人话讲

找这些地标：

1. 文件开头的三引号说明——给人看的运行方式。
2. \`import ...\`——「借用工具」。
3. \`def main():\`——「步骤写在这里」。
4. \`print(...)\`——「显示点什么」。
5. \`if __name__ == "__main__":\`——「直接运行本文件时才执行 main」。

第 0 部分不要求你发明新程序。会读、会跑就够。

## 对应代码

\`ep00_check_setup.py\` 的 \`main\` 只做两件事：打印 Python 版本、检查 \`torch\` 能否导入。

## 现在就做

1. 打开文件。
2. 光标移到 \`def main\`。
3. 找到最后的 \`OK\` 打印。
4. 关掉文件——你已经「读过」了。

## 如果翻车了

- 编辑器乱码：用 UTF-8 纯文本打开，不要当 Word 文档。
- 怕改坏：文件很小；用 git 的话 \`git checkout -- 文件\` 可恢复。

## 先别深挖

类、类型注解、PyTorch 内部。网站会先讲清楚概念。

## 下一章

**旅程地图**（\`?c=start-map\`）。
`,
  },
  {
    id: 'start-map',
    ep: '00f',
    title: '旅程地图',
    summary: '第 1–4 部分一眼看懂；Llama 3.2 谱系；何时需要 Hugging Face 登录。',
    script: '',
    partId: 'start',
    estimatedMinutes: 10,
    markdown: `## 一句话

地图已经清楚：主干 → 更快推理 → 小规模训练 → 前沿引物。

## 开始之前

- 第 0 部分其余章节读完。
- 约 10 分钟。

## 用人话讲

**Llama 3.2 谱系**的意思是：我们跟的是现代 dense 解码器的「零件清单」（RMSNorm、RoPE、SwiGLU、GQA），再按需看 MoE / MLA 等前沿想法。

\`\`\`mermaid
flowchart TB
  p0[第 0 部分 开场] --> p1[第 1 部分 Dense 主干]
  p1 --> p2[第 2 部分 更快推理]
  p2 --> p3[第 3 部分 小规模训练]
  p3 --> p4[第 4 部分 前沿]
\`\`\`

**Hugging Face 登录**只在需要官方分词器/权重（门禁模型）时才要。很多章节用随机小张量，不用下载。

## 对应代码

库在 \`arche/\`，章节调用它；网站负责叙述。

## 现在就做

1. 把第 0 部分都标成已读。
2. 点 **下一章** 进入第 1 部分「总览」。

## 如果翻车了

- 被 MoE、MLA 名字吓到：等到第 4 部分再说；第 1 部分不需要它们。

## 先别深挖

复现工业级训练。本课要的是理解与可运行的小零件。

## 下一章

第 1 部分开始：**总览**（\`?c=overview\`）。
`,
  },
];
