import type { LocaleDef } from './locales';
import type { PartId } from '../content/types';

export type UiCopy = {
  brandTagline: string;
  chaptersNav: string;
  lessonHeading: string;
  companionPython: string;
  siteLabel: string;
  github: string;
  langMenu: string;
  progressLabel: string;
  demoHeading: string;
  demoChatHint: string;
  demoChatRun: string;
  demoBatchHint: string;
  demoBatchRun: string;
  demoTrainHint: string;
  demoTrainRun: string;
  demoVizHeading: string;
  walkthroughHeading: string;
  sourceHeading: string;
  sourceLibraryTab: string;
  sourceLabTab: string;
  sourceEmpty: string;
  sourceGithub: string;
  fixHeading: string;
  markDone: string;
  markedDone: string;
  prevChapter: string;
  nextChapter: string;
  minutesLabel: string;
  partLabels: Record<PartId, string>;
  homeNav: string;
  homeTitle: string;
  homeLede: string;
  continueLabel: string;
  continueStart: string;
  homePartsHeading: string;
  homeDoneLabel: string;
};

const EN: UiCopy = {
  brandTagline: 'ἀρχή — build an LLM by hand, from zero',
  chaptersNav: 'Chapters',
  lessonHeading: 'Lesson',
  companionPython: 'Companion Python',
  siteLabel: 'Site',
  github: 'GitHub',
  langMenu: 'Language',
  progressLabel: 'Progress',
  demoHeading: 'Try it',
  demoChatHint: 'Render a tiny chat transcript the way ep16 falls back without a HF template.',
  demoChatRun: 'Render template',
  demoBatchHint: 'Three decode jobs share steps until each hits max_new_tokens — the ep17 scheduler shape.',
  demoBatchRun: 'Run scheduler',
  demoTrainHint: 'Toy pretrain then SFT/DPO: loss numbers are illustrative, not a real training run.',
  demoTrainRun: 'Simulate steps',
  demoVizHeading: 'Interactive picture',
  walkthroughHeading: 'In the source',
  sourceHeading: 'Code',
  sourceLibraryTab: 'Library',
  sourceLabTab: 'Lab script',
  sourceEmpty: 'No source for this chapter.',
  sourceGithub: 'Open on GitHub',
  fixHeading: 'If something breaks',
  markDone: 'Mark as read',
  markedDone: 'Read',
  prevChapter: 'Previous',
  nextChapter: 'Next',
  minutesLabel: 'min',
  partLabels: {
    start: 'Part 0 — Start here',
    trunk: 'Part 1 — Build the dense trunk',
    train: 'Part 2 — Train from zero (write weights)',
    infer: 'Part 3 — Generate & serve faster',
    frontier: 'Part 4 — Frontier structures (MoE / MLA…)',
  },
  homeNav: 'Home',
  homeTitle: 'See what an LLM is doing—by writing one',
  homeLede:
    'From zero toward frontier-ready—one path. Hand-write the trunk, train weights, serve faster, then MoE/MLA maps. Llama 3.2 ~1B is the running example. Continue picks up where you left off.',
  continueLabel: 'Continue',
  continueStart: 'Start Part 0',
  homePartsHeading: 'Path',
  homeDoneLabel: 'read',
};

const CN: UiCopy = {
  brandTagline: 'ἀρχή — 从零开始，亲手写出大模型',
  chaptersNav: '章节',
  lessonHeading: '课程',
  companionPython: '配套 Python',
  siteLabel: '站点',
  github: 'GitHub',
  langMenu: '语言',
  progressLabel: '进度',
  demoHeading: '动手试',
  demoChatHint: '用 ep16 的回退模板渲染一小段对话（无需 HF chat_template）。',
  demoChatRun: '渲染模板',
  demoBatchHint: '三个解码任务共享 step，直到各自打满 max_new_tokens——对应 ep17 调度形状。',
  demoBatchRun: '跑调度器',
  demoTrainHint: '玩具 pretrain → SFT/DPO：数字仅示意，不是真实训练。',
  demoTrainRun: '模拟几步',
  demoVizHeading: '互动示意',
  walkthroughHeading: '源码对照',
  sourceHeading: '代码',
  sourceLibraryTab: '库实现',
  sourceLabTab: '实验脚本',
  sourceEmpty: '本章暂无源码。',
  sourceGithub: '在 GitHub 打开',
  fixHeading: '翻车了怎么办',
  markDone: '标记已读',
  markedDone: '已读',
  prevChapter: '上一章',
  nextChapter: '下一章',
  minutesLabel: '分钟',
  partLabels: {
    start: '第 0 部分 — 从这里开始',
    trunk: '第 1 部分 — 写出 dense 本体',
    train: '第 2 部分 — 从零训练（写出权重）',
    infer: '第 3 部分 — 生成与服务加速',
    frontier: '第 4 部分 — 前沿结构（MoE / MLA…）',
  },
  homeNav: '首页',
  homeTitle: '亲手写出一个大模型，看懂它在算什么',
  homeLede:
    '从零对接前沿——一条路。写出主干 → 训练写出权重 → 服务加速 → MoE/MLA 地图。Llama 3.2 ~1B 是贯穿例子。回访点「继续」。',
  continueLabel: '继续',
  continueStart: '开始第 0 部分',
  homePartsHeading: '路径',
  homeDoneLabel: '已读',
};

export const uiFor = (locale: LocaleDef): UiCopy => (locale.id === 'cn' ? CN : EN);
