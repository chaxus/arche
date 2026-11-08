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
  brandTagline: 'ἀρχή — modern dense from first principles',
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
  markDone: 'Mark as read',
  markedDone: 'Read',
  prevChapter: 'Previous',
  nextChapter: 'Next',
  minutesLabel: 'min',
  partLabels: {
    start: 'Part 0 — Start here',
    trunk: 'Part 1 — The dense trunk',
    infer: 'Part 2 — Faster inference',
    train: 'Part 3 — Tiny training',
    frontier: 'Part 4 — Frontier primers',
  },
  homeNav: 'Home',
  homeTitle: 'Learn a modern dense LLM by writing it',
  homeLede:
    'Absolute beginners start in Part 0. Returners jump back with Continue. No quizzes — read, run, and use the pictures.',
  continueLabel: 'Continue',
  continueStart: 'Start Part 0',
  homePartsHeading: 'Path',
  homeDoneLabel: 'read',
};

const CN: UiCopy = {
  brandTagline: 'ἀρχή — 从第一性原理手写现代 dense',
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
  markDone: '标记已读',
  markedDone: '已读',
  prevChapter: '上一章',
  nextChapter: '下一章',
  minutesLabel: '分钟',
  partLabels: {
    start: '第 0 部分 — 从这里开始',
    trunk: '第 1 部分 — Dense 主干',
    infer: '第 2 部分 — 更快推理',
    train: '第 3 部分 — 小规模训练',
    frontier: '第 4 部分 — 前沿引物',
  },
  homeNav: '首页',
  homeTitle: '用手写学会现代 dense 大模型',
  homeLede: '零基础从第 0 部分开始；回访点「继续」。没有章末测验——读、跑、看图即可。',
  continueLabel: '继续',
  continueStart: '开始第 0 部分',
  homePartsHeading: '路径',
  homeDoneLabel: '已读',
};

export const uiFor = (locale: LocaleDef): UiCopy => (locale.id === 'cn' ? CN : EN);
