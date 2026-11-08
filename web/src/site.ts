/** Site constants and URL helpers — same single-mapping idea as ran docs `build/urls.ts`. */

import { LOCALES, ROOT_LOCALE, localeHref, type LocaleDef } from './i18n/locales';

export const ORIGIN = 'https://arche.chaxus.com';

export const SITE = {
  name: 'arche',
  author: 'chaxus',
  authorUrl: 'https://chaxus.com/',
  github: 'https://github.com/chaxus/arche',
  descriptionEn:
    'From zero toward frontier-ready: hand-write a Llama 3.2–style LLM, train or load weights, serve faster, then map MoE/MLA.',
  descriptionCn:
    '从零对接前沿：亲手写出 Llama 3.2 风格大模型主干，训练或加载权重，服务加速，再摸 MoE/MLA 地图。',
} as const;

export const HOME_ID = 'home';

/** Absolute URL, no trailing slash on origin. */
export const absoluteUrl = (path: string): string => {
  if (path.startsWith('http')) return path;
  return `${ORIGIN}${path.startsWith('/') ? path : `/${path}`}`;
};

/**
 * Canonical public path for home or a chapter.
 * Home: `/` or `/cn/`. Chapter: `/tokenizer/` or `/cn/tokenizer/` (trailing slash → index.html).
 */
export const pagePath = (chapterId: string, locale: LocaleDef = ROOT_LOCALE): string => {
  if (chapterId === HOME_ID) {
    return locale.dir ? `/${locale.dir}/` : '/';
  }
  const base = localeHref(`/${chapterId}/`, locale);
  return base.endsWith('/') ? base : `${base}/`;
};

export const pageUrl = (chapterId: string, locale: LocaleDef = ROOT_LOCALE): string =>
  absoluteUrl(pagePath(chapterId, locale));

/** Parse locale + chapter from pathname. Unknown segments → home. */
export const routeFromPathname = (
  pathname: string,
  chapterIds: ReadonlySet<string>,
): { locale: LocaleDef; chapterId: string } => {
  const locale =
    LOCALES.find((item) => item.dir && pathname === `/${item.dir}`) ||
    LOCALES.find((item) => item.dir && pathname.startsWith(`/${item.dir}/`)) ||
    ROOT_LOCALE;

  let rest = pathname;
  if (locale.dir) {
    rest = pathname.slice(`/${locale.dir}`.length) || '/';
  }
  const seg = rest.replace(/^\/+|\/+$/g, '').split('/')[0] ?? '';
  if (!seg) return { locale, chapterId: HOME_ID };
  if (chapterIds.has(seg)) return { locale, chapterId: seg };
  return { locale, chapterId: HOME_ID };
};

export { LOCALES, ROOT_LOCALE };
