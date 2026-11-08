/**
 * Locale registry — same idea as ran docs: one list every i18n path reads.
 * English is the root locale (no URL prefix); Chinese lives under `/cn/`.
 */

export type LocaleDir = '' | 'cn';

export type LocaleDef = {
  id: 'root' | 'cn';
  dir: LocaleDir;
  lang: string;
  label: string;
  ogLocale: string;
};

/** Menu order: Chinese then English (author-facing), matching ran.chaxus.com. */
export const LOCALES: readonly LocaleDef[] = [
  { id: 'cn', dir: 'cn', lang: 'zh-CN', label: '简体中文', ogLocale: 'zh_CN' },
  { id: 'root', dir: '', lang: 'en', label: 'English', ogLocale: 'en_US' },
];

export const ROOT_LOCALE: LocaleDef = LOCALES.find((locale) => locale.dir === '')!;

export const prefixOf = (locale: LocaleDef): string => (locale.dir ? `/${locale.dir}` : '');

export const localeFromUrlPath = (pathname: string): LocaleDef => {
  const segment = pathname.replace(/^\/+/, '').split('/')[0] ?? '';
  return LOCALES.find((locale) => locale.dir && locale.dir === segment) ?? ROOT_LOCALE;
};

export const localeHref = (path: string, locale: LocaleDef): string => {
  const normalized = path.startsWith('/') ? path : `/${path}`;
  if (!locale.dir) return normalized === '/' ? '/' : normalized;
  if (normalized === '/') return `/${locale.dir}/`;
  return `/${locale.dir}${normalized}`;
};
