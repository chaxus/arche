#!/usr/bin/env python3
"""Inject web/src/content/beginner/{en,cn}/<id>.md into chapter markdown fields."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEGINNER = ROOT / "web/src/content/beginner"


def escape_ts_template(md: str) -> str:
    # Content lives inside TypeScript template literals.
    return md.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")


def patch_file(ts_path: Path, locale: str, only: set[str] | None = None) -> list[str]:
    text = ts_path.read_text(encoding="utf-8")
    updated: list[str] = []
    for md_path in sorted((BEGINNER / locale).glob("*.md")):
        chapter_id = md_path.stem
        if only is not None and chapter_id not in only:
            continue
        body = escape_ts_template(md_path.read_text(encoding="utf-8").rstrip() + "\n")

        pattern = re.compile(
            rf"(id:\s*'{re.escape(chapter_id)}'[\s\S]*?markdown:\s*`)([\s\S]*?)(\n[ \t]*`,)",
            re.M,
        )

        def repl(match: re.Match[str]) -> str:
            return match.group(1) + body + match.group(3)

        new_text, n = pattern.subn(repl, text, count=1)
        if n != 1:
            raise SystemExit(f"failed to patch {locale}:{chapter_id} in {ts_path}")
        text = new_text
        updated.append(chapter_id)
    ts_path.write_text(text, encoding="utf-8")
    return updated


def main() -> None:
    only = set(sys.argv[1:]) or None
    en = patch_file(ROOT / "web/src/content/en.ts", "en", only)
    cn = patch_file(ROOT / "web/src/content/cn.ts", "cn", only)
    print("EN", ",".join(en))
    print("CN", ",".join(cn))


if __name__ == "__main__":
    main()
